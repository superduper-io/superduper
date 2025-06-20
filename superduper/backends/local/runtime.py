import os
from dataclasses import dataclass
from typing import Dict, List

from dataclasses_json import dataclass_json

from superduper import logging
from superduper.backends.base import runtime
from superduper.backends.base.runtime import RuntimeEnvironment
from superduper.base import exceptions
from superduper.base.status import STATUS_RUNNING, STATUS_PENDING, STATUS_FAILED


#-----------------------------------------------------
# Format of local integrations
#-----------------------------------------------------
@dataclass_json
@dataclass
class ObjectMeta:
    """Kubernetes ObjectMeta equivalent."""
    name: str

@dataclass_json
@dataclass
class IntegrationSpec:
    """IntegrationSpec defines the desired state of an integration."""
    family: str
    provider: str
    credentials: Dict[str, str]

@dataclass_json
@dataclass
class Integration:
    """Integration is the Schema for the Integrations API."""
    metadata: ObjectMeta
    spec: IntegrationSpec

@dataclass_json
@dataclass
class IntegrationList:
    items: List[Integration]


# Collection of example integrations for different providers and use cases
SUPPORTED_INTEGRATIONS = [
    # Snowflake data warehouse integration
    Integration(
        metadata=ObjectMeta(
            name="databackend",
        ),
        spec=IntegrationSpec(
            family="databackend",
            provider="generic-connector",
            credentials={
                "databackend": "",
            }
        )
    ),

    # AWS S3 storage integration
    Integration(
        metadata=ObjectMeta(
            name="aws",
        ),
        spec=IntegrationSpec(
            family="datasource",
            provider="aws",
            credentials={
                "aws-access-key-id": "",
                "aws-secret-access-key": "",
            }
        )
    ),

    # OpenAI
    Integration(
        metadata=ObjectMeta(
            name="openai",
        ),
        spec=IntegrationSpec(
            family="llm",
            provider="openai",
            credentials={
                "openai-api-key": "",
            }
        )
    ),
]

#-----------------------------------------------------
# Helper functions
#-----------------------------------------------------

def fetch_integrations(
        required_integrations: List[str]
) -> List[Integration]:
    """
    Check if the required integrations are supported
    """
    if not required_integrations:
        logging.warn("No required integrations specified")
        return []


    # Build lookup map for all integrations
    all_integrations_map = {}
    for integration in SUPPORTED_INTEGRATIONS:
        integration_name = integration.metadata.name
        all_integrations_map[integration_name] = integration

    # Filter to only required integrations and validate they exist
    required_integrations_list = []
    missing_integrations = []

    for required_integration in required_integrations:
        integration = all_integrations_map.get(required_integration)
        if not integration:
            missing_integrations.append(required_integration)
        else:
            required_integrations_list.append(integration)

    # Raise error if any required integrations are missing
    if missing_integrations:
        available_integrations = list(all_integrations_map.keys())
        raise exceptions.NotFound(
            "integration",
            f"Missing required integrations: {missing_integrations}. "
            f"Available integrations: {available_integrations}"
        )

    logging.info(f"Successfully fetched {len(required_integrations_list)} required integrations")
    return required_integrations_list

def check_local_integration_status(secrets_dir: str, integration: Integration) -> runtime.IntegrationStatus:
    for secret_name, _ in integration.spec.credentials.items():
        # Ensure that secrets are properly mounted
        secret_file_path = os.path.join(secrets_dir, secret_name, 'secret_string')

        if not os.path.isfile(secret_file_path):
            logging.warn(f"Warning: No 'secret_string' file found in {secret_file_path}.")
            return runtime.IntegrationStatus(
                Name=integration.metadata.name.lower(),
                Phase=STATUS_FAILED,
                Reason="NotFound",
                Msg=f"Unable to find secret '{secret_name}' on local secrets directory",
            )

        with open(secret_file_path, 'r') as file:
            local_value = file.read().strip()

        # Ensure that keys are not empty
        if not local_value:
            return runtime.IntegrationStatus(
                Name=integration.metadata.name.lower(),
                Phase=STATUS_PENDING,
                Reason="EmptyValue",
                Msg=f"Secret '{secret_name}' was found, but the value is empty",
            )

    return runtime.IntegrationStatus(
        Name=integration.metadata.name.lower(),
        Phase=STATUS_RUNNING,
        Reason="InSync",
        Msg=f"all secrets are mounted correctly",
    )

def load_local_secrets(secrets_dir: str, integration: Integration) -> None:
    for secret_name, _ in integration.spec.credentials.items():
        # Ensure that secrets are properly mounted
        secret_file_path = os.path.join(secrets_dir, secret_name, 'secret_string')

        if not os.path.isfile(secret_file_path):
            logging.warn(f"Warning: No 'secret_string' file found in {secret_file_path}.")
            return

        with open(secret_file_path, 'r') as file:
            local_value = file.read().strip()

        # Convert key to uppercase and replace hyphens with underscores
        env_key = secret_name.replace("-", "_").upper()
        os.environ[env_key] = local_value
        logging.info(f'Successfully loaded secret {env_key} into environment.')


#-----------------------------------------------------
# Local Runtime Implementation
#-----------------------------------------------------
@dataclass
class RuntimeEnvironmentParams:
    pass

class LocalRuntimeEnvironment(RuntimeEnvironment):
    def __init__(self, secrets_dir:str, params: RuntimeEnvironmentParams):
        # Secrets
        if not os.path.isdir(secrets_dir):
            raise exceptions.BadRequest(f"The path '{secrets_dir}' is not a valid secrets directory.")
        self._secrets_dir = secrets_dir

        # other params
        self.params = params


    def name(self) -> str:
        return "local"

    def check_integrations(self, required_integrations: List[str]) -> runtime.IntegrationStatusList:
        # Fetch required integrations
        integrations_list = fetch_integrations(
            required_integrations
        )

        # Process all required integrations
        integration_statuses = []
        for integration in integrations_list:
            try:
                # Check the local status of the integration
                integration_status = check_local_integration_status(self._secrets_dir, integration)
                integration_statuses.append(integration_status)

                logging.debug(f"Successfully checked status for integration '{integration.metadata.name}'")

            except Exception as e:
                logging.error(f"Failed to check status for integration '{integration.metadata.name}': {e}")
                # Depending on requirements, you might want to raise here instead
                continue

        logging.info(f"Checked status for {len(integration_statuses)} integrations")
        return runtime.IntegrationStatusList(Items=integration_statuses)


    def load_integrations(self, required_integrations: List[str]) -> None:
        # Fetch required integrations
        integrations_list = fetch_integrations(
            required_integrations
        )

        # Process all required integrations
        loaded_count = 0

        for integration in integrations_list:
            try:
                # Load the local secrets for the integration
                load_local_secrets(self._secrets_dir, integration)
                loaded_count += 1

                logging.debug(f"Successfully loaded secrets for integration '{integration.metadata.name}'")
            except Exception as e:
                logging.error(f"Failed to load secrets for integration '{integration.metadata.name}': {e}")
                # Depending on requirements, you might want to raise here instead
                continue

        logging.info(f"Successfully loaded {loaded_count}/{len(integrations_list)} integrations")
