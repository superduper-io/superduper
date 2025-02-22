import concurrent.futures
import hashlib
import os
import traceback

from superduper import CFG, logging
from superduper.base.exceptions import IncorrectSecretException, MissingSecretsException


def check_openai(name):
    """Check if the OpenAI credentials are correct.

    :param name: The name of the secret
    """
    from openai import AuthenticationError, Client, OpenAIError

    try:
        _ = Client().models.list()
    except AuthenticationError:
        raise IncorrectSecretException(
            'OpenAI API key is incorrect. Please check the key and try again.'
        )
    except OpenAIError:
        raise MissingSecretsException(
            'OpenAI API key is missing. Set as enviroment variable OPENAI_API_KEY.'
        )


def check_s3(name):
    """Check if the AWS credentials are correct.

    :param name: The name of the secret
    """
    if name == 'AWS_ACCESS_KEY_ID':
        assert 'AWS_SECRET_ACCESS_KEY' in os.environ, 'AWS_SECRET_ACCESS_KEY not found'
    elif name == 'AWS_SECRET_ACCESS_KEY':
        assert 'AWS_ACCESS_KEY_ID' in os.environ, 'AWS_ACCESS_KEY_ID not found'
    else:
        raise ValueError(f'Unknown secret {name}')

    import boto3
    from botocore.exceptions import ClientError

    try:
        session = boto3.Session(
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        )
        s3_client = session.client('s3')
        s3_client.list_buckets()
    except KeyError:
        raise MissingSecretsException(
            (
                'AWS credentials are missing. Set as environment'
                'variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.'
            )
        )
    except ClientError as e:
        if 'does not exist' in str(e):
            raise IncorrectSecretException(
                'AWS credentials are incorrect. '
                'Please check the credentials and try again. '
                'The provided AWS key-pair does not exist.'
            )
        elif 'not authorized' in str(e):
            raise IncorrectSecretException(
                'AWS credentials are incorrect. Please check the credentials and '
                'try again. '
                'The provided AWS key-pair is not authorized to access '
                'the requested resources. '
                'Minimum required permissions: s3:ListBucket, s3:GetObject.'
            )
        logging.error(traceback.format_exc())
        raise e
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e


def not_empty(name):
    """Check if the secret is not empty.

    :param name: The name of the secret
    """
    assert name in os.environ, f'{name} not found in environment'


class MatchersFactory:
    """Matchers factory for checking secrets."""

    def __getitem__(self, key):
        if key.startswith('AWS_ACCESS_KEY_ID'):
            return lambda: check_s3(key)
        if key == ('OPENAI_API_KEY'):
            return lambda: check_openai(key)
        return lambda: not_empty(key)


MATCHERS = MatchersFactory()


def check_secrets():
    """Check that the secrets connect."""
    required = os.environ.get('SUPERDUPER_REQUIRED_SECRETS', '').split(',')
    logging.info(f'Checking secrets {required}')
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for secret in required:
            future = executor.submit(MATCHERS[secret])
            try:
                future.result(timeout=3)
            except concurrent.futures.TimeoutError:
                errors.append(f'Timeout for secret: {secret} after 3 seconds; '
                              'check your connectivity and secret(s).')
            except Exception as e:
                errors.append(e)

    if errors:
        raise IncorrectSecretException(
            'Some secrets are incorrect. Please check the secrets and try again.\n'
            '\n'.join([str(e) for e in errors])
        )


def load_secrets():
    """Load secrets directory into env vars."""
    secrets_dir = CFG.secrets_volume

    if not os.path.isdir(secrets_dir):
        raise ValueError(f"The path '{secrets_dir}' is not a valid secrets directory.")

    for key_dir in os.listdir(secrets_dir):
        key_path = os.path.join(secrets_dir, key_dir)

        if not os.path.isdir(key_path):
            continue

        secret_file_path = os.path.join(key_path, 'secret_string')

        if not os.path.isfile(secret_file_path):
            logging.warn(f"Warning: No 'secret_string' file found in {key_path}.")
            continue

        with open(secret_file_path, 'r') as file:
            content = file.read().strip()
        os.environ[key_dir] = content
        logging.info(f'Successfully loaded secret {key_dir} into environment.')


def get_file_from_uri(uri):
    """
    Get file name from uri.

    >>> _get_file('file://test.txt')
    'test.txt'
    >>> _get_file('http://test.txt')
    '414388bd5644669b8a92e45a96318890f6e8de54'

    :param uri: The uri to get the file from
    """
    if uri.startswith('file://'):
        file = uri[7:]
    elif (
        uri.startswith('http://')
        or uri.startswith('https://')
        or uri.startswith('s3://')
    ):
        file = f'{CFG.downloads.folder}/{hashlib.sha1(uri.encode()).hexdigest()}'
    else:
        raise NotImplementedError(f'File type of {uri} not supported')
    return file
