import os


class CallableWithSecret:
    """
    Class allowing functions to use secrets which are then saved separately/ or potentially
    not at all.
    """
    def __init__(self, secrets):
        """
        :param secrets: dictionary of secrets - are added as environment variables
        """
        self._secrets = None
        self.secrets = secrets

    @property
    def secrets(self):
        return self._secrets

    @secrets.setter
    def secrets(self, value):
        self._secrets = value
        if value is None:
            return
        self._set_envs()

    def _set_envs(self):
        for k in self.secrets:
            os.environ[k] = self.secrets[k]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError