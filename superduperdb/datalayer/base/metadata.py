import time
from typing import Dict, Any
from abc import ABC, abstractmethod


class MetaDataStore(ABC):
    """
    Abstraction for storing meta-data separately from primary data.
    """

    @abstractmethod
    def create_component(self, info: Dict):
        pass

    @abstractmethod
    def get_job(self, job_id: str):
        pass

    @abstractmethod
    def update_job(self, job_id: str, key: str, value: Any):
        pass

    @abstractmethod
    def watch_job(self, identifier: str):
        try:
            status = 'pending'
            n_lines = 0
            n_lines_stderr = 0
            while status in {'pending', 'running'}:
                r = self.get_job(identifier)
                status = r['status']
                if status == 'running':
                    if len(r['stdout']) > n_lines:
                        print(''.join(r['stdout'][n_lines:]), end='')
                        n_lines = len(r['stdout'])
                    if len(r['stderr']) > n_lines_stderr:
                        print(''.join(r['stderr'][n_lines_stderr:]), end='')
                        n_lines_stderr = len(r['stderr'])
                    time.sleep(0.2)
                else:
                    time.sleep(0.2)
            r = self.get_job(identifier)
            if status == 'success':
                if len(r['stdout']) > n_lines:
                    print(''.join(r['stdout'][n_lines:]), end='')
                if len(r['stderr']) > n_lines_stderr:
                    print(''.join(r['stderr'][n_lines_stderr:]), end='')
            elif status == 'failed':  # pragma: no cover
                print(r['msg'])
        except KeyboardInterrupt:  # pragma: no cover
            return

    @abstractmethod
    def list_jobs(self):
        pass

    @abstractmethod
    def list_components(self, variety: str):
        pass

    @abstractmethod
    def delete_component(self, identifier: str, variety: str):
        pass

    @abstractmethod
    def get_metadata(self, key):
        pass

    @abstractmethod
    def _get_object(self, identifier: str, variety: str):
        pass

    def get_object(self, identifier: str, variety: str):
        r = self._get_object(identifier, variety)
        if r is None:
            raise FileNotFoundError(f'Object {identifier} does not exist in metadata')
        return r

    @abstractmethod
    def update_object(self, identifier, variety, key, value):
        pass

    @abstractmethod
    def write_output_to_job(self, identifier, msg, stream):
        pass
