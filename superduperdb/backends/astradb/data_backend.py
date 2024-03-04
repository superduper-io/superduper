from superduperdb.backends.base.data_backend import BaseDataBackend
from astrapy.db import AstraDB
from .metadata import AstraMetaDataStore
from superduperdb.backends.local.artifacts import FileSystemArtifactStore


class AstraDataBackend(BaseDataBackend):
    """
    Data backend for AstraDB.

    :param conn: AstraDB client connection
    :param name: Name of database to host filesystem
    """

    id_field = '_id'
    astra_api_v2_base_url = 'https://api.astra.datastax.com/v2'

    def __init__(self, conn: AstraDB, name: str):
        super().__init__(conn=conn, name=name)
        self._db = conn

    @property
    def db(self):
        return self._db

    @property
    def authorization_headers(self):
        return {
            'Authorization': f'Bearer {self.db.token}',
            'Content-Type': 'application/json'
        }

    @property
    def database_id(self):
        return '-'.join(self.db.base_url.split('//')[1].split('-')[:5])

    def url(self):
        """
        Databackend connection url
        """
        return self.db.base_url

    def build_metadata(self):
        """
        Build a default metadata store based on current connection.
        """
        return AstraMetaDataStore(self.conn, self.name)

    def build_artifact_store(self):
        """
        Build a default artifact store based on current connection.
        """
        return FileSystemArtifactStore(conn='.superduperdb/artifacts/', name='astra')

    def get_table_or_collection(self, identifier):
        return self.db.collection(identifier)

    def drop(self, force: bool = False):
        """
        Drop the databackend.
        The token to be used must be for Organization Administrator role
        It uses Astra DevOps API
        """
        try:
            if not force:
                if not click.confirm(
                        f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                        f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                        'Are you sure you want to drop the data-backend? ',
                        default=False,
                ):
                    logging.warn('Aborting...')
            # Send a DELETE request to delete the database
            response = self.db.client.post(f'{self.astra_api_v2_base_url}/databases/{self.database_id}/terminate',
                                           headers=self.authorization_headers)

            # Check if the deletion was successful
            if response.status_code == 202:
                print('Database deleted successfully')
                return response
            else:
                try:
                    error_data = json.loads(response.text)
                    if 'errors' in error_data:
                        error_messages = error_data['errors']
                        # Print each error message
                        for error in error_messages:
                            print(error.get('message', 'Unknown error'))
                    else:
                        print('No error messages found in the response')
                except json.JSONDecodeError as e:
                    print(f'Error decoding Errors JSON: {e}')
                except KeyError as e:
                    print(f'Error accessing key for errors: {e}')
                except Exception as e:
                    print(f'An error occurred: {e}')
            return response
        except Exception as e:
            print(f'An error occurred while dropping the database: {e}')

    def disconnect(self):
        """
        Disconnect the client
        """
        pass
