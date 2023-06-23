import io

import requests


class TestServer:
    base_url: str = 'http://localhost:3223'

    def test_get_health(self):
        """Test that the server is healthy."""
        url = f'{self.base_url}/health'

        response = requests.get(url)

        assert response.status_code == 200
        assert response.content.decode() == 'ok'

    def test_post_execute(self):
        """Test that the server allows the execution of functions and."""

        token = 'test-key'
        blob = bytes(range(256))
        fp = io.BytesIO(blob)
        files = {
            'file': (token, fp, 'multipart/form-data')}

        requests.post(f'{self.base_url}/upload/{token}', files=files)

        response = requests.post(f'{self.base_url}/read_content/{token}')
