from superduperdb.datalayer.base import build
from superduperdb.server.make import make_server
import requests
import superduperdb as s
import threading
import time
import typing as t
import uvicorn


class ServerThread:
    def __init__(self, db: t.Any):
        super().__init__()

        self.server = make_server(db)

        self.cfg = s.CFG.server.web_server.deepcopy()
        self.cfg.port = s.CFG.server.test_port
        cfg = self.cfg.dict()
        cfg.pop('protocol')

        self.uvicorn = uvicorn.Server(uvicorn.Config(self.server.app, **cfg))

        self.thread = threading.Thread(target=self.uvicorn.run)
        self.thread.start()

        while not self.uvicorn.started:
            time.sleep(0.001)

    def stop(self):
        self.uvicorn.should_exit = True
        self.thread.join()


def test_start_stop():
    db = build.build_datalayer()
    st = ServerThread(db)
    assert requests.get(f'{st.cfg.uri}/health').text == 'ok'
    st.stop()


if __name__ == '__main__':
    test_start_stop()
