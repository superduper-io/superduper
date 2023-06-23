from superduperdb import JSONable

from superduperdb.server.server import Server


class ContentResult(JSONable):
    pass


server = Server()


class SomeDatabase:
    @server.register
    def read_content(self, artifact_key: str) -> ContentResult:
        pass


server.cfg.web_server.host = '0.0.0.0'
server.run(SomeDatabase())
