from superduperdb import JSONable


from superduperdb.server.server import Server


class BaseJSON(JSONable):
    pass


class SomeResult(JSONable):
    pass


class One(BaseJSON):
    pass


class Two(BaseJSON):
    pass


class Other(BaseJSON):
    pass


server = Server()


class SomeDatabase:
    @server.register
    def method_to_expose(self, one: One, two: Two) -> SomeResult:
        return SomeResult()

    @server.register
    def method_two(self) -> Other:
        return Other()


server.run(SomeDatabase())
