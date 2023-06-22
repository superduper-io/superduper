from superduperdb.server.server import Server


class SomeResult:
    pass


class One:
    pass


class Two:
    pass


class Other:
    pass


class SomeDatabase:
    def method_to_expose(self, one: One, two: Two) -> SomeResult:
        return SomeResult()

    def method_two(self) -> Other:
        return Other()


Server().auto_run(SomeDatabase())
