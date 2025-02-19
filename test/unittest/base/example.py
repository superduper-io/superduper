from superduper.base.base import imported


@imported
class MyClass:
    def __init__(self, value):
        self.value = value

    def process(self, x):
        return self.value + x
