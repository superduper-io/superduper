from superduper.base.logger import Logging


def test_logging_integration():
    log = Logging()
    log.info("This is an info message")
    log.warn("This is a warning message")
    log.error("This is an error message")
    log.exception("This is an exception", Exception("Test exception"))


if __name__ == "__main__":
    test_logging_integration()
