from superduperdb.misc.runnable import thread


def test_has_thread():
    result = []

    ht = thread.HasThread()

    def callback():
        result.append(0)
        if len(result) >= 4:
            ht.stop()

    ht.callback = callback

    with ht:
        pass

    assert ht.stopped
    assert result == [0]
