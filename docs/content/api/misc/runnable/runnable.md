**`superduperdb.misc.runnable.runnable`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/runnable/runnable.py)

## `Event` 

```python
Event(self,
     *on_set: Callable[[],
     NoneType])
```
| Parameter | Description |
|-----------|-------------|
| on_set | Callbacks to call when the event is set |

An Event that calls a list of callbacks when set or cleared.

A threading.Event that also calls back to zero or more functions when its state
is set or reset, and has a __bool__ method.

Note that the callback might happen on some completely different thread,
so these functions cannot block

## `Runnable` 

```python
Runnable(self)
```
A base class for things that start, run, finish, stop and join.

Stopping is requesting immediate termination: finishing is saying that
there is no more work to be done, finish what you are doing.

A Runnable has two `Event`s, `running` and `stopped`, and you can either
`wait` on either of these conditions to be true, or add a callback function
(which must be non-blocking) to either of them.

`running` is not set until the setup for a `Runnable` has finished;
`stopped` is not set until all the computations in a thread have ceased.

An Runnable can be used as a context manager:

with runnable:
# The runnable is running by this point
do_stuff()
# By the time you get to here, the runnable has completely stopped

The above means roughly the same as

runnable.start()
try:
do_stuff()
runnable.finish()
runnable.join()
finally:
runnable.stop()

