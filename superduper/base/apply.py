import json
import time
import typing as t

import click
from rich.console import Console

from superduper import CFG, Component, logging
from superduper.base import exceptions
from superduper.base.document import Document
from superduper.base.event import Create, Signal, Update
from superduper.components.component import running_status
from superduper.misc.tree import dict_to_tree

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.metadata import Job

_WAIT_TIMEOUT = 60


class HashingError(Exception):
    """Raised when hashing is not deterministic.

    # noqa
    """

    pass


def apply(
    db: 'Datalayer',
    object: t.Union['Component', t.Sequence[t.Any], t.Any],
    force: bool | None = None,
    wait: bool = False,
    jobs: bool = True,
) -> 'Component':
    """
    Add functionality in the form of components.

    Components are stored in the configured artifact store
    and linked to the primary database through metadata.

    :param db: Datalayer instance
    :param object: Object to be stored.
    :param force: List of jobs which should execute before component
                  initialization begins.
    :param wait: Blocks execution till create events finish.
    :param jobs: Whether to execute jobs or not.
    """
    if not isinstance(object, Component):
        raise ValueError('Only components can be applied')

    if object.uuid != object.uuid:
        raise HashingError(
            'The component you specified did not yield a deterministic hash. '
            'This is a requirement for the object to be stored in the '
            'system. Modify your classes and check `object.uuid`.'
        )

    if force is None:
        force = db.cfg.force_apply

    # This holds a record of the changes
    diff: t.Dict = {}
    # context allows us to track the origin of the component creation
    create_events, job_events = _apply(
        db=db,
        object=object,
        context=object.uuid,
        job_events={},
        global_diff=diff,
        non_breaking_changes={},
    )

    if not jobs:
        job_events = {}

    # this flags that the context is not needed anymore
    if not create_events:
        logging.info('No changes needed, doing nothing!')
        return object

    if diff:
        logging.info('Found this diff:')
        Console().print(dict_to_tree(diff, root=object.identifier), soft_wrap=True)

    logging.info('Found these changes and/ or additions that need to be made:')

    logging.info('-' * 100)
    logging.info('METADATA EVENTS:')
    logging.info('-' * 100)

    steps = {c.data['uuid']: str(i) for i, c in enumerate(create_events.values())}

    for i, c in enumerate(create_events.values()):
        if c.parent:
            try:
                logging.info(
                    f'[{i}]: {c.huuid}: {c.__class__.__name__} ~ [{steps[c.parent[1]]}]'
                )
            except KeyError:
                logging.info(f'[{i}]: {c.huuid}: {c.__class__.__name__}')
        else:
            logging.info(f'[{i}]: {c.huuid}: {c.__class__.__name__}')

    logging.info('-' * 100)
    logging.info('JOBS EVENTS:')
    logging.info('-' * 100)
    steps = {j.job_id: str(i) for i, j in enumerate(job_events.values())}

    if not job_events:
        logging.info('No job events...')
    else:
        for i, j in enumerate(job_events.values()):
            if j.dependencies:
                logging.info(
                    f'[{i}]: {j.huuid}: {j.method} ~ '
                    f'[{",".join([steps[d] for d in j.dependencies])}]'
                )
            else:
                logging.info(f'[{i}]: {j.huuid}: {j.method}')

    logging.info('-' * 100)

    events = [
        *list(create_events.values()),
        *list(job_events.values()),
        Signal(context=object.uuid, msg='done'),
    ]

    if not force:
        if not click.confirm(
            '\033[1mPlease approve this deployment plan.\033[0m',
            default=True,
        ):
            return object
    assert db.cluster is not None
    db.cluster.scheduler.publish(events=events)
    if wait:
        unique_create_events = list(create_events.values())
        _wait_on_events(db, unique_create_events)
    return object


def _wait_on_events(db, events):
    time_left = _WAIT_TIMEOUT
    while True:
        remaining = len(events)
        for event in events:
            identifier = event.data['identifier']
            component = event.component
            version = event.data['version']
            try:
                db.load(component=component, identifier=identifier, version=version)
            except exceptions.NotFound:
                pass
            else:
                remaining -= 1

        if remaining <= 0:
            return
        elif time_left == 0:
            raise TimeoutError("Timeout error while waiting for create events.")
        else:
            time.sleep(1)
            time_left -= 1


def _apply(
    db: 'Datalayer',
    object: 'Component',
    non_breaking_changes: t.Dict,
    context: str | None = None,
    job_events: t.Dict[str, 'Job'] | None = None,
    parent: t.Optional[t.List] = None,
    global_diff: t.Dict | None = None,
):

    if context is None:
        context = object.uuid

    if job_events and any(x.startswith(object.huuid) for x in job_events):
        return [], []

    if job_events is None:
        job_events = {}

    if object.huuid in job_events:
        return [], []

    object.db = db

    create_events = {}
    children = []

    def wrapper(child):
        nonlocal create_events

        c, j = _apply(
            db=db,
            object=child,
            context=context,
            job_events=job_events,
            parent=[object.component, object.identifier, object.uuid],
            global_diff=global_diff,
            non_breaking_changes=non_breaking_changes,
        )

        job_events.update(j)
        create_events.update(c)
        children.append((child.component, child.identifier, child.uuid))
        return f'&:component:{child.huuid}'

    try:
        current = db.load(object.__class__.__name__, object.identifier)
        if current.hash == object.hash:
            apply_status = 'same'
            object.version = current.version
            object.status = running_status()
        elif current.uuid == object.uuid:
            apply_status = 'update'
            object.version = current.version
        else:
            apply_status = 'breaking'
            assert current.version is not None
            object.version = current.version + 1
    except exceptions.NotFound:
        apply_status = 'new'
        object.version = 0

    if apply_status in {'update', 'breaking'}:

        diff = object.diff(current)

        if global_diff is not None:
            global_diff[object.identifier] = {
                'status': apply_status,
                'changes': diff,
                'component': object.component,
            }

    serialized = object.dict()

    # This map function applies `wrapper` to anything
    # "found" inside the `Document`, which is a `Component`
    # The output document has the output of `wrapper`
    # as replacement for those leaves which are `Component`
    # instances.
    serialized = serialized.map(wrapper, lambda x: isinstance(x, Component))
    Document(object.metadata).map(wrapper, lambda x: isinstance(x, Component))
    serialized = db._save_artifact(serialized.encode())

    if apply_status == 'same':
        return create_events, job_events

    elif apply_status == 'new':

        metadata_event = Create(
            context=context,
            path=object.__module__ + '.' + object.__class__.__name__,
            data=serialized,
            parent=parent,
            children=children,
        )

        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=list(job_events.values()),
            context=context,
        )
    elif apply_status == 'breaking':

        metadata_event = Create(
            context=context,
            path=object.__module__ + '.' + object.__class__.__name__,
            data=serialized,
            parent=parent,
            children=children,
        )

        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=list(job_events.values()),
            context=context,
        )
    else:
        assert apply_status == 'update'

        metadata_event = Update(
            context=context,
            component=object.__class__.__name__,
            data=serialized,
            parent=parent,
        )

        # the requires flag, allows
        # the developer to restrict jobs "on-update"
        # to only be those jobs concerned with the
        # change data
        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=list(job_events.values()),
            context=context,
            requires=diff,
        )

    # If nothing needs to be done, then don't
    # require the status to be "initializing"

    if not these_job_events:
        if not CFG.json_native:
            metadata_event.data['status'] = json.dumps(running_status())
        else:
            metadata_event.data['status'] = running_status()

        object.status = running_status()

    create_events[metadata_event.huuid] = metadata_event
    job_events.update({jj.huuid: jj for jj in these_job_events})
    return create_events, job_events
