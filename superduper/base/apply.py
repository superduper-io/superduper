import time
import typing as t

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from superduper import CFG, Component, logging
from superduper.base import exceptions
from superduper.base.document import Document
from superduper.base.event import Create, PutComponent, Signal, Update
from superduper.base.status import pending_status, running_status
from superduper.misc.tree import dict_to_tree

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.metadata import Job

_WAIT_TIMEOUT = 60

console = Console()


class HashingError(Exception):
    """Raised when hashing is not deterministic.

    #noqa
    """

    pass


def apply(
    db: 'Datalayer',
    object: t.Union['Component', t.Sequence[t.Any], t.Any],
    force: bool | None = None,
    wait: bool = False,
    jobs: bool = True,
) -> 'Component':
    """Apply a `superduper.Component`.

    :param db: The Datalayer instance to use.
    :param object: The component to apply.
    :param force: Whether to force the application without confirmation.
    :param wait: Whether to wait for the component to be created.
    :param jobs: Whether to execute jobs after applying the component.
    """
    if not isinstance(object, Component):
        raise ValueError('Only components can be applied')

    object.setup()

    # Detect non‑deterministic UUIDs early -----------------------------------
    if object.uuid != object.uuid:
        raise HashingError(
            'The component you specified did not yield a deterministic hash. '
            'This is a requirement for the object to be stored in the system. '
            'Modify your classes and check `object.uuid`.'
        )

    if force is None:
        force = db.cfg.force_apply

    diff: dict[str, t.Any] = {}

    # -----------------------------------------------------------------------
    # 1. Show the component that is about to be applied
    # -----------------------------------------------------------------------
    if CFG.log_level == 'USER':
        console.print(Panel('Component to apply', style='bold green'))
        object.show()

    # -----------------------------------------------------------------------
    # 2. Analyse what needs to change
    # -----------------------------------------------------------------------
    table_events, create_events, put_events, job_events = _apply(
        db=db,
        object=object,
        context=object.uuid,
        job_events={},
        global_diff=diff,
        non_breaking_changes={},
    )

    # Skip job execution if required
    if not jobs:
        logging.info('Skipping job execution because of the --no-jobs flag')
        job_events = {}

    # Nothing to do? Exit early.
    if not create_events:
        console.print(Panel('No changes needed – doing nothing!', style='bold yellow'))
        return object

    # -----------------------------------------------------------------------
    # 3. Present the diff (if any) and the deployment plan
    # -----------------------------------------------------------------------
    if diff:
        console.print(Panel('Diff vs current state', style='bold blue'))
        console.print(dict_to_tree(diff, root=object.identifier), soft_wrap=True)

    console.print(Panel('Deployment plan', style='bold blue'))

    # -----------------------------------------------------------------------
    # 3a. Build a *single* consolidated table covering ALL events
    # -----------------------------------------------------------------------
    consolidated: list[tuple[str, str, str]] = []  # (idx, type, details)
    idx = 0

    for ev in table_events.values():
        consolidated.append((str(idx), 'TABLE', ev.huuid))
        idx += 1

    for ev in create_events.values():
        consolidated.append(
            (str(idx), 'CREATE', f'{ev.huuid}: {ev.__class__.__name__}')
        )
        idx += 1

    for ev in put_events.values():
        consolidated.append((str(idx), 'PUT', ev.huuid))
        idx += 1

    job_lookup = {}
    for ev in job_events.values():
        int_dependencies = [str(job_lookup[job_id]) for job_id in ev.dependencies]
        dep_str = f" deps→{','.join(int_dependencies)}" if int_dependencies else ''
        consolidated.append((str(idx), 'JOB', f'{ev.huuid}: {ev.method}{dep_str}'))
        job_lookup[ev.job_id] = len(consolidated) - 1
        idx += 1

    tbl = Table(show_header=True, header_style='bold magenta')
    tbl.add_column('#', style='cyan', no_wrap=True)
    tbl.add_column('Event type', style='magenta')
    tbl.add_column('Details', style='white')

    for row in consolidated:
        tbl.add_row(*row)

    console.print(
        Panel(tbl, title='Deployment plan – events overview', border_style='green')
    )

    # -----------------------------------------------------------------------
    # 4. Confirm (unless --force) and publish the events
    # -----------------------------------------------------------------------
    if not force:
        if not click.confirm(
            '\033[1mPlease approve this deployment plan.\033[0m', default=True
        ):
            return object

    assert db.cluster is not None  # mypy‑friendliness

    events = [
        *table_events.values(),
        *create_events.values(),
        *put_events.values(),
        *job_events.values(),
        Signal(context=object.uuid, msg='done'),
    ]
    db.cluster.scheduler.publish(events=events)

    if wait:
        _wait_on_events(db, list(create_events.values()))

    return object


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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
            raise TimeoutError('Timeout while waiting for create events.')
        else:
            time.sleep(1)
            time_left -= 1


# ---------------------------------------------------------------------------
# Recursive helper to build up metadata / job / put / table events
# ---------------------------------------------------------------------------


def _apply(
    db: 'Datalayer',
    object: 'Component',
    non_breaking_changes: t.Dict,
    context: str | None = None,
    job_events: t.Dict[str, 'Job'] | None = None,
    parent: t.Optional[t.List] = None,
    global_diff: t.Dict | None = None,
    processed_components: t.Optional[t.Set] = None,
):

    object.status, object.details = pending_status()

    processed_components = processed_components or set()
    if context is None:
        context = object.uuid

    if job_events and object.huuid in processed_components:
        return {}, {}, {}, {}

    if job_events is None:
        job_events = {}

    if object.huuid in job_events:
        return {}, {}, {}, {}

    object.db = db

    create_events: dict[str, Create | Update] = {}
    table_events: dict[str, t.Any] = {}
    put_events: dict[str, PutComponent] = {}
    children = []

    # ------------------------------------------------------------------
    # Wrapper to recursively walk component trees
    # ------------------------------------------------------------------
    def wrapper(child):
        nonlocal create_events, processed_components

        t_, c_, p_, j_ = _apply(
            db=db,
            object=child,
            context=context,
            job_events=job_events,
            parent=[object.component, object.identifier, object.uuid],
            global_diff=global_diff,
            non_breaking_changes=non_breaking_changes,
            processed_components=processed_components,
        )

        job_events.update(j_)
        processed_components |= {j__.rsplit('.')[0] for j__ in j_}
        create_events.update(c_)
        table_events.update(t_)
        put_events.update(p_)
        children.append((child.component, child.identifier, child.uuid))
        return f'&:component:{child.huuid}'

    # ------------------------------------------------------------------
    # Decide whether this is a new / update / breaking / same component
    # ------------------------------------------------------------------
    try:
        current = db.load(object.__class__.__name__, object.identifier)
        current.setup()
        if current.hash == object.hash:
            apply_status = 'same'
            object.version = current.version
            object.status, object.details = running_status()
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

    # ------------------------------------------------------------------
    # Serialize the component (recursively replacing child components)
    # ------------------------------------------------------------------
    serialized: Document = object.dict()
    serialized = serialized.map(wrapper, lambda x: isinstance(x, Component))
    serialized = db._save_artifact(serialized.encode())

    # ------------------------------------------------------------------
    # Build create / update events depending on apply_status
    # ------------------------------------------------------------------
    if apply_status == 'same':
        return table_events, create_events, put_events, job_events

    if apply_status == 'new':
        metadata_event = Create(
            context=context,
            path=f'{object.__module__}.{object.__class__.__name__}',
            data=serialized,
            parent=parent,
            children=children,
        )
        table_events.update(object.create_table_events())
        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=list(job_events.values()),
            context=context,
        )
        for service in object.services:
            put_events[f'{object.huuid}/{service}'] = PutComponent(
                component=object.component,
                identifier=object.identifier,
                uuid=object.uuid,
                context=context,
                service=service,
            )

    elif apply_status == 'breaking':
        metadata_event = Create(
            context=context,
            path=f'{object.__module__}.{object.__class__.__name__}',
            data=serialized,
            parent=parent,
            children=children,
        )
        table_events.update(object.create_table_events())
        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=list(job_events.values()),
            context=context,
        )
        for service in object.services:
            put_events[f'{object.huuid}/{service}'] = PutComponent(
                component=object.component,
                identifier=object.identifier,
                uuid=object.uuid,
                context=context,
                service=service,
            )

    else:  # apply_status == 'update'
        diff = object.diff(current)
        metadata_event = Update(
            context=context,
            component=object.__class__.__name__,
            data=serialized,
            parent=parent,
        )
        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=list(job_events.values()),
            context=context,
            requires=diff,
        )

    create_events[metadata_event.huuid] = metadata_event
    job_events.update({j.huuid: j for j in these_job_events})

    return table_events, create_events, put_events, job_events
