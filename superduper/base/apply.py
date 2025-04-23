import dataclasses as dc
import time
import typing as t
from collections import defaultdict

import click
from rich.console import Console

from superduper import Component, logging
from superduper.base.document import Document
from superduper.base.event import Create, Signal, Update
from superduper.base.exceptions import InvalidArguments
from superduper.base.metadata import NonExistentMetadataError
from superduper.components.component import Status
from superduper.misc.tree import dict_to_tree

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.event import Job

# Constants should be in UPPER_CASE and at the top level
#: Maximum time in seconds to wait for events to be processed
WAIT_TIMEOUT = 60
#: Default delay in seconds between retry attempts
RETRY_DELAY = 3


# Status constants for component application
@dc.dataclass
class ApplyStatus:
    """Constants representing the status of component application.

    # noqa
    """

    #: Component is unchanged, no action needed
    SAME: str = 'same'
    #: Component is new, needs to be created
    NEW: str = 'new'
    #: Component has non-breaking changes, needs update
    UPDATE: str = 'update'
    #: Component has breaking changes, needs recreation
    BREAKING: str = 'breaking'


class HashingError(Exception):
    """Raised when hashing is not deterministic.

    This exception indicates that a component's UUID is not consistently
    reproducible, which is a requirement for the object storage system.

    # noqa
    """

    pass


def _wait_on_events(db: 'Datalayer', events: list, retry_delay=RETRY_DELAY):
    """Wait for events to be processed by the database.

    :param db: Datalayer instance to use for loading events
    :param events: List of events to wait for completion
    :param retry_delay: Time to wait between retries in seconds

    :raises TimeoutError: If events don't complete within timeout period
    :return: None
    """
    time_left = WAIT_TIMEOUT
    pending_events = events

    while pending_events:
        still_pending = []

        for event in pending_events:
            identifier = event.data['identifier']
            component = event.component
            version = event.data['version']
            event_key = f"{component}/{identifier}:{version}"

            try:
                logging.info(f"Waiting for event '{event_key}'")
                db.load(component=component, identifier=identifier, version=version)
                # Successfully loaded, don't add to still_pending
            except NonExistentMetadataError:
                # Not loaded yet, keep in the pending list
                still_pending.append(event)
                logging.warn(f"Event not yet available: {event_key}")
            except Exception as e:
                # Handle other exceptions
                logging.error(f"Error loading event {event_key}: {str(e)}")
                # Still keep in pending list for retry
                still_pending.append(event)

        # Update our pending list
        pending_events = still_pending

        # If we've processed all events, return
        if not pending_events:
            return

        # Check for timeout - do this before sleep to avoid unnecessary waiting
        if time_left <= 0:
            remaining = len(still_pending)
            raise TimeoutError(
                f"Timeout after {WAIT_TIMEOUT}s while waiting for {remaining} events."
            )

        # Wait before next iteration
        time.sleep(retry_delay)
        time_left -= retry_delay


def apply(
    db: 'Datalayer',
    object: t.Union['Component', t.Sequence[t.Any], t.Any],
    force: bool | None = None,
    wait: bool = False,
    jobs: bool = True,
) -> 'Component':
    """Add functionality in the form of components.

    Components are stored in the configured artifact store
    and linked to the primary database through metadata.

    :param db: Datalayer instance
    :param object: Object to be stored
    :param force: Whether to skip confirmation prompt. If None, uses db.cfg.force_apply
    :param wait: Blocks execution till create events finish
    :param jobs: Whether to execute jobs or not

    :return: The processed Component object

    :raises ValueError: If object is not a Component
    :raises HashingError: If component UUID is not deterministic
    """
    if not isinstance(object, Component):
        raise ValueError('Only components can be applied')

    # Fix UUID comparison - likely was meant to check against a cached value
    if object.uuid is None or not object.uuid:
        raise HashingError(
            'The component you specified has an empty UUID. '
            'UUID is a requirement for the object to be stored in the '
            'system.'
        )

    # Check that UUID is deterministic by generating it twice
    first_uuid = object.uuid
    # Recompute UUID - implementation dependent, but assume it recalculates
    second_uuid = object.uuid
    if first_uuid != second_uuid:
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

    logging.info('Found these changes and/or additions that need to be made:')

    # Helper function to display events
    def display_events(event_list, title, id_field='uuid'):
        logging.info('-' * 80)
        logging.info(f'{title}:')
        logging.info('-' * 80)

        # Use a defaultdict to avoid KeyError
        steps = defaultdict(str)
        for i, event in enumerate(event_list):
            steps[
                (
                    getattr(event, id_field)
                    if hasattr(event, id_field)
                    else event.data.get(id_field)
                )
            ] = str(i)

        if not event_list:
            logging.info(f'No {title.lower()} events...')
            return

        for i, event in enumerate(event_list):
            # For Create/Update events (that have parent)
            if hasattr(event, 'parent') and event.parent:
                try:
                    parent = steps[event.parent[1]]
                    logging.info(f'[{i}]: {event.huuid}: {event.genus} ~ [{parent}]')
                except (KeyError, IndexError):
                    logging.info(f'[{i}]: {event.huuid}: {event.genus}')
            # For Job events (that have dependencies but not parent)
            elif hasattr(event, 'dependencies') and event.dependencies:
                deps = []
                for dep in event.dependencies:
                    if dep in steps:
                        deps.append(steps[dep])
                if deps:
                    logging.info(
                        f'[{i}]: {event.huuid}: {event.method} ~ [{",".join(deps)}]'
                    )
                else:
                    logging.info(f'[{i}]: {event.huuid}: {event.method}')
            else:
                # For events without parent or dependencies
                genus = getattr(event, 'genus', None)
                method = getattr(event, 'method', None)
                info = genus if genus else (method if method else "unknown")
                logging.info(f'[{i}]: {event.huuid}: {info}')

    # Display events in a more organized way
    display_events(list(create_events.values()), 'METADATA EVENTS')
    display_events(list(job_events.values()), 'JOBS EVENTS', id_field='job_id')

    # Prepare all events for publishing
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

    db.cluster.scheduler.publish(events=events)

    if wait and create_events:
        unique_create_events = list(create_events.values())
        _wait_on_events(db, unique_create_events)

    return object


def _apply(
    db: 'Datalayer',
    object: 'Component',
    non_breaking_changes: t.Dict,
    context: str | None = None,
    job_events: t.Dict[str, 'Job'] | None = None,
    parent: t.Optional[t.List] = None,
    global_diff: t.Dict | None = None,
):
    """Recursively process a component and its children.

    :param db: Datalayer instance
    :param object: Component to process
    :param non_breaking_changes: Dictionary of non-breaking changes
    :param context: Context identifier (defaults to object.uuid)
    :param job_events: Dictionary of job events
    :param parent: Parent component information
    :param global_diff: Dictionary to track changes globally

    :return: Tuple containing create events and job events dictionaries
    """
    # Use object UUID if context not provided
    if context is None:
        context = object.uuid

    # Initialize job_events dictionary if not provided
    if job_events is None:
        job_events = {}

    # Skip if we've already processed this object
    if job_events and (
        object.huuid in job_events
        or any(x.startswith(object.huuid) for x in job_events)
    ):
        return {}, job_events

    # Set database reference
    object.db = db

    create_events = {}
    children = []

    def wrapper(child):
        """Process a child component and update events dictionaries."""
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

        # Update tracking dictionaries
        job_events.update(j)
        create_events.update(c)
        children.append((child.component, child.identifier, child.uuid))
        return f'&:component:{child.huuid}'

    # Determine component status by checking if it exists and comparing hashes
    try:
        current = db.load(object.__class__.__name__, object.identifier)

        if current.hash == object.hash:
            # No changes
            apply_status = ApplyStatus.SAME
            object.version = current.version
            object.status = Status.ready
        elif current.uuid == object.uuid:
            # Non-breaking update
            apply_status = ApplyStatus.UPDATE
            object.version = current.version
        else:
            # Breaking change
            apply_status = ApplyStatus.BREAKING
            assert current.version is not None
            object.version = current.version + 1
    except NonExistentMetadataError:
        # New component
        apply_status = ApplyStatus.NEW
        object.version = 0

    # Calculate and record differences for update or breaking changes
    if apply_status in {ApplyStatus.UPDATE, ApplyStatus.BREAKING}:
        diff = object.diff(current)

        if global_diff is not None:
            global_diff[object.identifier] = {
                'status': apply_status,
                'changes': diff,
                'component': object.component,
            }

    # Serialize the component and process its children
    #
    # This map function applies `wrapper` to anything
    # "found" inside the `Document`, which is a `Component`
    # The output document has the output of `wrapper`
    # as replacement for those leaves which are `Component`
    # instances.
    serialized = object.dict()
    serialized = serialized.map(wrapper, lambda x: isinstance(x, Component))
    Document(object.metadata).map(wrapper, lambda x: isinstance(x, Component))
    serialized = db._save_artifact(serialized.encode())

    # If no changes, return early
    if apply_status == ApplyStatus.SAME:
        return create_events, job_events

    # Handle event creation based on status
    if apply_status == ApplyStatus.NEW:
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
    elif apply_status == ApplyStatus.BREAKING:
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
    elif apply_status == ApplyStatus.UPDATE:
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
    else:
        raise InvalidArguments(f"unknown operation: {apply_status}")

    # Mark as ready if no jobs need to be run
    if not these_job_events:
        metadata_event.data['status'] = Status.ready
        object.status = Status.ready

    # Record events
    create_events[metadata_event.huuid] = metadata_event
    job_events.update({job.huuid: job for job in these_job_events})

    return create_events, job_events
