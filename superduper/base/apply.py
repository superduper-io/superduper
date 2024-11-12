import typing as t

import click

from superduper import Component, logging
from superduper.base.document import Document
from superduper.base.event import Create, Signal, Update
from superduper.components.component import Status

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.event import Job


def apply(
    db: 'Datalayer',
    object: t.Union['Component', t.Sequence[t.Any], t.Any],
    force: bool | None = None,
):
    """
    Add functionality in the form of components.

    Components are stored in the configured artifact store
    and linked to the primary database through metadata.

    :param db: Datalayer instance
    :param object: Object to be stored.
    :param force: List of jobs which should execute before component
                            initialization begins.
    :return: Tuple containing the added object(s) and the original object(s).
    """
    if force is None:
        force = db.cfg.force_apply

    if not isinstance(object, Component):
        raise ValueError('Only components can be applied')

    # This populates the component with data fetched
    # from `db` if necessary
    # We need pre as well as post-create, since the order
    # between parents and children are reversed in each
    # sometimes parents might need to grab things from children
    # and vice-versa
    object.pre_create(db)

    # context allows us to track the origin of the component creation
    create_events, job_events = _apply(
        db=db,
        object=object,
        context=object.uuid,
        job_events=[],
    )
    # this flags that the context is not needed anymore
    if not create_events:
        logging.info('No changes needed, doing nothing!')
        return object

    # TODO for some reason the events get created multiple times
    # we need to fix that to prevent inefficiencies
    unique_create_ids = []
    unique_create_events = []
    for e in create_events:
        if e.component['uuid'] not in unique_create_ids:
            unique_create_ids.append(e.component['uuid'])
            unique_create_events.append(e)

    unique_job_ids = []
    unique_job_events = []
    for e in job_events:
        if e.job_id not in unique_job_ids:
            unique_job_ids.append(e.job_id)
            unique_job_events.append(e)

    logging.info('-' * 100)
    logging.info('METADATA EVENTS:')
    logging.info('-' * 100)
    steps = {c.component['uuid']: str(i) for i, c in enumerate(unique_create_events)}
    for i, c in enumerate(unique_create_events):
        if c.parent:
            try:
                logging.info(f'[{i}]: {c.huuid}: {c.genus} ~ [{steps[c.parent]}]')
            except KeyError:
                logging.info(f'[{i}]: {c.huuid}: {c.genus}')
        else:
            logging.info(f'[{i}]: {c.huuid}: {c.genus}')

    logging.info('-' * 100)
    logging.info('JOBS EVENTS:')
    logging.info('-' * 100)
    steps = {j.job_id: str(i) for i, j in enumerate(unique_job_events)}

    # TODO why do we need this?
    def uniquify(x):
        return sorted(list(set(x)))

    for i, j in enumerate(unique_job_events):
        if j.dependencies:
            logging.info(
                f'[{i}]: {j.huuid}: {j.method} ~ '
                f'[{",".join(uniquify([steps[d] for d in j.dependencies]))}]'
            )
        else:
            logging.info(f'[{i}]: {j.huuid}: {j.method}')

    logging.info('-' * 100)

    events = [
        *unique_create_events,
        *unique_job_events,
        Signal(context=object.uuid, msg='done'),
    ]

    if not force:
        if not click.confirm(
            '\033[1mPlease approve this deployment plan.\033[0m',
            default=True,
        ):
            return object
    db.cluster.queue.publish(events=events)
    return object


def _apply(
    db: 'Datalayer',
    object: 'Component',
    context: str,
    job_events: t.List['Job'],
    parent: t.Optional[str] = None,
):
    object.db = db

    serialized = object.dict(metadata=False)
    del serialized['uuid']

    create_events = []
    job_events = list(job_events)

    def wrapper(child):
        nonlocal job_events
        nonlocal create_events

        # handle the @component decorator
        # this shouldn't be applied, but
        # only saved as a quasi-leaf
        if getattr(child, 'inline', True):
            return child

        c, j = _apply(
            db=db,
            object=child,
            context=context,
            job_events=job_events,
            parent=object.uuid,
        )
        create_events += c
        job_events += j

        return f'&:component:{child.huuid}'

    # This map function applies `wrapper` to anything
    # "found" inside the `Document`, which is a `Component`
    # The output document has the output of `wrapper`
    # as replacement for those leaves which are `Component`
    # instances.
    serialized = serialized.map(wrapper, Component)

    try:
        current = db.load(object.type_id, object.identifier)

        # only check for diff not in metadata/ uuid
        # also only
        current_serialized = current.dict(metadata=False, refs=True)
        del current_serialized['uuid']

        # finds the fields where there is a difference
        this_diff = Document(current_serialized).diff(serialized)
        logging.info(f'Found identical {object.huuid}')

        if not this_diff:
            # if no change then update the component
            # to have the same info as the "existing" version
            current.handle_update_or_same(object)
            return create_events, job_events

        elif set(this_diff.keys(deep=True)).intersection(object.breaks):
            # if this is a breaking change then create a new version
            apply_status = 'broken'

            # this overwrites the fields which were made
            # during the `.map` to the children
            # serializer.map...
            # this means replacing components with references
            serialized = object.dict().update(serialized)

            # this is necessary to prevent inconsistencies
            # this takes the difference between
            # the current and
            serialized = serialized.update(this_diff).encode()

            # assign/ increment the version since
            # this breaks a previous version
            assert current.version is not None
            object.version = current.version + 1
            serialized['version'] = current.version + 1
            logging.info(f'Found broken {object.huuid}')

            # if the metadata includes components, which
            # need to be applied, do that now
            Document(object.metadata).map(wrapper, Component)

        else:
            apply_status = 'update'
            current.handle_update_or_same(object)

            serialized['version'] = current.version
            serialized['uuid'] = current.uuid

            # update the existing component with the change
            # data from the applied component
            serialized = current.dict().update(serialized).update(this_diff).encode()

            logging.info(f'Found update {object.huuid}')

    except FileNotFoundError:
        serialized['version'] = 0
        serialized = object.dict().update(serialized)

        # if the metadata includes components, which
        # need to be applied, do that now
        Document(object.metadata).map(wrapper, Component)

        serialized = serialized.encode()

        object.version = 0
        apply_status = 'new'
        logging.info(f'Found new {object.huuid}')

    serialized = db._save_artifact(object.uuid, serialized)

    if apply_status in {'new', 'broken'}:
        metadata_event = Create(
            context=context,
            component=serialized,
            parent=parent,
        )

        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=job_events,
            context=context,
        )
    else:
        assert apply_status == 'update'

        metadata_event = Update(
            context=context,
            component=serialized,
            parent=parent,
        )

        # the requires flag, allows
        # the developer to restrict jobs "on-update"
        # to only be those jobs concerned with the
        # change data
        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=job_events,
            context=context,
            requires=list(this_diff.keys()),
        )

    # If nothing needs to be done, then don't
    # require the status to be "initializing"
    if not these_job_events:
        metadata_event.component['status'] = Status.ready
        object.status = Status.ready

    create_events.append(metadata_event)
    return create_events, job_events + these_job_events
