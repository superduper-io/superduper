import typing as t

import click
from rich.console import Console

from superduper import Component, logging
from superduper.base.document import Document
from superduper.base.event import Create, Signal, Update
from superduper.components.component import Status
from superduper.misc.tree import dict_to_tree

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

    # This holds a record of the changes
    diff: t.Dict = {}
    # context allows us to track the origin of the component creation
    create_events, job_events = _apply(
        db=db,
        object=object,
        context=object.uuid,
        job_events={},
        global_diff=diff,
    )
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

    steps = {c.component['uuid']: str(i) for i, c in enumerate(create_events.values())}

    for i, c in enumerate(create_events.values()):
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
    db.cluster.queue.publish(events=events)
    return object


def _apply(
    db: 'Datalayer',
    object: 'Component',
    context: str | None = None,
    job_events: t.Dict[str, 'Job'] | None = None,
    parent: t.Optional[str] = None,
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

    serialized = object.dict(metadata=False)
    del serialized['uuid']

    create_events = {}

    def wrapper(child):
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
            global_diff=global_diff,
        )

        job_events.update(j)
        create_events.update(c)

        return f'&:component:{child.huuid}'

    # This map function applies `wrapper` to anything
    # "found" inside the `Document`, which is a `Component`
    # The output document has the output of `wrapper`
    # as replacement for those leaves which are `Component`
    # instances.

    serialized = serialized.map(wrapper, lambda x: isinstance(x, Component))

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
            apply_status = 'breaking'

            if object.uuid == current.uuid:
                # This happens if the developer performs "surgery"
                # on an already instantiated object (uuid is not rebuilt)

                raise NotImplementedError(
                    'Component was modified in place. This is currently not '
                    'supported. '
                    'To re-apply a component, rebuild the Python object.'
                )

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
            logging.info(f'Found breaking changes in {object.huuid}')

            # if the metadata includes components, which
            # need to be applied, do that now
            Document(object.metadata).map(wrapper, lambda x: isinstance(x, Component))

        else:
            # if object.identifier ==
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
        Document(object.metadata).map(wrapper, lambda x: isinstance(x, Component))

        serialized = serialized.encode()

        object.version = 0
        apply_status = 'new'
        logging.info(f'Found new {object.huuid}')

    if global_diff is not None and apply_status in {'update', 'breaking'}:
        this_diff = this_diff.map(
            lambda x: '?' + x.split(':')[3],
            lambda x: isinstance(x, str) and x.startswith('&:component:'),
        )
        global_diff[object.identifier] = {
            'status': apply_status,
            'changes': dict(this_diff),
            'type_id': object.type_id,
        }

    serialized = db._save_artifact(object.uuid, serialized)

    if apply_status in {'new', 'breaking'}:
        metadata_event = Create(
            context=context,
            component=serialized,
            parent=parent,
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
            component=serialized,
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
            requires=list(this_diff.keys()),
        )

    # If nothing needs to be done, then don't
    # require the status to be "initializing"
    if not these_job_events:
        metadata_event.component['status'] = Status.ready
        object.status = Status.ready

    create_events[metadata_event.huuid] = metadata_event
    job_events.update({jj.huuid: jj for jj in these_job_events})
    return create_events, job_events