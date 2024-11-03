import click
import networkx
from superduper import logging
import typing as t

from superduper.base.constant import KEY_BUILDS
from superduper.base.document import Document
from superduper.base.event import Create, Signal, Update
from superduper.components.component import Status
from superduper.misc.special_dicts import recursive_update
from superduper import Component

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

    logging.info('Here are the CREATION EVENTS:')
    steps = {
        c.component['uuid']: str(i) for i, c in enumerate(unique_create_events)
    }
    for i, c in enumerate(unique_create_events):
        if c.parent:
            try:
                logging.info(f'[{i}]: {c.huuid}: {c.genus} ~ [{steps[c.parent]}]')
            except KeyError:
                logging.info(f'[{i}]: {c.huuid}: {c.genus}')
        else:
            logging.info(f'[{i}]: {c.huuid}: {c.genus}')

    logging.info('JOBS EVENTS:')
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
    existing_versions = db.show(object.type_id, object.identifier)
    already_exists = bool(existing_versions)

    serialized = object.dict()
    broken = True
    if already_exists:
        current = db.load(object.type_id, object.identifier)
        diff = current.dict().diff(serialized)
        diff = Document({k: v for k, v in diff.items() if k not in {'uuid', 'version', 'status'}})
        if not diff:
            return [], job_events

        broken = bool(set(diff.keys()).intersection(object.breaks))
        if not broken:
            serialized = diff

    serialized = serialized.encode(leaves_to_keep=(Component,))

    children = [
        v for v in serialized[KEY_BUILDS].values() if isinstance(v, Component)
    ]

    create_events, j = _apply_child_components(
        db=db,
        components=children,
        parent=object,
        job_events=job_events,
        context=context,
    )
    job_events += j

    if children:
        serialized = _change_component_reference_prefix(serialized)

    serialized = db._save_artifact(object.uuid, serialized)

    if broken:
        if existing_versions:
            object.version = max(existing_versions) + 1
            serialized['version'] = object.version
        else:
            serialized['version'] = 0
            object.version = 0
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
        current_serialized = Document(db.metadata.get_component_by_uuid(uuid=current.uuid))
        serialized = dict(current_serialized.update(serialized))
        object.version = current.version
        object.uuid = current.uuid
        metadata_event = Update(
            context=context,
            component=serialized,
            parent=parent,
        )

        these_job_events = object.create_jobs(
            event_type='apply',
            jobs=job_events,
            context=context,
            requires=list(diff.keys())
        )

    if not these_job_events:
        metadata_event.component['status'] = Status.ready
        object.status = Status.ready

    create_events.append(metadata_event)
    return create_events, job_events + these_job_events
    

def _change_component_reference_prefix(serialized):
    """Replace '?' to '&' in the serialized object."""
    references = {}
    for reference in list(serialized[KEY_BUILDS].keys()):
        if isinstance(serialized[KEY_BUILDS][reference], Component):
            comp = serialized[KEY_BUILDS][reference]
            serialized[KEY_BUILDS].pop(reference)
            references[reference] = (
                comp.type_id + ':' + comp.identifier + ':' + comp.uuid
            )

    # Only replace component references
    if not references:
        return

    def replace_function(value):
        # Change value if it is a string and starts with '?'
        # and the value is in references
        # ?:xxx: -> &:xxx:
        if (
            isinstance(value, str)
            and value.startswith('?')
            and value[1:] in references
        ):
            return '&:component:' + references[value[1:]]
        return value

    serialized = recursive_update(serialized, replace_function)
    return serialized
    

def _apply_child_components(db, components, parent, job_events, context):
    # TODO this is a bit of a mess
    # it handles the situation in `Stack` when
    # the components should be added in a certain order
    G = networkx.DiGraph()
    lookup = {(c.type_id, c.identifier): c for c in components}
    for k in lookup:
        G.add_node(k)
        for d in lookup[k].get_children_refs():  # dependencies:
            if d[:2] in lookup:
                G.add_edge(d, lookup[k].id_tuple)

    nodes = networkx.topological_sort(G)
    create_events = []
    job_events = []
    for n in nodes:
        c, j = _apply(
            db=db, object=lookup[n], parent=parent.uuid, job_events=job_events, context=context
        )
        create_events += c
        job_events += j
    return create_events, job_events
    

def _update_component(object, parent: t.Optional[str] = None):
    # TODO add update logic here to check changed attributes
    logging.debug(
        f'{object.type_id},{object.identifier} already exists - doing nothing'
    )
    return []