import threading
import typing as t
from contextlib import contextmanager

import click
from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from superduperdb import logging
from superduperdb.backends.base.metadata import MetaDataStore, NonExistentMetadataError
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component as _Component
from superduperdb.misc.colors import Colors

if t.TYPE_CHECKING:
    from superduperdb.backends.base.query import Select

Base = declarative_base()
DEFAULT_LENGTH = 255


class DictMixin:
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class QueryID(Base):  # type: ignore[valid-type, misc]
    __tablename__ = 'query_id_table'

    query_id = Column(String(DEFAULT_LENGTH), primary_key=True)
    query = Column(JSON)
    model = Column(String(DEFAULT_LENGTH))


class Job(Base, DictMixin):  # type: ignore[valid-type, misc]
    __tablename__ = 'job'

    identifier = Column(String(DEFAULT_LENGTH), primary_key=True)
    component_identifier = Column(String(DEFAULT_LENGTH))
    type_id = Column(String(DEFAULT_LENGTH))
    info = Column(JSON)
    time = Column(DateTime)
    status = Column(String(DEFAULT_LENGTH))
    args = Column(JSON)
    kwargs = Column(JSON)
    method_name = Column(String(DEFAULT_LENGTH))
    stdout = Column(JSON)
    stderr = Column(JSON)
    cls = Column(String(DEFAULT_LENGTH))


class ParentChildAssociation(Base):  # type: ignore[valid-type, misc]
    __tablename__ = 'parent_child_association'

    parent_id = Column(String(DEFAULT_LENGTH), primary_key=True)
    child_id = Column(String(DEFAULT_LENGTH), primary_key=True)


class Component(Base, DictMixin):  # type: ignore[valid-type, misc]
    __tablename__ = 'component'

    id = Column(String(DEFAULT_LENGTH), primary_key=True)
    identifier = Column(String(DEFAULT_LENGTH))
    version = Column(Integer)
    hidden = Column(Boolean)
    type_id = Column(String(DEFAULT_LENGTH))
    cls = Column(String(DEFAULT_LENGTH))
    module = Column(String(DEFAULT_LENGTH))
    dict = Column(JSON)

    # Define the parent-child relationship
    children = relationship(
        "Component",
        secondary=ParentChildAssociation.__table__,
        primaryjoin=id == ParentChildAssociation.parent_id,
        secondaryjoin=id == ParentChildAssociation.child_id,
        backref="parents",
        cascade="all, delete",
    )


class Meta(Base, DictMixin):  # type: ignore[valid-type, misc]
    __tablename__ = 'meta'

    key = Column(String(DEFAULT_LENGTH), primary_key=True)
    value = Column(String(DEFAULT_LENGTH))


class SQLAlchemyMetadata(MetaDataStore):
    """
    Abstraction for storing meta-data separately from primary data.

    :param conn: connection to the meta-data store
    :param name: Name to identify DB using the connection
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn
        Base.metadata.create_all(self.conn)

        self._lock = threading.Lock()

    def url(self):
        return self.conn.url + self.name

    def drop(self, force: bool = False):
        """
        Drop the metadata store.
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all meta-data? ',
                default=False,
            ):
                logging.warn('Aborting...')
        Base.metadata.drop_all(self.conn)

    @contextmanager
    def session_context(self):
        sm = sessionmaker(bind=self.conn)
        session = sm()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # --------------- COMPONENTS -----------------

    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ):
        unique_id = _Component.make_unique_id(type_id, identifier, version)
        with self.session_context() as session:
            return (
                session.query(ParentChildAssociation)
                .filter(
                    ParentChildAssociation.child_id == unique_id,
                )
                .first()
                is not None
            )

    def create_component(self, info: t.Dict):
        if 'hidden' not in info:
            info['hidden'] = False
        info['id'] = f'{info["type_id"]}/{info["identifier"]}/{info["version"]}'
        with self.session_context() as session:
            session.add(Component(**info))

    def create_parent_child(self, parent_id: str, child_id: str):
        with self.session_context() as session:
            association = ParentChildAssociation(parent_id=parent_id, child_id=child_id)
            session.add(association)

    def delete_component_version(self, type_id: str, identifier: str, version: int):
        with self.session_context() as session:
            cv = (
                session.query(Component)
                .filter(
                    Component.type_id == type_id,
                    Component.identifier == identifier,
                    Component.version == version,
                )
                .first()
            )
            if cv:
                session.delete(cv)

    def _get_component(
        self,
        type_id: str,
        identifier: str,
        version: int,
        allow_hidden: bool = False,
    ):
        with self.session_context() as session:
            if not allow_hidden:
                res = (
                    session.query(Component)
                    .filter(
                        Component.type_id == type_id,
                        Component.identifier == identifier,
                        Component.version == version,
                        Component.hidden.__eq__(False),
                    )
                    .first()
                )
            else:
                res = (
                    session.query(Component)
                    .filter(
                        Component.type_id == type_id,
                        Component.identifier == identifier,
                        Component.version == version,
                    )
                    .first()
                )

            return res.as_dict() if res else None

    def get_component_version_parents(self, unique_id: str):
        with self.session_context() as session:
            assocations = (
                session.query(ParentChildAssociation)
                .filter(
                    ParentChildAssociation.child_id == unique_id,
                )
                .all()
            )
            parents = [a.parent_id for a in assocations]
            return parents

    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
        with self.session_context() as session:
            component = (
                session.query(Component)
                .filter(
                    Component.type_id == type_id,
                    Component.identifier == identifier,
                    Component.hidden == allow_hidden,
                )
                .order_by(Component.version.desc())
                .first()
            )
            if component is None:
                raise FileNotFoundError(
                    f'Can\'t find {type_id}: {identifier} in metadata'
                )
            return component.version

    def hide_component_version(self, type_id: str, identifier: str, version: int):
        with self.session_context() as session:
            session.query(Component).filter(
                Component.type_id == type_id,
                Component.identifier == identifier,
                Component.version == version,
            ).update({'hidden': True})

    def _replace_object(self, info, identifier, type_id, version):
        with self.session_context() as session:
            session.query(Component).filter(
                Component.type_id == type_id,
                Component.identifier == identifier,
                Component.version == version,
            ).update(info)

    def replace_component(
        self,
        info: t.Dict[str, t.Any],
        identifier: str,
        type_id: str,
        version: t.Optional[int] = None,
    ) -> None:
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        return self._replace_object(
            info=info,
            identifier=identifier,
            type_id=type_id,
            version=version,
        )

    def show_components(self, type_id: t.Optional[str] = None, **kwargs):
        if type_id is not None:
            with self.session_context() as session:
                identifiers = [
                    c.identifier
                    for c in session.query(Component)
                    .filter(Component.type_id == type_id)
                    .all()
                ]
        else:
            with self.session_context() as session:
                identifiers = [c.identifier for c in session.query(Component).all()]
        identifiers = sorted(set(identifiers), key=lambda x: identifiers.index(x))
        return identifiers

    def show_component_versions(self, type_id: str, identifier: str):
        with self.session_context() as session:
            return [
                c.version
                for c in session.query(Component)
                .filter(
                    Component.type_id == type_id, Component.identifier == identifier
                )
                .all()
            ]

    def _update_object(
        self,
        identifier: str,
        type_id: str,
        key: str,
        value: t.Any,
        version: int,
    ):
        with self.session_context() as session:
            session.query(Component).filter(
                Component.type_id == type_id,
                Component.identifier == identifier,
                Component.version == version,
            ).update({key: value})

    # --------------- JOBS -----------------

    def create_job(self, info: t.Dict):
        with self.session_context() as session:
            session.add(Job(**info))

    def get_job(self, job_id: str):
        with self.session_context() as session:
            return session.query(Job).filter(Job.identifier == job_id).first()

    def listen_job(self, identifier: str):
        # Not supported currently
        raise NotImplementedError

    def show_jobs(self):
        with self.session_context() as session:
            return [j.identifier for j in session.query(Job).all()]

    def update_job(self, job_id: str, key: str, value: t.Any):
        with self.session_context() as session:
            session.query(Job).filter(Job.identifier == job_id).update({key: value})

    def write_output_to_job(self, identifier, msg, stream):
        # Not supported currently
        raise NotImplementedError

    # --------------- METADATA -----------------

    def create_metadata(self, key, value):
        with self.session_context() as session:
            session.add(Meta(key=key, value=value))

    def get_metadata(self, key):
        with self.session_context() as session:
            return session.query(Meta).filter(Meta.key == key).first().value

    def update_metadata(self, key, value):
        with self.session_context() as session:
            session.query(Meta).filter(Meta.key == key).update({Meta.value: value})

    # --------------- Query ID -----------------
    def add_query(self, query: 'Select', model: str):
        query_hash = str(hash(query))

        with self.session_context() as session:
            with self._lock:
                row = {
                    'query': query.serialize(),
                    'model': model,
                    'query_id': query_hash,
                }

            session.add(QueryID(**row))

    def get_query(self, query_hash: str):
        """
        Get the query from the query table corresponding to the query hash
        """
        try:
            with self.session_context() as session:
                out = (
                    session.query(QueryID)
                    .filter(QueryID.query_id == str(query_hash))
                    .first()
                )
        except AttributeError as e:
            if 'NoneType' in str(e):
                raise NonExistentMetadataError(
                    f'Query hash {query_hash} does not exist'
                )
            raise e

        if out is None:
            raise NonExistentMetadataError(f'Query hash {query_hash} does not exist')

    def get_model_queries(self, model: str):
        """
        Get queries related to the given model.
        """
        with self.session_context() as session:
            queries = session.query(QueryID).filter(QueryID.model == model).all()

            unpacked_queries = []
            for row in queries:
                id = row.query_id
                serialized = row.query
                query = Serializable.deserialize(serialized)
                unpacked_queries.append(
                    {'query_id': id, 'query': query, 'sql': query.repr_()}
                )
            return unpacked_queries

    def disconnect(self):
        """
        Disconnect the client
        """

        # TODO: implement me
