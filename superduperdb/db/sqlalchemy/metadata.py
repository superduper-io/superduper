import typing as t
from contextlib import contextmanager

import click
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from superduperdb.db.base.metadata import MetaDataStore
from superduperdb.misc.colors import Colors

Base = declarative_base()


class DictMixin:
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Job(Base, DictMixin):  # type: ignore[valid-type, misc]
    __tablename__ = 'job'

    identifier = Column(String, primary_key=True)
    info = Column(JSON)
    time = Column(DateTime)
    status = Column(String)
    args = Column(JSON)
    kwargs = Column(JSON)


parent_child_association = Table(
    'parent_child_association',
    Base.metadata,
    Column('parent_id', String, ForeignKey('component.id')),
    Column('child_id', String, ForeignKey('component.id')),
)


class Component(Base, DictMixin):  # type: ignore[valid-type, misc]
    __tablename__ = 'component'

    id = Column(String, primary_key=True)
    identifier = Column(String)
    version = Column(Integer)
    hidden = Column(Boolean)
    type_id = Column(String)
    cls = Column(String)
    module = Column(String)
    dict = Column(JSON)

    # Define the parent-child relationship
    parents = relationship(
        "Component",
        secondary=parent_child_association,
        primaryjoin=id == parent_child_association.c.child_id,
        secondaryjoin=id == parent_child_association.c.parent_id,
        backref="children",
        cascade="all, delete",
    )


class Meta(Base, DictMixin):  # type: ignore[valid-type, misc]
    __tablename__ = 'meta'

    key = Column(String, primary_key=True)
    value = Column(String)


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
                print('Aborting...')
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
        with self.session_context() as session:
            return (
                session.query(Component)
                .filter(
                    Component.type_id == type_id,
                    Component.identifier == identifier,
                    Component.version == version,
                )
                .first()
                .parent_id
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
            session.add(
                parent_child_association.insert().values(
                    parent_id=parent_id, child_id=child_id
                )
            )

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

            return res.as_dict()

    def get_component_version_parents(self, unique_id: str):
        with self.session_context() as session:
            components = (
                session.query(Component)
                .filter(
                    Component.id == unique_id,
                )
                .all()
            )
            return sum([c.parents for c in components], [])

    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
        with self.session_context() as session:
            return (
                session.query(Component)
                .filter(
                    Component.type_id == type_id,
                    Component.identifier == identifier,
                    Component.hidden == allow_hidden,
                )
                .order_by(Component.version.desc())
                .first()
                .version
            )

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
            ).update({'dict': info})

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

    def show_components(self, type_id: str, **kwargs):
        with self.session_context() as session:
            return [
                c.identifier
                for c in session.query(Component)
                .filter(Component.type_id == type_id)
                .all()
            ]

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
