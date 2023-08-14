from contextlib import contextmanager
import typing as t

from sqlalchemy import JSON, Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from superduperdb.db.base.metadata import MetaDataStore


Base = declarative_base()


class MetaData(Base):  # TODO make configuration in s.CFG, not metadata
    __tablename__ = 'meta'
    
    id = Column(Integer, primary_key=True)
    key: Column(String)
    value: Column(String)


class Component(Base):
    __tablename__ = 'objects'
    
    id = Column(Integer, primary_key=True)
    type_id: Column(String)
    identifier: Column(String)
    info: Column(JSON)
    parent: Column(Integer, ForeignKey('objects.id'), nullable=True)


class Job(Base):
    identifier: Column(String, primary_key=True)
    info: Column(JSON)
    status: Column(String) 
    args: Column(JSON)
    kwargs: Column(JSON)
    stdout: Column(String)
    stderr: Column(String)


@contextmanager
def session_scope(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.commit()


class SQLAlchemyMetadata(MetaDataStore):
    """
    This is a metadata store that uses SQLAlchemy to store metadata in a SQL database.
    """
    def __init__(self, conn):
        self.engine = conn
        Base.metadata.create_all(self.engine)

    def create_component(self, info: t.Dict):
        component = Component(
            info=info,
            identifier=info['identifier'],
            type_id=info['type_id'],
            parent=info.get('parent'),
        )
        with session_scope(self.engine) as session:
            session.add(component)

    def create_job(self, info: t.Dict):
        with session_scope(self.engine) as session:
            session.add(Job(**info))