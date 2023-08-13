from sqlalchemy import JSON, Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base



Base = declarative_base()


class MetaData(Base):  # TODO make configuration in s.CFG
    __tablename__ = 'meta'
    
    id = Column(Integer, primary_key=True)
    n_download_workers: Column(Integer)
    headers: Column(JSON, nullable=True)
    download_timeout: Column(Integer)


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


class IbisMetadata:
    def __init__(self, conn):
        self.alchemy = conn._find_backend().con
        Base.metadata.create_all(self.alchemy)

    