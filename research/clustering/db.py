import time
from typing import Dict

from sqlalchemy import Column, INTEGER, TEXT, BOOLEAN, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker


EntityMeta = declarative_base()

CREATED_TIME = int(time.time())
DB_CLUSTERING_NAME = f'clustering_{CREATED_TIME}'
DB_REGEXES_NAME = f'regexes_{CREATED_TIME}'


class Regexes(EntityMeta):
    __tablename__ = DB_REGEXES_NAME
    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    # core fields
    regex = Column(TEXT, nullable=False)
    label = Column(TEXT, nullable=False)
    is_ast = Column(BOOLEAN, nullable=False)
    is_preprocessed = Column(BOOLEAN, nullable=False)
    # extra fields
    created_at = Column(TEXT, nullable=False)

    def normalize(self):
        return {
            'id': self.id,
            'regex': self.regex.__str__(),
            'label': self.label.__str__(),
            'is_ast': self.is_ast,
            'is_preprocessed': self.is_preprocessed,
            'created_at': self.created_at.__str__(),
        }


class Clustering(EntityMeta):
    __tablename__ = DB_CLUSTERING_NAME

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    # core fields
    regex_id = Column(INTEGER, ForeignKey(f'{DB_REGEXES_NAME}.id'), nullable=False)
    cluster_id = Column(INTEGER, nullable=False)
    vectorizer = Column(TEXT, nullable=False)
    cluster_number = Column(INTEGER, nullable=False)
    clustering_algorithm = Column(TEXT, nullable=False)
    input_data_shape = Column(TEXT, nullable=False)
    # extra fields
    created_at = Column(TEXT, nullable=False)

    def normalize(self):
        return {
            'id': self.id,
            'regex_id': self.old_id,
            'cluster_id': self.cluster_id,
            'vectorizer': self.pattvectorizerern.__str__(),
            'cluster_number': self.cluster_number,
            'clustering_algorithm': self.clustering_algorithm.__str__(),
            'input_data_shape': self.input_data_shape.__str__(),
            'created_at': self.created_at.__str__(),
        }


class RegexesRepository:
    def __init__(self, database_url, entity_meta):
        self.engine = create_engine(
            database_url,
            echo=False,
            future=True,
            pool_pre_ping=True,
        )

        self.session = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        entity_meta.metadata.create_all(bind=self.engine)
        self.entity_meta = entity_meta
        self.db = scoped_session(self.session)

    def create_row(self, meta: Regexes) -> Dict:
        if not meta.is_ast:
            meta.is_ast = False
        if not meta.is_preprocessed:
            meta.is_preprocessed = False
        self.db.add(meta)
        self.db.commit()
        self.db.refresh(meta)
        return meta.normalize()


class ClusteringRepository:
    def __init__(self, database_url, entity_meta):
        self.engine = create_engine(
            database_url,
            echo=False,
            future=True,
            pool_pre_ping=True,
        )

        self.session = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        entity_meta.metadata.create_all(bind=self.engine)
        self.entity_meta = entity_meta
        self.db = scoped_session(self.session)

    def create_row(self, meta: Clustering) -> Dict:
        self.db.add(meta)
        self.db.commit()
        self.db.refresh(meta)
        return meta.normalize()
