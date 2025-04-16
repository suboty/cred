import time

from sqlalchemy import Column, INTEGER, TEXT, BOOLEAN, FLOAT, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker


EntityMeta = declarative_base()

CREATED_TIME = int(time.time())
CLUSTERING_TABLE_NAME = f'clustering_{CREATED_TIME}'
EXPERIMENTS_TABLE_NAME = f'experiments_{CREATED_TIME}'
RESULTS_TABLE_NAME = f'results_{CREATED_TIME}'
REGEXES_TABLE_NAME = f'regexes_{CREATED_TIME}'


class Regexes(EntityMeta):
    __tablename__ = REGEXES_TABLE_NAME

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


class Experiments(EntityMeta):
    __tablename__ = EXPERIMENTS_TABLE_NAME

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    # core fields
    vectorizer = Column(TEXT, nullable=False)
    cluster_number = Column(INTEGER, nullable=False)
    clustering_algorithm = Column(TEXT, nullable=False)
    input_data_shape = Column(TEXT, nullable=False)
    # extra fields
    created_at = Column(TEXT, nullable=False)


class Clustering(EntityMeta):
    __tablename__ = CLUSTERING_TABLE_NAME

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    # core fields
    regex_id = Column(INTEGER, ForeignKey(f'{REGEXES_TABLE_NAME}.id'), nullable=False)
    experiment_id = Column(INTEGER, ForeignKey(f'{EXPERIMENTS_TABLE_NAME}.id'), nullable=False)
    cluster_id = Column(INTEGER, nullable=False)
    # extra fields
    created_at = Column(TEXT, nullable=False)


class Results(EntityMeta):
    __tablename__ = CLUSTERING_TABLE_NAME

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    # core fields
    experiment_id = Column(INTEGER, ForeignKey(f'{EXPERIMENTS_TABLE_NAME}.id'), nullable=False)
    metric_name = Column(TEXT, nullable=False)
    metric_value = Column(FLOAT, nullable=False)


class ResearchRepository:
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

    def create_regex(self, meta: Regexes) -> None:
        if not meta.is_ast:
            meta.is_ast = False
        if not meta.is_preprocessed:
            meta.is_preprocessed = False
        self.db.add(meta)
        self.db.commit()
        self.db.refresh(meta)

    def create_experiment(self, meta: Experiments) -> None:
        self.db.add(meta)
        self.db.commit()
        self.db.refresh(meta)

    def create_result(self, meta: Results) -> None:
        self.db.add(meta)
        self.db.commit()
        self.db.refresh(meta)

    def create_clustering(self, meta: Clustering) -> None:
        self.db.add(meta)
        self.db.commit()
        self.db.refresh(meta)
