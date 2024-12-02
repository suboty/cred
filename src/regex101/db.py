from typing import Dict

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, INTEGER, TEXT, REAL

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

EntityMeta = declarative_base()


class Regexes(EntityMeta):
    __tablename__ = f"regexes"

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    # core fields
    regex = Column(TEXT, nullable=False)
    test_string = Column(TEXT, nullable=True)
    flags = Column(TEXT, nullable=True)
    delimiter = Column(TEXT, nullable=True)
    dialect = Column(TEXT, nullable=False)
    title = Column(TEXT, nullable=True)
    description = Column(TEXT, nullable=True)
    # extra fields
    created_at = Column(TEXT, nullable=False)

    def normalize(self):
        return {
            'id': self.id,
            'regex': self.regex.__str__(),
            'test_string': self.test_string.__str__(),
            'flags': self.flags.__str__(),
            'delimiter': self.delimiter.__str__(),
            'dialect': self.dialect.__str__(),
            'title': self.title.__str__(),
            'description': self.description.__str__(),
            'created_at': self.created_at.__str__(),
        }


class DBRepository:
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

    def create_regex_row(self, meta: Regexes) -> Dict:
        self.db.add(meta)
        self.db.commit()
        self.db.refresh(meta)
        return meta.normalize()
