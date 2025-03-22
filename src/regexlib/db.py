from typing import Dict

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, INTEGER, TEXT

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

EntityMeta = declarative_base()


class Regexes(EntityMeta):
    __tablename__ = f"regexes"

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    # core fields
    old_id = Column(INTEGER, nullable=True)
    title = Column(TEXT, nullable=False)
    pattern = Column(TEXT, nullable=False)
    matching_text = Column(TEXT, nullable=True)
    non_matching_text = Column(TEXT, nullable=True)
    description = Column(TEXT, nullable=True)
    is_dirty = Column(TEXT, nullable=True)
    author_name = Column(TEXT, nullable=True)
    rating = Column(INTEGER, nullable=True)
    date_modified = Column(TEXT, nullable=True)
    # extra fields
    created_at = Column(TEXT, nullable=False)

    def normalize(self):
        return {
            'id': self.id,
            'old_id': self.old_id,
            'title': self.title.__str__(),
            'pattern': self.pattern.__str__(),
            'matching_text': self.matching_text.__str__(),
            'non_matching_text': self.non_matching_text.__str__(),
            'description': self.description.__str__(),
            'is_dirty': self.is_dirty.__str__(),
            'author_name': self.author_name.__str__(),
            'rating': self.rating,
            'date_modified': self.date_modified.__str__(),
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
