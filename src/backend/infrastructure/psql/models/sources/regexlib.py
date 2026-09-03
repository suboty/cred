from datetime import datetime

from sqlalchemy import String, DateTime, BigInteger, func, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.infrastructure.config import db_settings
from src.backend.infrastructure.psql.db import Base


class RegexLibModel(Base):
    __tablename__ = "source__regexlib"
    __table_args__ = {"schema": db_settings.DB_SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    title: Mapped[str] = mapped_column(String, nullable=False)
    pattern: Mapped[str] = mapped_column(String, nullable=False)
    matching_text: Mapped[str] = mapped_column(String, nullable=False)
    non_matching_text: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    is_dirty: Mapped[int] = mapped_column(Integer, nullable=False)
    author_name: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)

    source_date_modified: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
        onupdate=func.now(),
    )
