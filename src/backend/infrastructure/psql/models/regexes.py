from datetime import datetime

from sqlalchemy import String, DateTime, BigInteger, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.infrastructure.config import db_settings
from src.backend.infrastructure.psql.db import Base
from src.backend.domain.entities.regexes import Sources



class RegexesModel(Base):
    __tablename__ = "regexes"
    __table_args__ = {"schema": db_settings.DB_SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    regex: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[Sources] = mapped_column(String, nullable=False)

    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSON, nullable=False)

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
