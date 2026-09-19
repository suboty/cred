from datetime import datetime

from sqlalchemy import String, DateTime, BigInteger, func
from sqlalchemy.orm import Mapped, mapped_column

from infrastructure.config import db_settings
from infrastructure.psql.db import Base


class Regex101Model(Base):
    __tablename__ = "source__regex101"
    __table_args__ = {"schema": db_settings.DB_SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    permalink: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    regex: Mapped[str] = mapped_column(String, nullable=False)
    flags: Mapped[str] = mapped_column(String, nullable=True)
    delimiter: Mapped[str] = mapped_column(String, nullable=False)
    dialect: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)

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
