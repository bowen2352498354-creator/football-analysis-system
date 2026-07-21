# -*- coding: utf-8 -*-
"""本地 SQLite 引擎与会话工厂（边缘计算 / 数据不出域）。"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from models import Base

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "cluster_rct.db")

# 可通过环境变量覆盖；默认落在项目根目录本地文件，符合全边缘架构
DATABASE_URL = os.environ.get(
    "CLUSTER_RCT_DATABASE_URL",
    f"sqlite:///{DEFAULT_DB_PATH.replace(os.sep, '/')}",
)


def _configure_sqlite(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    resolved = url or DATABASE_URL
    connect_args = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
    engine = create_engine(resolved, echo=echo, future=True, connect_args=connect_args)
    _configure_sqlite(engine)
    return engine


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(bind: Engine | None = None) -> None:
    """创建全部科研表（含伦理映射表）。"""
    # 确保模型注册进 metadata
    import models  # noqa: F401

    Base.metadata.create_all(bind or engine)


@contextmanager
def session_scope(bind: Engine | None = None) -> Generator[Session, None, None]:
    """事务作用域：成功 commit，异常 rollback。"""
    factory = sessionmaker(bind=bind or engine, autoflush=False, autocommit=False, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
