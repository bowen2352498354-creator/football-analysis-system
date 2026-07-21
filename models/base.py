# -*- coding: utf-8 -*-
"""SQLAlchemy 声明式基类。"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """全部科研 ORM 模型的公共基类。"""
