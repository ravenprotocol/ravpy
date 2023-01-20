from __future__ import annotations

import datetime
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import orm

metadata = sa.MetaData()
Base = orm.declarative_base(metadata=metadata)


class Subgraph(Base):
    __tablename__ = 'subgraphs'
    id = Column(Integer, primary_key=True)
    graph_id = Column(Integer, nullable=False)
    subgraph_id = Column(Integer, nullable=False)
    status = Column(String(50), nullable=True, default=None)
    progress = Column(Float, nullable=True, default=None)
    tokens = Column(Float, nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def as_dict(self):
        values = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        values['created_at'] = values['created_at'].strftime("%Y-%m-%d %H:%M:%S")
        return values
