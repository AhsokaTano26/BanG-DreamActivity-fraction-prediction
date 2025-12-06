from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class Event(Base):
    __tablename__ = 'event'
    ID = Column(String(100), primary_key=True, nullable=False)
    EventID = Column(Integer)
    EventBand = Column(String(100))
    EventName = Column(String(100))
    EventType = Column(String(100))
    StartAt = Column(Integer)
    EndAt = Column(Integer)
    Rank = Column(Integer)
    PointRank = Column(String(100))
    Country = Column(String(100))