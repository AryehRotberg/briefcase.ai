from sqlalchemy import Column, String, Integer
from database.base import Base


class Policy(Base):
    __tablename__ = "policies"

    point_id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=False)
    source = Column(String, index=False)
    service_id = Column(Integer, index=True)
    service_name = Column(String, index=True)


class PolicyProcessed(Base):
    __tablename__ = "policies_processed"

    point_id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=False)
    source = Column(String, index=False)
    service_id = Column(Integer, index=True)
    service_name = Column(String, index=True)
