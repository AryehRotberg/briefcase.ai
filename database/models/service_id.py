from sqlalchemy import Column, String, Integer
from database.base import Base


class ServiceID(Base):
    __tablename__ = "service_ids"

    service_id = Column(Integer, primary_key=True, index=True)
    service_name = Column(String, index=True)
    page_number = Column(Integer, index=True)
