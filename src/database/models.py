import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class InferenceLog(Base):  # type:ignore
    __tablename__ = "inference_logs"

    id = sa.Column(sa.Integer, primary_key=True)
    city = sa.Column(sa.String)
    district = sa.Column(sa.String)
    latitude = sa.Column(sa.Float)
    longitude = sa.Column(sa.Float)
    street = sa.Column(sa.String)
    floor_area_m2 = sa.Column(sa.Float)
    number_of_rooms = sa.Column(sa.Float)
    floor = sa.Column(sa.Float)
    number_of_floors = sa.Column(sa.Float)
    build_year = sa.Column(sa.Float)
    building_type = sa.Column(sa.String)
    heating_type = sa.Column(sa.String)
    equipment = sa.Column(sa.String)
    inferred_monthly_rent = sa.Column(sa.Float)
