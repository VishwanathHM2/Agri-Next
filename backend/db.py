from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)  # store hashed
    full_name = db.Column(db.String(200))
    location = db.Column(db.String(200))
    preferred_language = db.Column(db.String(10), default="en")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SoilRecord(db.Model):
    __tablename__ = "soil_records"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    n = db.Column(db.Float)
    p = db.Column(db.Float)
    k = db.Column(db.Float)
    ph = db.Column(db.Float)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    weather_temp = db.Column(db.Float, nullable=True)
    weather_humidity = db.Column(db.Float, nullable=True)
    rainfall = db.Column(db.Float, nullable=True)
    crop_predicted = db.Column(db.String(200), nullable=True)

    user = db.relationship("User", backref="soil_records")

class CropHistory(db.Model):
    __tablename__ = "crop_history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    crop = db.Column(db.String(200))
    sow_date = db.Column(db.Date)
    harvest_date = db.Column(db.Date)
    notes = db.Column(db.Text)

    user = db.relationship("User", backref="crop_history")

class MarketPrice(db.Model):
    __tablename__ = "market_prices"
    id = db.Column(db.Integer, primary_key=True)
    crop = db.Column(db.String(200))
    price = db.Column(db.Float)
    date = db.Column(db.Date)
    source = db.Column(db.String(200))


class CropSchedule(db.Model):
    __tablename__ = "crop_schedule"
    id = db.Column(db.Integer, primary_key=True)
    crop = db.Column(db.String(100), unique=True)
    duration_days = db.Column(db.Integer)
    watering_weeks = db.Column(db.Integer)
    fertilizer_days = db.Column(db.JSON)  # Store list as JSON
