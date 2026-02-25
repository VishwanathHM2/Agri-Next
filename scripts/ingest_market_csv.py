import os
import pandas as pd
from backend.db import db, MarketPrice
from backend.app import app

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "market_prices.csv")

def ingest():
    df = pd.read_csv(CSV)
    with app.app_context():
        for _, row in df.iterrows():
            m = MarketPrice(crop=row['crop'], price=float(row['price']), date=pd.to_datetime(row['date']).date(), source=row.get('source','csv'))
            db.session.add(m)
        db.session.commit()
        print("Ingested market CSV.")

if __name__ == "__main__":
    ingest()
