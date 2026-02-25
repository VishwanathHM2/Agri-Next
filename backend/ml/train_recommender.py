import os
from recommender import train_recommender

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "npk_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Place your dataset as data/npk_dataset.csv")
    train_recommender(csv_path)

