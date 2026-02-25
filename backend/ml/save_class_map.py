import os, json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset (PlantVillage)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "plantvillage")

# Where to save the mapping
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "class_indices.json")

# Just initialize the generator (no training)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Save class indices to JSON
with open(CLASS_MAP_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

print("✅ Saved class mapping to", CLASS_MAP_PATH)
