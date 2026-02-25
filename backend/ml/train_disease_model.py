import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "plantvillage")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "disease_cnn.h5")
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "class_indices.json")

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(224,224,3)),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError("Download PlantVillage dataset and put under data/plantvillage/")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                 rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True)
    train_gen = datagen.flow_from_directory(DATA_DIR, target_size=(224,224), batch_size=32, class_mode="categorical", subset="training")
    val_gen = datagen.flow_from_directory(DATA_DIR, target_size=(224,224), batch_size=32, class_mode="categorical", subset="validation")
    num_classes = train_gen.num_classes
    model = build_model(num_classes)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    model.fit(train_gen, validation_data=val_gen, epochs=12, callbacks=[checkpoint])
    print("Saved disease model to", MODEL_PATH)

if __name__ == "__main__":
    train()
