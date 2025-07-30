import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import traceback

# === Configuration ===
IMAGE_SIZE = (256, 256)
CLASSIFIER_MODEL_PATH = "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Py. Code/classifier_model.keras"
REGRESSION_MODEL_PATHS = {
    "Ischemic": "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Py. Code/regression_ischemic_model.keras",
    "Haemorrhagic": "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Py. Code/regression_haemorrhagic_model.keras",
}

DATASET_PATHS = {
    "Normal": "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Datasets/Normal",
    "Ischemic": "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Datasets/Ischemic",
    "Haemorrhagic": "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Datasets/Haemorrhagic"
}

# === Load Images for Classification ===
def load_images_for_classification(base_path):
    images, labels = [], []
    label_map = {"Normal": 0, "Ischemic": 1, "Haemorrhagic": 2}
    for label_name, path in base_path.items():
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    try:
                        img = Image.open(os.path.join(root, file)).convert("L").resize(IMAGE_SIZE)
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(label_map[label_name])
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue
    return np.array(images), np.array(labels)

# === Load Images for Regression ===
def load_images_for_regression(path, stroke_type):
    images, labels = [], []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                try:
                    img = Image.open(os.path.join(root, file)).convert("L").resize(IMAGE_SIZE)
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)

                    if stroke_type == "Normal":
                        recovery = 0
                    elif stroke_type == "Ischemic":
                        recovery = np.clip(np.random.normal(loc=3.0, scale=0.8), 1.5, 4.5)
                    elif stroke_type == "Haemorrhagic":
                        recovery = np.clip(np.random.normal(loc=4.5, scale=1.0), 2.5, 6.0)
                    else:
                        recovery = 0
                    labels.append(recovery)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
    return np.array(images), np.array(labels)

# === Model Architectures ===
def build_classifier_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(256, 256, 1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),

        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.00025), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_regression_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(256, 256, 1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),

        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.00025), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

# === Training Function ===
def train_all():
    print("Training classifier...")

    try:
        X_class, y_class = load_images_for_classification(DATASET_PATHS)
        X_class = X_class.reshape(-1, 256, 256, 1)
        X_train, X_val, y_train, y_val = train_test_split(X_class, y_class, test_size=0.2, stratify=y_class)
        classifier = build_classifier_model()
        classifier.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val))
        classifier.save(CLASSIFIER_MODEL_PATH)
        print(f"Classifier model saved to {CLASSIFIER_MODEL_PATH}")
    except Exception as e:
        print(f"Error during classifier training: {e}")
        print(traceback.format_exc())

    for stroke_type in ["Ischemic", "Haemorrhagic"]:
        print(f"Training regression model for {stroke_type}...")

        try:
            X_reg, y_reg = load_images_for_regression(DATASET_PATHS[stroke_type], stroke_type)
            X_reg = X_reg.reshape(-1, 256, 256, 1)
            X_train, X_val, y_train, y_val = train_test_split(X_reg, y_reg, test_size=0.2)
            model = build_regression_model()
            model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val))
            model.save(REGRESSION_MODEL_PATHS[stroke_type])
            print(f"Regression model for {stroke_type} saved to {REGRESSION_MODEL_PATHS[stroke_type]}")
        except Exception as e:
            print(f"Error during regression training for {stroke_type}: {e}")
            print(traceback.format_exc())

# === Main Function ===
if __name__ == "__main__":
    train_all()