import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import json

# ==========================================
# 1. PATH CONFIGURATION (YOUR STRUCTURE)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "bone_xray")

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR   = os.path.join(DATASET_PATH, "val")

MODEL_DIR = os.path.join(BASE_DIR, "train_model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_SAVE_NAME = os.path.join(MODEL_DIR, "bone_model.h5")

CLASS_MAP_PATH = os.path.join(MODEL_DIR, "bone_classes.json")

# ==========================================
# 2. SETTINGS
# ==========================================

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10

# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

print("\nDetected Classes:", class_names)
print("Number of Classes:", NUM_CLASSES)

# ==========================================
# 4. BUILD MODEL (MobileNetV2)
# ==========================================

print("\nBuilding model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining started...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ==========================================
# 7. SAVE MODEL + CLASS MAP
# ==========================================

model.save(MODEL_SAVE_NAME)

with open(CLASS_MAP_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)

print("\n==============================")
print("MODEL SAVED AT:", MODEL_SAVE_NAME)
print("CLASS MAP SAVED AT:", CLASS_MAP_PATH)
print("TRAINING COMPLETE!")
print("==============================")