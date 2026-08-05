import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ====================================================
# Configuration
# ====================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

INITIAL_EPOCHS = 20

FINE_TUNE_EPOCHS = 10

LEARNING_RATE = 1e-4

train_dir = "dataset/training"

val_dir = "dataset/validation"

SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

# ====================================================
# Data Augmentation
# ====================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=15,

    width_shift_range=0.15,

    height_shift_range=0.15,

    zoom_range=0.20,

    brightness_range=(0.8,1.2),

    shear_range=0.10,

    fill_mode="nearest"

)

validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

# ====================================================
# Data Loader
# ====================================================

train_data = train_datagen.flow_from_directory(

    train_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True

)

validation_data = validation_datagen.flow_from_directory(

    val_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

print("\nClass Mapping")

print(train_data.class_indices)

with open("saved_models/class_indices.json","w") as f:

    json.dump(train_data.class_indices,f,indent=4)

# ====================================================
# Class Weights
# ====================================================

classes = np.unique(train_data.classes)

weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=train_data.classes

)

class_weights = dict(enumerate(weights))

print("\nClass Weights")

print(class_weights)

# ====================================================
# EfficientNetB0
# ====================================================

base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(224,224,3)

)

base_model.trainable = False

# ====================================================
# Build Model
# ====================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.40)(x)

x = Dense(

    256,

    activation="relu"

)(x)

x = Dropout(0.30)(x)

output = Dense(

    train_data.num_classes,

    activation="softmax"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=output

)

# ====================================================
# Compile
# ====================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=LEARNING_RATE

    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ====================================================
# Callbacks
# ====================================================

checkpoint = ModelCheckpoint(

    filepath="saved_models/best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1

)

reduce = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    verbose=1

)

# ====================================================
# First Training
# ====================================================

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        checkpoint,

        early,

        reduce

    ]

)
# ====================================================
# Fine Tuning
# ====================================================

print("\nStarting Fine Tuning...\n")

base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early,
        reduce
    ]
)

# ====================================================
# Save Final Model
# ====================================================

model.save("saved_models/currency_model.keras")

print("\nFinal model saved successfully!")

# ====================================================
# Evaluation
# ====================================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    validation_data,
    verbose=1
)

print(f"\nValidation Accuracy : {accuracy*100:.2f}%")
print(f"Validation Loss     : {loss:.4f}")

# ====================================================
# Predictions
# ====================================================

validation_data.reset()

predictions = model.predict(
    validation_data,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

y_true = validation_data.classes

# ====================================================
# Classification Report
# ====================================================

from sklearn.metrics import classification_report

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(validation_data.class_indices.keys())
    )
)

# ====================================================
# Confusion Matrix
# ====================================================

from sklearn.metrics import confusion_matrix

import seaborn as sns

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(validation_data.class_indices.keys()),
    yticklabels=list(validation_data.class_indices.keys())
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("saved_models/confusion_matrix.png")

plt.show()

# ====================================================
# Accuracy Graph
# ====================================================

train_acc = history.history["accuracy"] + history_fine.history["accuracy"]

val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

plt.figure(figsize=(8,5))

plt.plot(train_acc,label="Train Accuracy")

plt.plot(val_acc,label="Validation Accuracy")

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("saved_models/accuracy.png")

plt.show()

# ====================================================
# Loss Graph
# ====================================================

train_loss = history.history["loss"] + history_fine.history["loss"]

val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(8,5))

plt.plot(train_loss,label="Train Loss")

plt.plot(val_loss,label="Validation Loss")

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("saved_models/loss.png")

plt.show()

# ====================================================
# Model Information
# ====================================================

print("\n===================================")
print("Training Completed Successfully")
print("===================================")
print(f"Classes : {validation_data.class_indices}")
print(f"Images  : {train_data.samples}")
print(f"Validation Images : {validation_data.samples}")
print("Model : EfficientNetB0")
print("Saved Model : saved_models/currency_model.keras")
print("===================================")