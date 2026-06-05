import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 120

TRAIN_CSV = "dataset_procesado/train/labels.csv"
VAL_CSV = "dataset_procesado/val/labels.csv"


def cargar_dataset(csv_file):
    df = pd.read_csv(csv_file)
    imagenes = []
    labels = []

    for _, row in df.iterrows():
        img = cv2.imread(row["imagen"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        imagenes.append(img)

        labels.append([
            row["x"],
            row["y"],
            row["w"],
            row["h"]
        ])

    return np.array(imagenes), np.array(labels, dtype=np.float32)


x_train, y_train = cargar_dataset(TRAIN_CSV)
x_val, y_val = cargar_dataset(VAL_CSV)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
])

inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = augmentation(inputs)

x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(4, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "modelo_placas_640.keras",
        save_best_only=True,
        monitor="val_loss"
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=20,
        restore_best_weights=True,
        monitor="val_loss"
    )
]

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

model.save("modelo_placas_640_final.keras")

print("Modelo v2 guardado")