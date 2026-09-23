import os

DLL_DIRS = [
    r"C:\Users\Lidices\AppData\Local\Programs\Python\Python39\Lib\site-packages\torch\lib",
    r"C:\Program Files\MATLAB\R2025b\bin\win64",
]

for d in DLL_DIRS:
    if os.path.exists(d):
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        os.environ["PATH"] = d + ";" + os.environ["PATH"]

import glob
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Conv2DTranspose, concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = r"D:\LIDICES\PYTHON\bigdata\gabriel\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
# El paper se entrenó con BATCH_SIZE = 16 en Tesla T4 (ver glioma_colab_final_v2.ipynb).
# Aquí queda en 8 para GPUs con menos memoria.
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
IMG_SIZE = 128
SEED = 42

np.random.seed(SEED)        # barajado del generador
tf.random.set_seed(SEED)    # inicialización de pesos y dropout

# ─────────────────────────────────────────────
# BUSCAR ARCHIVOS H5
# ─────────────────────────────────────────────
def get_h5_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True))
    print(f"[INFO] H5 files encontrados: {len(files)}")
    if len(files) == 0:
        raise RuntimeError("No se encontraron archivos .h5")
    return files


# ─────────────────────────────────────────────
# LEER UNA MUESTRA
# ─────────────────────────────────────────────
def read_h5_sample(path, img_size=128):
    with h5py.File(path, "r") as f:
        x = f["image"][()]
        y = f["mask"][()]

    x = np.array(x, dtype=np.float32)   # (240,240,4)
    y = np.array(y, dtype=np.float32)   # (240,240,3)

    # Convertir máscara multicanal a binaria
    # tumor = unión de los 3 canales
    if y.ndim == 3 and y.shape[-1] == 3:
        y = (np.sum(y, axis=-1) > 0).astype(np.float32)
        y = y[..., np.newaxis]  # (240,240,1)
    elif y.ndim == 2:
        y = (y > 0).astype(np.float32)[..., np.newaxis]
    elif y.ndim == 3 and y.shape[-1] == 1:
        y = (y > 0).astype(np.float32)
    else:
        raise RuntimeError(f"Formato de máscara no soportado en {path}: {y.shape}")

    # Resize
    x = tf.image.resize(x, [img_size, img_size], method="bilinear").numpy()
    y = tf.image.resize(y, [img_size, img_size], method="nearest").numpy()
    y = (y > 0).astype(np.float32)

    return x, y


# ─────────────────────────────────────────────
# GENERADOR
# ─────────────────────────────────────────────
class SliceH5Generator(tf.keras.utils.Sequence):
    def __init__(self, file_list, batch_size=16, shuffle=True, img_size=128):
        self.file_list = np.array(file_list)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        self.indices = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = self.file_list[batch_idx]

        X_batch, Y_batch = [], []
        for path in batch_files:
            x, y = read_h5_sample(path, img_size=self.img_size)
            X_batch.append(x)
            Y_batch.append(y)

        return np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ─────────────────────────────────────────────
# MÉTRICAS Y LOSS
# ─────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def specificity_np(y_true, y_pred):
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (tn + 1e-6) / (tn + fp + 1e-6)

def hausdorff_distance_binary(y_true, y_pred):
    pts_true = np.argwhere(y_true > 0)
    pts_pred = np.argwhere(y_pred > 0)

    if len(pts_true) == 0 and len(pts_pred) == 0:
        return 0.0
    if len(pts_true) == 0 or len(pts_pred) == 0:
        return np.nan

    hd1 = directed_hausdorff(pts_true, pts_pred)[0]
    hd2 = directed_hausdorff(pts_pred, pts_true)[0]
    return max(hd1, hd2)


# ─────────────────────────────────────────────
# MODELO U-NET 2D
# ─────────────────────────────────────────────
def conv_block(x, filters, dropout_rate=0.1):
    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def build_unet(input_shape=(128, 128, 4)):
    inputs = Input(shape=input_shape)

    c1 = conv_block(inputs, 32, 0.1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64, 0.1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128, 0.2)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256, 0.2)
    p4 = MaxPooling2D((2, 2))(c4)

    bn = conv_block(p4, 512, 0.3)

    u6 = Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 256, 0.2)

    u7 = Conv2DTranspose(128, 2, strides=2, padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 128, 0.2)

    u8 = Conv2DTranspose(64, 2, strides=2, padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 64, 0.1)

    u9 = Conv2DTranspose(32, 2, strides=2, padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 32, 0.1)

    outputs = Conv2D(1, 1, activation="sigmoid")(c9)
    return Model(inputs, outputs, name="UNet2D_H5Slices")


# ─────────────────────────────────────────────
# CURVAS
# ─────────────────────────────────────────────
def plot_learning_curves(history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Dice Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["dice_coefficient"], label="train_dice")
    plt.plot(history.history["val_dice_coefficient"], label="val_dice")
    plt.title("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("learning_curves_h5slices.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────
def evaluate_model(model, test_files, img_size=128, threshold=0.5):
    dices = []
    recalls = []
    specs = []
    hds = []

    for path in test_files:
        x, y_true = read_h5_sample(path, img_size=img_size)
        y_prob = model.predict(x[np.newaxis, ...], verbose=0)[0]
        y_pred = (y_prob >= threshold).astype(np.uint8)

        yt = y_true[..., 0].astype(np.uint8)
        yp = y_pred[..., 0].astype(np.uint8)

        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))

        dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        spec = specificity_np(yt, yp)
        hd = hausdorff_distance_binary(yt, yp)

        dices.append(dice)
        recalls.append(recall)
        specs.append(spec)
        if not np.isnan(hd):
            hds.append(hd)

    print("\n========== RESULTADOS TEST ==========")
    print(f"Dice:         {np.mean(dices):.4f}")
    print(f"Sensitivity:  {np.mean(recalls):.4f}")
    print(f"Specificity:  {np.mean(specs):.4f}")
    print(f"Hausdorff:    {np.mean(hds):.4f}" if len(hds) > 0 else "Hausdorff:    NaN")


# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────
def visualize_predictions(model, test_files, img_size=128, n_samples=4, threshold=0.5):
    chosen = test_files[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, path in enumerate(chosen):
        x, y_true = read_h5_sample(path, img_size=img_size)
        y_prob = model.predict(x[np.newaxis, ...], verbose=0)[0]
        y_pred = (y_prob >= threshold).astype(np.float32)

        flair = x[:, :, 0]  # si quieres luego lo cambiamos a otro canal

        axes[i, 0].imshow(flair, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(y_true[:, :, 0], cmap="hot")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(y_pred[:, :, 0], cmap="hot")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("qualitative_results_h5slices.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    all_files = get_h5_files(DATA_DIR)

    # Prueba rápida
    x0, y0 = read_h5_sample(all_files[0], img_size=IMG_SIZE)
    print("[INFO] shape imagen:", x0.shape)
    print("[INFO] shape máscara:", y0.shape)

    train_files, test_files = train_test_split(
        all_files, test_size=0.10, random_state=SEED
    )
    train_files, val_files = train_test_split(
        train_files, test_size=0.15 / 0.90, random_state=SEED
    )

    print(f"[INFO] train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    train_gen = SliceH5Generator(train_files, batch_size=BATCH_SIZE, shuffle=True, img_size=IMG_SIZE)
    val_gen = SliceH5Generator(val_files, batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)

    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 4))
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss=dice_loss,
        metrics=[dice_coefficient, tf.keras.metrics.Recall(name="sensitivity")]
    )

    model.summary()

    callbacks = [
        ModelCheckpoint("best_unet_h5slices.keras", monitor="val_dice_coefficient", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    plot_learning_curves(history)
    evaluate_model(model, test_files, img_size=IMG_SIZE)
    visualize_predictions(model, test_files, img_size=IMG_SIZE, n_samples=4)

    print("\n[DONE] Entrenamiento y evaluación completados.")