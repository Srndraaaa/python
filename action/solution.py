# Import semua library yang dibutuhkan
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB3, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Machine Learning untuk clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seed untuk reproduksibilitas
np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# KONFIGURASI PARAMETER
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Direktori data
TRAIN_DIR = 'd:/Vscode/python/action/train'
TEST_DIR = 'd:/Vscode/python/action/test'

# Label kategori makanan
FOOD_LABELS = [
    'Ayam Bakar', 'Ayam Betutu', 'Ayam Goreng', 'Ayam Pop',
    'Bakso', 'Coto Makassar', 'Gado Gado', 'Gudeg',
    'Nasi Goreng', 'Pempek', 'Rawon', 'Rendang',
    'Sate Madura', 'Sate Padang', 'Soto'
]
NUM_CLASSES = len(FOOD_LABELS)

print("="*60)
print("ACTION 2025 - Image Classification Competition")
print("Label Discovery & Model Training")
print("="*60)

# ===============================
# TAHAP 1: EXTRACT FITUR DARI GAMBAR
# ===============================
print("\n[TAHAP 1] Mengekstrak fitur dari gambar training...")

def load_and_preprocess_image(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Membaca dan preprocessing gambar"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

# Gunakan pre-trained model untuk ekstraksi fitur
base_model = EfficientNetB3(weights='imagenet', include_top=False, 
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
feature_extractor = Model(inputs=base_model.input, 
                          outputs=GlobalAveragePooling2D()(base_model.output))

# Load semua gambar training
train_images = []
train_filenames = []

for filename in tqdm(sorted(os.listdir(TRAIN_DIR))):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(TRAIN_DIR, filename)
        img = load_and_preprocess_image(img_path)
        if img is not None:
            train_images.append(img)
            train_filenames.append(filename)

train_images = np.array(train_images)
print(f"Total gambar training: {len(train_images)}")

# Ekstrak fitur menggunakan model pre-trained
print("\nMengekstrak fitur deep learning...")
train_features = feature_extractor.predict(train_images, batch_size=32, verbose=1)
print(f"Shape fitur: {train_features.shape}")

# ===============================
# TAHAP 2: LABEL DISCOVERY DENGAN CLUSTERING
# ===============================
print("\n[TAHAP 2] Melakukan label discovery menggunakan K-Means clustering...")

# Reduce dimensi menggunakan PCA untuk clustering yang lebih baik
print("Mengurangi dimensi dengan PCA...")
pca = PCA(n_components=128, random_state=42)
features_reduced = pca.fit_transform(train_features)
print(f"Variance dijelaskan oleh PCA: {pca.explained_variance_ratio_.sum():.4f}")

# Clustering dengan K-Means
print(f"\nMelakukan K-Means clustering dengan {NUM_CLASSES} cluster...")
kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=20, max_iter=500)
predicted_labels = kmeans.fit_predict(features_reduced)

# Mapping cluster ke label makanan
# Kita perlu assign cluster ke kategori makanan yang paling cocok
cluster_to_label = {}
for i in range(NUM_CLASSES):
    cluster_to_label[i] = FOOD_LABELS[i]

# Assign label ke setiap gambar
train_labels = [cluster_to_label[label] for label in predicted_labels]

# Buat DataFrame untuk tracking
train_df = pd.DataFrame({
    'filename': train_filenames,
    'cluster': predicted_labels,
    'label': train_labels
})

print("\nDistribusi label yang ditemukan:")
print(train_df['label'].value_counts())

# ===============================
# TAHAP 3: TRAINING MODEL KLASIFIKASI
# ===============================
print("\n[TAHAP 3] Training model klasifikasi...")

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_labels)
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=NUM_CLASSES)

# Split data untuk validasi
X_train, X_val, y_train, y_val = train_test_split(
    train_images, y_categorical, test_size=0.15, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} gambar")
print(f"Validation set: {X_val.shape[0]} gambar")

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

# ===============================
# TAHAP 4: MEMBANGUN ENSEMBLE MODEL
# ===============================
print("\n[TAHAP 4] Membangun ensemble model...")

def create_model(base_model_func, model_name):
    """Membuat model dengan transfer learning"""
    print(f"\nMembangun model {model_name}...")
    
    base = base_model_func(weights='imagenet', include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze beberapa layer awal
    for layer in base.layers[:-30]:
        layer.trainable = False
    
    # Build model
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                               min_lr=1e-7, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, 
                           verbose=1)

# Model 1: EfficientNetB3
print("\n" + "="*60)
print("TRAINING MODEL 1: EfficientNetB3")
print("="*60)
model1 = create_model(EfficientNetB3, "EfficientNetB3")

history1 = model1.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

# Evaluasi Model 1
val_pred1 = model1.predict(X_val, verbose=0)
val_pred1_classes = np.argmax(val_pred1, axis=1)
val_true_classes = np.argmax(y_val, axis=1)
acc1 = accuracy_score(val_true_classes, val_pred1_classes)
print(f"\nModel 1 Validation Accuracy: {acc1:.4f}")

# Model 2: ResNet50
print("\n" + "="*60)
print("TRAINING MODEL 2: ResNet50")
print("="*60)
model2 = create_model(ResNet50, "ResNet50")

history2 = model2.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

# Evaluasi Model 2
val_pred2 = model2.predict(X_val, verbose=0)
val_pred2_classes = np.argmax(val_pred2, axis=1)
acc2 = accuracy_score(val_true_classes, val_pred2_classes)
print(f"\nModel 2 Validation Accuracy: {acc2:.4f}")

# ===============================
# TAHAP 5: ENSEMBLE PREDICTION
# ===============================
print("\n[TAHAP 5] Membuat ensemble prediction...")

# Ensemble validation
ensemble_val_pred = (val_pred1 + val_pred2) / 2
ensemble_val_classes = np.argmax(ensemble_val_pred, axis=1)
ensemble_acc = accuracy_score(val_true_classes, ensemble_val_classes)

print(f"\nEnsemble Validation Accuracy: {ensemble_acc:.4f}")
print("\nClassification Report:")
print(classification_report(val_true_classes, ensemble_val_classes, 
                           target_names=label_encoder.classes_))

# ===============================
# TAHAP 6: PREDIKSI TEST DATA
# ===============================
print("\n[TAHAP 6] Memprediksi test data...")

# Load test images
test_images = []
test_filenames = []

for filename in tqdm(sorted(os.listdir(TEST_DIR))):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(TEST_DIR, filename)
        img = load_and_preprocess_image(img_path)
        if img is not None:
            test_images.append(img)
            test_filenames.append(filename)

test_images = np.array(test_images)
print(f"Total gambar test: {len(test_images)}")

# Prediksi dengan ensemble
print("\nMelakukan prediksi dengan ensemble model...")
test_pred1 = model1.predict(test_images, batch_size=32, verbose=1)
test_pred2 = model2.predict(test_images, batch_size=32, verbose=1)

# Ensemble prediction
ensemble_test_pred = (test_pred1 + test_pred2) / 2
test_pred_classes = np.argmax(ensemble_test_pred, axis=1)
test_pred_labels = label_encoder.inverse_transform(test_pred_classes)

# ===============================
# TAHAP 7: BUAT SUBMISSION FILE
# ===============================
print("\n[TAHAP 7] Membuat submission file...")

# Buat submission dataframe
submission_df = pd.DataFrame({
    'ID': range(1, len(test_pred_labels) + 1),
    'label': test_pred_labels
})

# Save submission
submission_path = 'd:/Vscode/python/action/submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission file berhasil dibuat: {submission_path}")
print("\nPreview submission:")
print(submission_df.head(20))

print("\nDistribusi prediksi:")
print(submission_df['label'].value_counts().sort_index())

# ===============================
# VISUALISASI HASIL
# ===============================
print("\n[BONUS] Membuat visualisasi...")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Model 1
axes[0, 0].plot(history1.history['accuracy'], label='Train')
axes[0, 0].plot(history1.history['val_accuracy'], label='Val')
axes[0, 0].set_title('Model 1 (EfficientNetB3) - Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history1.history['loss'], label='Train')
axes[0, 1].plot(history1.history['val_loss'], label='Val')
axes[0, 1].set_title('Model 1 (EfficientNetB3) - Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Model 2
axes[1, 0].plot(history2.history['accuracy'], label='Train')
axes[1, 0].plot(history2.history['val_accuracy'], label='Val')
axes[1, 0].set_title('Model 2 (ResNet50) - Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history2.history['loss'], label='Train')
axes[1, 1].plot(history2.history['val_loss'], label='Val')
axes[1, 1].set_title('Model 2 (ResNet50) - Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('d:/Vscode/python/action/training_history.png', dpi=300, bbox_inches='tight')
print("Grafik training history disimpan: training_history.png")

# Confusion matrix
cm = confusion_matrix(val_true_classes, ensemble_val_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Ensemble Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('d:/Vscode/python/action/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix disimpan: confusion_matrix.png")

print("\n" + "="*60)
print("SELESAI!")
print("="*60)
print(f"\nRingkasan:")
print(f"- Model 1 (EfficientNetB3) Accuracy: {acc1:.4f}")
print(f"- Model 2 (ResNet50) Accuracy: {acc2:.4f}")
print(f"- Ensemble Accuracy: {ensemble_acc:.4f}")
print(f"- Total prediksi test: {len(test_pred_labels)}")
print(f"- File submission: submission.csv")
print("\nModel siap untuk kompetisi! 🚀")
