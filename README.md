import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

# CONFIGURATION
IMAGE_SIZE = (224, 224)
TRAIN_DIR = '/kaggle/input/soil-classification-part-2/soil_competition-2025/train'         # Folder with only soil images
TEST_DIR = '/kaggle/input/soil-classification-part-2/soil_competition-2025/test'         # Folder with unknown images to classify
AUGMENT_TIMES = 3                 # Number of augmentations per soil image
CONTAMINATION_RATE = 0.08        # Tune between 0.05–0.15

# Preprocess a single image
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# Load all valid images from folder
def load_dataset(folder):
    features, filenames = [], []
    for file in tqdm(os.listdir(folder)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            try:
                img = preprocess_image(os.path.join(folder, file))
                features.append(img)
                filenames.append(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return np.array(features), filenames

# Load EfficientNetB0 model for feature extraction
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# --- STEP 1: Load and Augment Training Data ---
print("Loading soil images...")
X_train_raw, _ = load_dataset(TRAIN_DIR)

print("Augmenting soil images...")
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

augmented_images = []
for img in X_train_raw:
    img = np.expand_dims(img, 0)
    it = datagen.flow(img, batch_size=1)
    for _ in range(AUGMENT_TIMES):
        augmented = next(it)[0]  # ✅ FIXED
        augmented_images.append(augmented)

X_train_aug = np.concatenate([X_train_raw] + [np.array(augmented_images)])

# --- STEP 2: Feature Extraction & Scaling ---
print("Extracting features from training images...")
X_train_features = model.predict(X_train_aug, verbose=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)

# --- STEP 3: Train Isolation Forest Model ---
print("Training IsolationForest model...")
clf = IsolationForest(contamination=CONTAMINATION_RATE, random_state=42)
clf.fit(X_train_scaled)

# --- STEP 4: Load & Predict on Test Images ---
print("Loading test images...")
X_test_raw, test_filenames = load_dataset(TEST_DIR)

print("Extracting features from test images...")
X_test_features = model.predict(X_test_raw, verbose=1)
X_test_scaled = scaler.transform(X_test_features)

print("Predicting with IsolationForest...")
preds = clf.predict(X_test_scaled)        # -1 = not soil, 1 = soil
boolean_preds = (preds == 1).astype(int)  # Convert to 1 (soil) or 0 (not soil)

# --- STEP 5: Save Results to CSV ---
df = pd.DataFrame({
    'image_id': test_filenames,
    'label': boolean_preds
})
df.to_csv('soil_predictionsfinal.csv', index=False)

print("✅ Done! Predictions saved to 'soil_predictionsfinal.csv'")
