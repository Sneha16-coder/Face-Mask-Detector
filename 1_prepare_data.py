import os
import cv2 # This is the OpenCV library
import numpy as np # This is for math
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# --- IMPORTANT: CHANGE THIS ---
# Point this to the "Train" folder inside the dataset you downloaded
DATA_DIR = "C:/Users/sneha123415/OneDrive/Desktop/Mask_detector/Face Mask Dataset/Train" 
# -----------------------------

CATEGORIES = ["WithMask", "WithoutMask"]
IMG_SIZE = 128 # We will resize all images to 128x128 pixels

# These are empty lists to hold our data
data = []
labels = []

print("Starting image loading...")

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category) # Path to "WithMask" or "WithoutMask"
        class_num = CATEGORIES.index(category) # 0 for "WithMask", 1 for "WithoutMask"
        
        # Loop over all images in the folder
        for img_name in os.listdir(path):
            try:
                # 1. Load the image
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                
                # 2. Convert to RGB (OpenCV's default is BGR, which is weird)
                img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                
                # 3. Resize to our standard 128x128 size
                new_array = cv2.resize(img_array_rgb, (IMG_SIZE, IMG_SIZE))
                
                # 4. Add the image and its label to our lists
                data.append(new_array)
                labels.append(class_num)
            except Exception as e:
                # This will skip any broken images
                print(f"Error loading image {img_path}: {e}")

# Run the function!
create_training_data()

print(f"Loading complete. Total images loaded: {len(data)}")

# --- Now, we format the data for the AI ---

print("Formatting data...")

# 1. Normalize and convert to numpy arrays
# We divide by 255.0 to turn pixel values (0-255) into a scale of 0.0 to 1.0.
# AI models work much better with this.
data = np.array(data) / 255.0
labels = np.array(labels)

# 2. One-hot encode the labels
# This turns 0 -> [1, 0] (for 'WithMask')
# and turns 1 -> [0, 1] (for 'WithoutMask')
# This is the format our specific AI model will need.
labels = to_categorical(labels)

# 3. Split the data
# We use 80% for training and 20% for testing.
# This lets us check if the AI *actually* learned, or just memorized the pictures.
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# 4. Save our prepped data to disk
# This is a HUGE time-saver. We don't want to do all that
# loading and resizing ever again.
print("Saving pre-processed data to files...")
np.save('train_data.npy', trainData)
np.save('train_labels.npy', trainLabels)
np.save('test_data.npy', testData)
np.save('test_labels.npy', testLabels)

print("All done. Your data is prepped and saved.")