import numpy as np
import matplotlib.pyplot as plt



# --- 1. Load our saved data ---
print("Loading pre-processed data...")
trainData = np.load('train_data.npy')
trainLabels = np.load('train_labels.npy')
testData = np.load('test_data.npy')
testLabels = np.load('test_labels.npy')

print(f"Training data shape: {trainData.shape}")
print(f"Training labels shape: {trainLabels.shape}")

# --- 2. Build the AI "Brain" (CNN Model) ---
print("Building the model...")

model = Sequential()
IMG_SIZE = 128 # Must be the same size as we used in the last step

# Block 1: The first "look"
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2: A deeper "look"
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3: An even deeper "look"
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout layer to prevent "over-thinking" (overfitting)
model.add(Dropout(0.25))

# --- 3. Prepare for Decision Making ---
model.add(Flatten()) # Flattens the 3D image data into a 1D line
model.add(Dense(128, activation='relu')) # A layer of 128 "neurons"
model.add(Dropout(0.5))

# The Final Output Layer
# We use 2 neurons because we have 2 classes: (Mask, No Mask)
# We use 'softmax' because it will give us a probability (e.g., 98% Mask, 2% No Mask)
model.add(Dense(2, activation='softmax')) 

# --- 4. Compile the Model ---
# This "assembles" the model we just designed
model.compile(
    loss='categorical_crossentropy', # Good for 2+ classes
    optimizer=Adam(learning_rate=0.0001), # The 'adam' optimizer is a solid default
    metrics=['accuracy']
)

# Let's see our model's architecture
print(model.summary())

# --- 5. Train the Model! ---
EPOCHS = 15 # How many times to show the AI all the images
print(f"Training model for {EPOCHS} epochs...")

history = model.fit(
    trainData, 
    trainLabels, 
    batch_size=32,
    epochs=EPOCHS,
    validation_data=(testData, testLabels) # We use the 'test' data to check its work
)

# --- 6. Save the Final, Trained Model ---
# This is the most important part!
model.save("face_mask_detector.keras")
print("All done. Model saved as 'face_mask_detector.keras'")


# --- 7. Plot the Results ---
print("Displaying training graphs...")

plt.style.use("ggplot")
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.suptitle("Training Loss and Accuracy")
plt.show()