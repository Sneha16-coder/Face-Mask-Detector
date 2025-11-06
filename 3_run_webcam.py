import cv2
import numpy as np
from keras.models import load_model

# --- 1. Load the "Brain" and the "Face Finder" ---
print("Loading model...")
model = load_model("face_mask_detector.keras")

print("Loading face cascade...")
# Note: You must have this file in the same folder
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the cascade file loaded (a good safety check)
if face_cascade.empty():
    print("Error: Could not load haarcascade file.")
    print("Make sure 'haarcascade_frontalface_default.xml' is in the same folder.")
    exit()

# --- 2. Define our labels and colors ---
# We know from Phase 2 that [1, 0] is "WithMask" (index 0)
# and [0, 1] is "WithoutMask" (index 1)
LABELS = ["Mask", "No Mask"]
COLORS = [(0, 255, 0), (0, 0, 255)] # Green for Mask, Red for No Mask
IMG_SIZE = 128 # Must be the same as our training image size

# --- 3. Start the Webcam ---
print("Starting video stream... Press 'q' to quit.")
# cv2.VideoCapture(0) uses your default webcam.
# If you have multiple, you might need to try 1 or 2.
cap = cv2.VideoCapture(0)

while True:
    # Read one frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break # If frame can't be read, stop
        
    # Flip the frame horizontally. It feels more natural, like a mirror.
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale for the face finder
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find all faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    # Loop over each face it found
    for (x, y, w, h) in faces:
        
        # --- 4. PREPARE THE FACE for the "Brain" ---
        # We must do the *exact* same preprocessing as in Phase 2

        # 1. Extract the face from the frame
        face_roi = frame[y:y+h, x:x+w]
        
        # 2. Convert to RGB (model was trained on RGB)
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # 3. Resize to 128x128
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        
        # 4. Normalize (0-1)
        face_normalized = face_resized / 255.0
        
        # 5. Add a "batch" dimension (model expects a list of images)
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        # --- 5. MAKE THE PREDICTION ---
        prediction = model.predict(face_batch, verbose=0)
        
        # 'prediction' will be something like [[0.98, 0.02]]
        # np.argmax finds the *index* of the highest number
        label_index = np.argmax(prediction) # This will be 0 or 1
        
        label = LABELS[label_index]
        color = COLORS[label_index]
        confidence = prediction[0][label_index]
        
        # Format the text to show on screen
        text = f"{label} - {confidence * 100:.2f}%"
        
        # --- 6. DRAW THE BOX and TEXT ---
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- 7. Display the final frame ---
    cv2.imshow('Face Mask Detection (Press "q" to quit)', frame)
    
    # Check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 8. Cleanup ---
print("Stopping video stream.")
cap.release()
cv2.destroyAllWindows()