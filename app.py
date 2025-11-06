import cv2
import numpy as np
import streamlit as st # This is the main streamlit library
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- 1. LOAD OUR MODELS ---
# We load them outside the main function so they only load ONCE
st.set_page_config(page_title="Real-Time Face Mask Detection")

@st.cache_resource  # This caches the model so it doesn't reload
def load_all_models():
    print("Loading models... This will only run once.")
    model = load_model("face_mask_detector.keras")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_all_models()

# --- 2. DEFINE LABELS AND COLORS ---
LABELS = ["Mask", "No Mask"]
COLORS = [(0, 255, 0), (0, 0, 255)] # Green for Mask, Red for No Mask
IMG_SIZE = 128

# --- 3. CREATE THE VIDEO PROCESSING CLASS ---
# This is the core of the Streamlit-webrtc component

class MaskDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.face_cascade = face_cascade

    def recv(self, frame):
        # Convert the frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")
        
        # --- Start of your detection logic from 3_run_webcam.py ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            prediction = self.model.predict(face_batch, verbose=0)
            label_index = np.argmax(prediction)
            
            label = LABELS[label_index]
            color = COLORS[label_index]
            confidence = prediction[0][label_index]
            
            text = f"{label} - {confidence * 100:.2f}%"
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # --- End of your detection logic ---
        
        # Return the processed frame
        return img

# --- 4. SETUP THE STREAMLIT APP INTERFACE ---
st.title("Real-Time Face Mask Detection")
st.write("This app uses your webcam to detect face masks in real-time.")
st.write("Click 'START' to begin. You may need to grant camera permissions.")

# This is the magic component that handles all the webcam streaming
webrtc_streamer(
    key="mask-detection",
    video_transformer_factory=MaskDetector,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    media_stream_constraints={"video": True, "audio": False},
)

st.subheader("How it works:")
st.markdown("""
- The app uses **OpenCV** to find a face (using a Haarcascade).
- The detected face is pre-processed and fed into a **Deep Learning (CNN)** model.
- The model (trained in Keras/TensorFlow) predicts **'Mask'** or **'No Mask'**.
- A **<font color='green'>Green (Mask)</font>** or **<font color='red'>Red (No Mask)</font>** box is drawn on the video feed.
""", unsafe_allow_html=True)