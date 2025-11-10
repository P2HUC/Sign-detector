import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import os

# Set page config
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="👋",
    layout="wide"
)

# Title and description
st.title("Sign Language Detection")
st.write("Real-time sign language detection using MediaPipe and machine learning")

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video Transformer Class
class SignLanguageDetector(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.3,
            max_num_hands=2
        )
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            H, W, _ = img.shape
            
            # Process the image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                # Process up to 2 hands
                hands_to_process = results.multi_hand_landmarks[:2]
                
                for hand_landmarks in hands_to_process:
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Collect x, y coordinates
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    
                    # Prepare data for prediction (relative coordinates)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                
                # If only 1 hand detected, pad with zeros
                if len(hands_to_process) == 1:
                    for _ in range(21):  # 21 landmarks per hand
                        data_aux.extend([0, 0])
                
                # Make prediction if we have the right number of features
                if model is not None and len(data_aux) == 84:  # 42 points * 2 (x,y)
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = prediction[0]  # Directly use the predicted character
                    
                    # Draw bounding box and prediction
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(img, predicted_character, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, 
                              cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            return frame

# Main App
def main():
    st.sidebar.title("Settings")
    
    # Instructions
    st.sidebar.markdown("### How to use:")
    st.sidebar.markdown("1. Click 'Start' to begin the webcam")
    st.sidebar.markdown("2. Show hand signs to the camera")
    st.sidebar.markdown("3. The detected sign will be displayed on the screen")
    
    # Model info
    if model is not None:
        st.sidebar.markdown("### Model Status")
        st.sidebar.success("✅ Model is ready")
        if hasattr(model, 'classes_'):
            st.sidebar.write(f"Number of classes: {len(model.classes_)}")
        st.sidebar.markdown("### How to use")
        st.sidebar.write("1. Click 'Start' to begin")
        st.sidebar.write("2. Show hand signs to the camera")
    
    # WebRTC Streamer
    st.markdown("### Live Detection")
    webrtc_ctx = webrtc_streamer(
        key="sign-language-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=SignLanguageDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    if model is None:
        st.error("❌ Failed to load the model. Please check if model.p exists and is valid.")
    else:
        main()
