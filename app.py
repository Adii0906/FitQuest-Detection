import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def add_styling():
    st.markdown(
        """
        <style>
            .quest-box {
                background: linear-gradient(to bottom, #0d1b2a, #1b263b);
                padding: 20px;
                border-radius: 15px;
                color: white;
                font-family: 'Arial', sans-serif;
                text-align: center;
                border: 2px solid #415a77;
            }
            .goal-text {
                color: #00ff99;
                font-size: 20px;
                margin-bottom: 10px;
            }
            .warning {
                color: #ff4444;
                font-weight: bold;
                font-size: 16px;
            }
            .penalty {
                color: #ffcc00;
                font-weight: bold;
                font-size: 16px;
            }
            .start-button {
                background-color: #1b9aaa;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def process_image(image, goal, exercise_type):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process image with MediaPipe
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Example: Display the image with landmarks
        st.image(image, channels="RGB", use_container_width=True)
    else:
        st.warning("No pose landmarks detected. Please ensure your body is visible.")

# Initialize session state for push-up and squat counts
if "pushup_count" not in st.session_state:
    st.session_state.pushup_count = 0
if "squat_count" not in st.session_state:
    st.session_state.squat_count = 0

st.title("🏆 Quest Tracker - Become a Warrior! 🏋️")
add_styling()

# Create a placeholder for the quest box
quest_placeholder = st.empty()

# Update the quest box with current counts
quest_placeholder.markdown(f"""
<div class='quest-box'>
    <h2>DAILY QUEST - TRAIN TO BECOME A FORMIDABLE COMBATANT</h2>
    <p class='goal-text'>GOALS</p>
    <p>- PUSH-UPS [{st.session_state.pushup_count}/100]</p>
    <p>- SQUATS [{st.session_state.squat_count}/100]</p>
</div>
""", unsafe_allow_html=True)

task = st.selectbox("Choose Exercise", ["Push-up", "Squat"])
goal = st.number_input("Set Goal", min_value=1, value=10)

# Capture image from the user's camera
img = st.camera_input("Take a photo for analysis")

if img is not None:
    # Process the image
    image = Image.open(img)
    process_image(image, goal, task)

if st.button("Reset", key="reset_button"):
    st.session_state.pushup_count = 0
    st.session_state.squat_count = 0
    
    # Update the quest box after reset
    quest_placeholder.markdown(f"""
    <div class='quest-box'>
        <h2>DAILY QUEST - TRAIN TO BECOME A FORMIDABLE COMBATANT</h2>
        <p class='goal-text'>GOALS</p>
        <p>- PUSH-UPS [0/100]</p>
        <p>- SQUATS [0/100]</p>
    </div>
    """, unsafe_allow_html=True)
