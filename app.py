import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

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

class PoseTracker(VideoTransformerBase):
    def __init__(self, task, goal, placeholder):
        self.task = task
        self.goal = goal
        self.placeholder = placeholder
        self.count = 0
        self.position = None
        self.position_confidence = 0
        self.angle_history = []
        self.visibility_threshold = 0.7

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            if self.task == "Push-up":
                # Push-up logic (same as before)
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                avg_angle = (left_angle + right_angle) / 2

                # Push-up logic (same as before)
                if avg_angle > 150:
                    if self.position == "down":
                        self.position_confidence += 1
                        if self.position_confidence > 5:
                            self.count += 1
                            self.position = "up"
                            self.position_confidence = 0
                    else:
                        self.position = "up"
                elif avg_angle < 90:
                    self.position = "down"
                    self.position_confidence = 0

            elif self.task == "Squat":
                # Squat logic (same as before)
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                avg_angle = (left_angle + right_angle) / 2

                # Squat logic (same as before)
                if avg_angle > 160:
                    if self.position == "down":
                        self.position_confidence += 1
                        if self.position_confidence > 5:
                            self.count += 1
                            self.position = "up"
                            self.position_confidence = 0
                    else:
                        self.position = "up"
                elif avg_angle < 110:
                    self.position = "down"
                    self.position_confidence = 0

            # Draw landmarks and count
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, f"{self.task}s: {self.count}/{self.goal}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update the UI with the current count
        st.session_state.pushup_count = self.count if self.task == "Push-up" else st.session_state.pushup_count
        st.session_state.squat_count = self.count if self.task == "Squat" else st.session_state.squat_count
        self.placeholder.markdown(f"""
        <div class='quest-box'>
            <h2>DAILY QUEST - TRAIN TO BECOME A FORMIDABLE COMBATANT</h2>
            <p class='goal-text'>GOALS</p>
            <p>- PUSH-UPS [{st.session_state.pushup_count}/100]</p>
            <p>- SQUATS [{st.session_state.squat_count}/100]</p>
        </div>
        """, unsafe_allow_html=True)

        return image  # Return the processed frame

def track_exercise(task, goal, placeholder):
    webrtc_streamer(
        key=f"{task}-tracker",
        video_transformer_factory=lambda: PoseTracker(task, goal, placeholder),
        async_transform=True,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    )

# Initialize session state for push-up and squat counts
if "pushup_count" not in st.session_state:
    st.session_state.pushup_count = 0
if "squat_count" not in st.session_state:
    st.session_state.squat_count = 0

st.title("🏆 Quest Tracker - Become a Warrior! 🏋️")

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

if st.button("Start Tracking", key="start_button"):
    if task == "Push-up":
        track_exercise("Push-up", goal, quest_placeholder)
    elif task == "Squat":
        track_exercise("Squat", goal, quest_placeholder)

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
