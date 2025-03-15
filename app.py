import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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

class PushupTracker(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.position = None
        self.position_confidence = 0
        self.angle_history = []
        self.visibility_threshold = 0.7

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Check visibility of key points
            key_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            all_visible = all(landmarks[lm.value].visibility > self.visibility_threshold for lm in key_landmarks)

            if all_visible:
                # Get key points for push-up tracking
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

                # Calculate angles
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                avg_angle = (left_angle + right_angle) / 2

                # Check if both angles are similar (indicating proper form)
                angle_diff = abs(left_angle - right_angle)

                # Add angle to history for stability check
                self.angle_history.append(avg_angle)
                if len(self.angle_history) > 5:  # Keep only last 5 frames
                    self.angle_history.pop(0)

                # Calculate angle stability
                angle_stability = 0
                if len(self.angle_history) >= 3:
                    angle_variations = [abs(self.angle_history[i] - self.angle_history[i-1]) for i in range(1, len(self.angle_history))]
                    avg_variation = sum(angle_variations) / len(angle_variations)
                    angle_stability = 10 if avg_variation < 5 else 0  # Stable if variation is small

                # Push-up logic with more robustness
                if angle_diff < 20 and angle_stability > 0:  # Ensure symmetrical form
                    if avg_angle > 150:  # Up position (arms mostly extended)
                        if self.position == "down":
                            self.position_confidence += 1
                            if self.position_confidence > 5:  # Require more stability
                                self.count += 1
                                self.position = "up"
                                self.position_confidence = 0
                                status_text = "Push-up counted!"
                                status_color = (0, 255, 0)  # Green for counted
                            else:
                                status_text = "Hold position..."
                                status_color = (0, 255, 255)  # Cyan for transition
                        else:
                            self.position = "up"
                            status_text = "Up position - Go down"
                            status_color = (0, 255, 0)  # Green for up

                    elif avg_angle < 90:  # Down position (arms bent)
                        self.position = "down"
                        self.position_confidence = 0
                        status_text = "Down position - Push up"
                        status_color = (0, 0, 255)  # Red for down
                    else:
                        # Intermediate position
                        status_text = "Intermediate - Go lower"
                        status_color = (255, 165, 0)  # Orange for intermediate
                else:
                    status_text = "Align body symmetrically"
                    status_color = (255, 165, 0)  # Orange
                    self.position_confidence = max(0, self.position_confidence - 1)  # Decrease confidence
            else:
                status_text = "Position not clear"
                status_color = (128, 128, 128)  # Gray for unclear
                self.position_confidence = 0

            # Draw landmarks and count
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(image, f"Push-ups: {self.count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            if 'avg_angle' in locals():
                cv2.putText(image, f"Angle: {avg_angle:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(image, f"Confidence: {self.position_confidence}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

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
    <p>- SIT-UPS [0/100]</p>
    <p>- SQUATS [{st.session_state.squat_count}/100]</p>
</div>
""", unsafe_allow_html=True)

task = st.selectbox("Choose Exercise", ["Push-up", "Squat"])
goal = st.number_input("Set Goal", min_value=1, value=10)

if st.button("Start Tracking", key="start_button"):
    if task == "Push-up":
        webrtc_streamer(
            key="pushup",
            video_processor_factory=PushupTracker,
            rtc_configuration=RTC_CONFIGURATION,
        )

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
