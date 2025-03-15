import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

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

def track_pushup(goal, pushup_placeholder):
    cap = cv2.VideoCapture(0)
    count = 0
    position = None
    position_confidence = 0
    
    # Lower visibility threshold
    visibility_threshold = 0.5
    
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Default status
        status_text = "Align your body properly"
        status_color = (255, 255, 0)  # Yellow for ready
        
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
            
            # Check if all key points are visible
            all_visible = True
            visibility_scores = {}
            for landmark in key_landmarks:
                visibility = landmarks[landmark.value].visibility
                visibility_scores[landmark.name] = visibility
                if visibility < visibility_threshold:
                    all_visible = False
            
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
                
                # Push-up logic
                if angle_diff < 20:  # Ensure symmetrical form
                    if avg_angle > 160:  # Up position (arms mostly extended)
                        if position == "down":
                            position_confidence += 1
                            if position_confidence > 5:  # Require more stability
                                count += 1
                                position = "up"
                                position_confidence = 0
                                status_text = "Push-up counted!"
                                status_color = (0, 255, 0)  # Green for counted
                            else:
                                status_text = "Hold position..."
                                status_color = (0, 255, 255)  # Cyan for transition
                        else:
                            position = "up"
                            status_text = "Up position - Go down"
                            status_color = (0, 255, 0)  # Green for up
                            
                    elif avg_angle < 90:  # Down position (arms bent)
                        position = "down"
                        position_confidence = 0
                        status_text = "Down position - Push up"
                        status_color = (0, 0, 255)  # Red for down
                    else:
                        # Intermediate position
                        status_text = "Intermediate - Go lower"
                        status_color = (255, 165, 0)  # Orange for intermediate
                else:
                    status_text = "Align body symmetrically"
                    status_color = (255, 165, 0)  # Orange
                    position_confidence = max(0, position_confidence - 1)  # Decrease confidence
            else:
                status_text = "Position not clear"
                status_color = (128, 128, 128)  # Gray for unclear
                position_confidence = 0
            
            # Draw landmarks and count
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        cv2.putText(image, f"Push-ups: {count}/{goal}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        if 'avg_angle' in locals():
            cv2.putText(image, f"Angle: {avg_angle:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"Confidence: {position_confidence}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display visibility scores for debugging
        visibility_text = "Visibility: " + ", ".join([f"{k}: {v:.2f}" for k, v in visibility_scores.items()])
        cv2.putText(image, visibility_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        stframe.image(image, channels="BGR", use_container_width=True)
        
        # Update the UI with the current push-up count
        st.session_state.pushup_count = count
        pushup_placeholder.markdown(f"""
        <div class='quest-box'>
            <h2>DAILY QUEST - TRAIN TO BECOME A FORMIDABLE COMBATANT</h2>
            <p class='goal-text'>GOALS</p>
            <p>- PUSH-UPS [{st.session_state.pushup_count}/{goal}]</p>
            <p>- SQUATS [{st.session_state.squat_count}/100]</p>
        </div>
        """, unsafe_allow_html=True)
        
        if count >= goal:
            st.success("Push-up goal reached!")
            cap.release()
            return True
    
    cap.release()
    return False

def track_squat(goal, squat_placeholder):
    cap = cv2.VideoCapture(0)
    count = 0
    position = None
    position_confidence = 0
    
    # Lower visibility threshold
    visibility_threshold = 0.5
    
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Default status
        status_text = "Align your body properly"
        status_color = (255, 255, 0)  # Yellow for ready
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Check visibility of key points
            key_landmarks = [
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
            
            # Check if all key points are visible
            all_visible = True
            visibility_scores = {}
            for landmark in key_landmarks:
                visibility = landmarks[landmark.value].visibility
                visibility_scores[landmark.name] = visibility
                if visibility < visibility_threshold:
                    all_visible = False
            
            if all_visible:
                # Get key points for squat tracking
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
                
                # Calculate angles for knee bend
                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                avg_knee_angle = (left_angle + right_angle) / 2
                
                # Check if both angles are similar (indicating proper form)
                angle_diff = abs(left_angle - right_angle)
                
                # Squat logic
                if angle_diff < 20:  # Ensure symmetrical form
                    if avg_knee_angle > 160:  # Standing position
                        if position == "down":
                            position_confidence += 1
                            if position_confidence > 5:  # Require more stability
                                count += 1
                                position = "up"
                                position_confidence = 0
                                status_text = "Squat counted!"
                                status_color = (0, 255, 0)  # Green for counted
                            else:
                                status_text = "Hold position..."
                                status_color = (0, 255, 255)  # Cyan for transition
                        else:
                            position = "up"
                            status_text = "Standing - Go down"
                            status_color = (0, 255, 0)  # Green for up
                            
                    elif avg_knee_angle < 100:  # Squat position (adjust this threshold as needed)
                        position = "down"
                        position_confidence = 0
                        status_text = "Squat position - Stand up"
                        status_color = (0, 0, 255)  # Red for down
                    else:
                        # Intermediate position
                        status_text = "Go lower"
                        status_color = (255, 165, 0)  # Orange for intermediate
                else:
                    status_text = "Align body symmetrically"
                    status_color = (255, 165, 0)  # Orange
                    position_confidence = max(0, position_confidence - 1)  # Decrease confidence
            else:
                status_text = "Position not clear"
                status_color = (128, 128, 128)  # Gray for unclear
                position_confidence = 0
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        cv2.putText(image, f"Squats: {count}/{goal}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        if 'avg_knee_angle' in locals():
            cv2.putText(image, f"Knee angle: {avg_knee_angle:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"Confidence: {position_confidence}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display visibility scores for debugging
        visibility_text = "Visibility: " + ", ".join([f"{k}: {v:.2f}" for k, v in visibility_scores.items()])
        cv2.putText(image, visibility_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        stframe.image(image, channels="BGR", use_container_width=True)
        
        # Update the UI with the current squat count
        st.session_state.squat_count = count
        squat_placeholder.markdown(f"""
        <div class='quest-box'>
            <h2>DAILY QUEST - TRAIN TO BECOME A FORMIDABLE COMBATANT</h2>
            <p class='goal-text'>GOALS</p>
            <p>- PUSH-UPS [{st.session_state.pushup_count}/100]</p>
            <p>- SQUATS [{st.session_state.squat_count}/{goal}]</p>
        </div>
        """, unsafe_allow_html=True)
        
        if count >= goal:
            st.success("Squat goal reached!")
            cap.release()
            return True
    
    cap.release()
    return False

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

if st.button("Start Tracking", key="start_button"):
    if task == "Push-up":
        success = track_pushup(goal, quest_placeholder)
    elif task == "Squat":
        success = track_squat(goal, quest_placeholder)

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
