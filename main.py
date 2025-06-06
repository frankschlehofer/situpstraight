import cv2 as cv
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from plyer import notification

# Global threshold
threshold = 0.2
last_notification_time = 0
notification_cooldown_seconds = 5 # Only notify once every 5 seconds


# Capture video
cap = cv.VideoCapture(1)

model_path = 'pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult # For type hinting
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result: PoseLandmarkerResult = None # Type hint global variable

# Result callback
def result_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Drawing landmark function
def draw_landmarks_on_image(numpy_rgb_image, detection_result: PoseLandmarkerResult):
    annotated_image = np.copy(numpy_rgb_image)

    if not detection_result.pose_landmarks: # Check if list is empty or None
        return annotated_image

    for pose_landmarks_instance in detection_result.pose_landmarks: # Each instance is List[NormalizedLandmark]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z,
                visibility=landmark.visibility if hasattr(landmark, 'visibility') else 0.0, # Safer check
                presence=landmark.presence if hasattr(landmark, 'presence') else 0.0 # Safer check
            ) for landmark in pose_landmarks_instance
        ])
        
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

def detect_slouching(detection_result: PoseLandmarkerResult):
    if not detection_result.pose_landmarks:
        return False, 0

    # For current method of detection, we want to calculate the y distance between the mouth and shoulders
    # 9, 10 are mouth, 11 and 12 are shoulders

    landmarks = detection_result.pose_landmarks[0]
    y_mouth_midpoint = (landmarks[9].y + landmarks[10].y) / 2
    y_shoulder_midpoint = (landmarks[11].y + landmarks[12].y) / 2

    # If the difference is < min_distance, likely slouching. Min distance may need to vary person by person
    
    distance = abs(y_mouth_midpoint - y_shoulder_midpoint)
    if distance < threshold:
        return True, distance
    return False, distance

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback # Use the renamed callback
)

with PoseLandmarker.create_from_options(options) as landmarker:
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while cap.isOpened():
        success, frame_bgr = cap.read() # Read BGR frame

        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_bgr = cv.flip(frame_bgr, 1) # Flip the BGR frame

        # Convert the BGR frame to RGB for MediaPipe
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        current_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, current_timestamp_ms)

        # Prepare frame for display
        display_image_bgr = frame_bgr # Default to original flipped BGR frame

        # Make a local copy of latest_result to use in this iteration
        current_latest_result = latest_result 

        if current_latest_result and current_latest_result.pose_landmarks:
            # Pass the RGB frame to the drawing function, as it might be what drawing utils expect
            # or it's easier to manage colors. The function returns an annotated RGB frame.
            annotated_rgb_frame = draw_landmarks_on_image(frame_rgb, current_latest_result)
            # Convert the annotated RGB frame back to BGR for OpenCV display
            display_image_bgr = cv.cvtColor(annotated_rgb_frame, cv.COLOR_RGB2BGR)


            text_to_display = "Live Feed - Press 'q' to quit"
            position = (10, 30) # (x, y) - pixels from top-left
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_color_bgr = (0, 255, 0) # Green
            text_thickness = 2
            
            cv.putText(display_image_bgr, 
                        text_to_display, 
                        position, 
                        font, 
                        font_scale, 
                        text_color_bgr, 
                        text_thickness, 
                        cv.LINE_AA)
            # Test for slouching, ret set to true if slouching
            ret, distance = detect_slouching(current_latest_result)
            if ret:
                # Display on OpenCV screen
                slouch_text = f"POSTURE ALERT! Distance: {distance:.2f}"
                slouch_text_position = (10, 70)
                slouch_text_color = (0, 0, 255) # Red
                cv.putText(display_image_bgr, slouch_text, slouch_text_position, font, 1.0, slouch_text_color, 2, cv.LINE_AA)

                # Send notification if one hasnt been sent in past 30 seconds
                current_time = time.time()
                if (current_time - last_notification_time) > notification_cooldown_seconds:
                    print("SENDING NOTIFICATION!")
                    notification.notify(
                        title='Posture Check!',
                        message='You seem to be slouching. Take a moment to sit up straight.',
                        app_name='Sit Up Straight',
                        timeout=10 # Notification will disappear after 10 seconds on some systems
                    )
                    last_notification_time = current_time # Reset the timer
               
        
        cv.imshow('window', display_image_bgr)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()