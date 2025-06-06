A Python application that uses your webcam and MediaPipe PoseLandmarker to monitor your posture in real-time. The goal is to detect slouching and remind you to sit up straight!

## Description

This project leverages computer vision to provide real-time feedback on user posture. By analyzing key body landmarks, it aims to identify when a user is slouching and help promote better sitting habits.

## Features (Current & Planned)

* **Real-time Pose Estimation:** Displays your pose skeleton on the webcam feed.
* **Slouch Detection Logic:** Determines the distance between your mouth and shoulders, if this is below a threshold you are likely slouching
* **User Feedback:** (Visual overlay is current; alerts are planned)

## Tech Stack

* Python 3.11
* OpenCV (`opencv-python`) for camera access and image manipulation.
* MediaPipe (`mediapipe`) for pose landmark detection.

## Setup Instructions

1.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3.11 -m venv myvenv_py3.11 
    ```
    
    ```bash
    source myvenv_py3.11/bin/activate
    ```
    

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Ensure your virtual environment is activated.
2.  Verify that the `model_path` in your script (e.g., `main.py`) points to the downloaded `.task` model file.
3.  Run the script from your terminal:
    ```bash
    python main.py 
    ```
4.  A window should appear showing your webcam feed with pose landmarks (if detected).
5.  Press 'q' to close the application window and stop the script.
6.  Update the threshold as needed based on your testing

7. Notes: Threshold may need to be adjusted on a user by user basis
8. Dependencies work on mac, not sure how the plyer will handle other devices

## To-Do / Future Enhancements

This project is a work in progress. Planned improvements include:
* Implementing robust slouch detection logic based on landmark angles and positions.
* Adding clear visual or audio notifications when slouching is detected.
* Allowing users to calibrate their "good" posture.
* Adding a simple UI for settings.
* Tracking posture statistics over time.

---

Feel free to modify and expand this README as your project grows! Good luck!