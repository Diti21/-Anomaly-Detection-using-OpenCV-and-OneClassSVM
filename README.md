# -Anomaly-Detection-using-OpenCV-and-OneClassSVM
This project implements a motion detection and anomaly detection system using OpenCV and a machine learning model (OneClassSVM). 
The program captures video from a webcam or a video file, detects motion using background subtraction, extracts motion features, and trains an anomaly detection model to highlight irregular activities.

Features
Real-time Motion Detection: Uses background subtraction with cv2.createBackgroundSubtractorMOG2 to detect motion in the video stream.
Feature Extraction: Extracts contour features such as aspect ratio, area, and perimeter for motion analysis.
Anomaly Detection: Implements a machine learning-based anomaly detection system using sklearn's OneClassSVM.
Real-Time Feedback: Displays detected motion with bounding boxes, distinguishing anomalies with red bounding boxes and regular motion with green.
Multi-Environment Support: Handles headless environments by saving frames or displaying results using matplotlib.
Getting Started

Prerequisites:
Ensure you have the following installed:
Python (3.7 or higher)
Required Python libraries:
OpenCV (cv2)
NumPy
Matplotlib
scikit-learn

Install the dependencies using:
Copy code:
pip install opencv-python numpy matplotlib scikit-learn

Setup
Clone the repository or download the script.
Ensure your webcam is connected, or replace cap = cv2.VideoCapture(0) with the path to a video file.


Run the script:
Copy code:
python motion_anomaly_detection.py

The program will:
Capture video frames from the webcam or video file.
Detect motion and extract features.
Train an anomaly detection model dynamically after collecting sufficient motion data.
Highlight detected motions and anomalies in real-time.

Modify settings as needed:
Change video source: Replace 0 in cv2.VideoCapture(0) with a file path.
Adjust training size: Modify the 200 frame threshold in the script.
Control frame delay: Change the time.sleep(0.03) for desired FPS.

Output
Real-time Visualization: The script displays two windows:
Motion Detection: Shows bounding boxes for detected motions and anomalies.
Foreground Mask: Displays the processed mask for motion detection.
Saved Images: In headless environments, frames are saved locally as .jpg files with names like motion_detection_frame_<frame_count>.jpg.


Code Structure
Motion Detection: Uses background subtraction and contour extraction.
Feature Extraction: Extracts shape-related features (aspect ratio, area, perimeter) from motion contours.
Model Training: Trains a OneClassSVM model to detect anomalies based on extracted features.
Anomaly Detection: Predicts anomalies in real-time using the trained model.

Known Limitations
Training Data Dependency: The anomaly detection model requires a sufficient amount of "normal" motion data for effective training.
Environmental Sensitivity: Performance may vary depending on lighting and camera positioning.
Processing Speed: High-resolution videos may reduce performance on low-end systems.






