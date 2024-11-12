# Import required libraries for computer vision, UI, threading, and logging
import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from dataclasses import dataclass
import threading
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Data structure to store stabilized landmarks for smoother hand tracking
@dataclass
class StabilizedLandmark:
    x: float
    y: float
    z: float


# Hand tracking thread for processing frames and detecting gestures
class HandTrackingThread(QThread):
    # Signals for inter-thread communication
    fingers_close = pyqtSignal(tuple)  # Emits thumb-index proximity as screen coordinates
    frame_processed = pyqtSignal(np.ndarray)  # Emits processed frame for GUI overlay

    def __init__(self, canvas_width, canvas_height):
        super().__init__()
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.running = True
        self.closeness_triggered = {}  # Tracks which hands triggered the "close" gesture
        self.hand_scale = 0.5  # Scaling factor for hand size
        self.previous_landmarks = {}  # For storing previous landmarks to stabilize movement
        self.stabilization_factor = 0.7  # Smoothing factor for landmarks
        self.lock = threading.Lock()  # Ensures thread-safe operations

        # Initialize Mediapipe's Hand module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Initialize video capture (camera)
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():  # Try another camera index if the first fails
            logging.warning("Camera index 1 failed. Attempting index 0.")
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():  # If no camera works, raise an error
            raise RuntimeError("Failed to open camera.")

    # The main thread loop that continuously processes camera frames
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to read frame from the camera.")
                continue

            frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)  # Detect hands in the frame

            # Create a blank canvas to draw on
            canvas = np.zeros((self.canvas_height, self.canvas_width, 4), dtype=np.uint8)

            # Process each detected hand's landmarks
            if results.multi_hand_landmarks:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    try:
                        self.process_hand_landmarks(hand_landmarks.landmark, hand_id, frame.shape, canvas)
                    except Exception as e:
                        logging.error(f"Error processing hand {hand_id}: {e}")

            # Emit the processed frame for the GUI to update
            self.frame_processed.emit(canvas)

    # Processes landmarks for a specific hand, checks for gestures, and draws on the canvas
    def process_hand_landmarks(self, current_landmarks, hand_id, frame_shape, canvas):
        frame_height, frame_width, _ = frame_shape

        # Define a central rectangle for hand scaling
        rect_width = int(frame_width * 0.7)
        rect_height = int(frame_height * 0.7)
        rect_x1, rect_y1 = (frame_width - rect_width) // 2, (frame_height - rect_height) // 2

        # Stabilize landmarks to reduce jitter
        smoothed_landmarks = self.stabilize_landmarks(current_landmarks, hand_id)

        # Calculate hand center coordinates
        hand_xs = [lm.x for lm in smoothed_landmarks]
        hand_ys = [lm.y for lm in smoothed_landmarks]
        hand_center_x = sum(hand_xs) / len(hand_xs)
        hand_center_y = sum(hand_ys) / len(hand_ys)

        # Extract and scale thumb and index fingertips
        thumb_tip = smoothed_landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = smoothed_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        thumb, index = self.scale_landmarks(
            [thumb_tip, index_tip],
            hand_center_x,
            hand_center_y,
            frame_width,
            frame_height,
            rect_x1,
            rect_y1,
            rect_width,
            rect_height,
        )

        # Check if thumb and index finger are close
        self.check_finger_proximity(thumb, index, hand_id)

        # Draw the hand on the canvas
        self.draw_hand(
            smoothed_landmarks,
            canvas,
            hand_center_x,
            hand_center_y,
            frame_width,
            frame_height,
            rect_x1,
            rect_y1,
            rect_width,
            rect_height,
        )

    # Scales landmarks based on hand position and canvas dimensions
    def scale_landmarks(self, landmarks, hand_center_x, hand_center_y, frame_width, frame_height, rect_x1, rect_y1, rect_width, rect_height):
        scaled_landmarks = []
        for lm in landmarks:
            scaled_x = (lm.x - hand_center_x) * self.hand_scale + hand_center_x
            scaled_y = (lm.y - hand_center_y) * self.hand_scale + hand_center_y

            x = int((scaled_x * frame_width - rect_x1) / rect_width * self.canvas_width)
            y = int((scaled_y * frame_height - rect_y1) / rect_height * self.canvas_height)

            scaled_landmarks.append((max(0, min(self.canvas_width, x)), max(0, min(self.canvas_height, y))))
        return scaled_landmarks

    # Checks if thumb and index fingertips are close and triggers a gesture if so
    def check_finger_proximity(self, thumb, index, hand_id):
        thumb_x, thumb_y = thumb
        index_x, index_y = index
        distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

        with self.lock:
            if hand_id not in self.closeness_triggered:
                self.closeness_triggered[hand_id] = False

            if distance < 50 and not self.closeness_triggered[hand_id]:
                self.fingers_close.emit(((thumb_x + index_x) // 2, (thumb_y + index_y) // 2))
                self.closeness_triggered[hand_id] = True
            elif distance >= 50:
                self.closeness_triggered[hand_id] = False

    # Draws the hand landmarks and connections on the canvas
    def draw_hand(self, landmarks, canvas, hand_center_x, hand_center_y, frame_width, frame_height, rect_x1, rect_y1, rect_width, rect_height):
        points = []
        for lm in landmarks:
            x, y = self.scale_landmarks([lm], hand_center_x, hand_center_y, frame_width, frame_height, rect_x1, rect_y1, rect_width, rect_height)[0]
            points.append((x, y))
            cv2.circle(canvas, (x, y), 5, (255, 255, 255, 255), -1)

        # Draw lines connecting the landmarks
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = points[start_idx]
            end_point = points[end_idx]
            cv2.line(canvas, start_point, end_point, (0, 255, 0, 255), 2)

    # Stabilizes landmarks to reduce jitter
    def stabilize_landmarks(self, current_landmarks, hand_id):
        if hand_id not in self.previous_landmarks:
            self.previous_landmarks[hand_id] = [StabilizedLandmark(lm.x, lm.y, lm.z) for lm in current_landmarks]
            return self.previous_landmarks[hand_id]

        smoothed_landmarks = []
        for curr, prev in zip(current_landmarks, self.previous_landmarks[hand_id]):
            factor = self.stabilization_factor
            smoothed_landmarks.append(
                StabilizedLandmark(
                    curr.x * (1 - factor) + prev.x * factor,
                    curr.y * (1 - factor) + prev.y * factor,
                    curr.z * (1 - factor) + prev.z * factor,
                )
            )

        self.previous_landmarks[hand_id] = smoothed_landmarks
        return smoothed_landmarks

    # Stops the thread and releases resources
    def stop(self):
        self.running = False
        self.cap.release()
        self.hands.close()


# A PyQt task that triggers a simulated mouse click
class ClickTask(QRunnable):
    def __init__(self, coordinates):
        super().__init__()
        self.coordinates = coordinates

    def run(self):
        screen_width, screen_height = pyautogui.size()
        scaled_x = int(self.coordinates[0] * screen_width / 2560)
        scaled_y = int(self.coordinates[1] * screen_height / 1600)
        pyautogui.click(scaled_x, scaled_y)


# Transparent GUI overlay for hand tracking visualization
class TransparentOverlay(QMainWindow):
    def __init__(self, canvas_width, canvas_height):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, canvas_width, canvas_height)

        self.thread_pool = QThreadPool()  # Thread pool for tasks
        self.thread = HandTrackingThread(canvas_width, canvas_height)  # Hand tracking thread
        self.thread.frame_processed.connect(self.update_frame)
        self.thread.fingers_close.connect(self.handle_fingers_close)
        self.thread.start()

    # Updates the overlay with the processed frame
    def update_frame(self, canvas):
        image = QImage(canvas.data, canvas.shape[1], canvas.shape[0], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

    # Handles thumb-index "close" gesture by simulating a click
    def handle_fingers_close(self, coordinates):
        logging.info(f"Thumb and index finger tips are close. Screen coordinates: {coordinates}")
        self.thread_pool.start(ClickTask(coordinates))

    # Cleans up resources when the window is closed
    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        event.accept()


# Entry point for the application
if __name__ == "__main__":
    canvas_width, canvas_height = 2560, 1600
    app = QApplication([])  # Create the Qt application
    window = TransparentOverlay(canvas_width, canvas_height)  # Initialize the overlay
    window.showFullScreen()  # Show it in fullscreen mode
    app.exec_()  # Run the application
