import mediapipe as mp
import numpy as np
import cv2
import pickle
from db_event_data import send_alert_data
from send_data_into_rabbitmq import log_info, log_exception, log_error

mp_face_mesh = mp.solutions.face_mesh  # Initialize MediaPipe face mesh solution for detecting and tracking facial landmarks.
# mp_hands = mp.solutions.hands          # Initialize MediaPipe hands solution for detecting and tracking hand landmarks.
# mp_drawing = mp.solutions.drawing_utils  # Initialize MediaPipe drawing utilities for rendering landmarks (face mesh, hands, pose).


''' Create a face mesh detector instance:
 static_image_mode=False: treats input as a video stream.
 max_num_faces=1: detects only one face.
 refine_landmarks=True: enables attention mesh for more accurate iris/contour detection. 
 '''
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

''' Create a hand detector instance:
 static_image_mode=False: treats input as a video stream.
 max_num_hands=1: detects only one hand.
 min_detection_confidence=0.5: confidence threshold to filter low-quality detections. 
 '''
#hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


LEFT_EYE = [33, 160, 158, 133, 153, 144]   # Indices for key points around the left eye (used for blink detection, gaze estimation, etc.)
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # Indices for key points around the right eye (same usage as above
NOSE_TIP = 1                               # Index of the tip of the nose (often used as a central reference point for face orientation)
#CHIN = 152                                 # Index of the bottom of the chin (can help with head tilt or pose estimation)
#LEFT_EAR_POINT = 234                       # Index of a landmark near the left ear (used in head pose estimation and ear detection)
#RIGHT_EAR_POINT = 454                      # Index of a landmark near the right ear (used similarly as the left)
UPPER_LIP = 13                             # Index for the upper lip (for mouth state detection, e.g., open/closed, talking, yawning)
LOWER_LIP = 14                             # Index for the lower lip (used with upper lip for mouth-related analysis)

EAR_THRESHOLD = 0.25


class DrowsinessYawnAlertHandler:
    def __init__(self, camera_id, user_id, exchange_name, rabbitmq_host, DROWSY_TIME_THRESHOLD, YAWN_TIME_THRESHOLD):
        self.camera_id = camera_id
        self.user_id = user_id
        self.exchange_name = exchange_name
        self.rabbitmq_host = rabbitmq_host
        self.DROWSY_TIME_THRESHOLD = DROWSY_TIME_THRESHOLD
        self.YAWN_TIME_THRESHOLD = YAWN_TIME_THRESHOLD

        # Drowsiness state
        self.drowsy_start_time = None
        self.drowsy_alert_trigger_time = None
        self.last_alert_duration_drowsy = 0
        self.drowsiness_alert_intervals = [0]
        self.next_drowsiness_alert_index = 0

        # Yawning state
        self.yawn_start_time = None
        self.yawn_alert_trigger_time = None
        self.last_alert_duration_yawn = 0
        self.yawning_alert_intervals = [0]
        self.next_yawning_alert_index = 0

    def calculate_eye(self, landmarks, eye_indices):
        eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear

    @staticmethod
    def is_yawning(landmarks, threshold=0.03):
        upper_lip_y = landmarks[UPPER_LIP].y
        lower_lip_y = landmarks[LOWER_LIP].y
        mouth_open_distance = lower_lip_y - upper_lip_y
        return mouth_open_distance > threshold

    def build_alert_data(self, current_time, date_time, duration_time, start_time, alert_type, threshold_time):
        return {
            "camera_id": self.camera_id,
            "user_id": self.user_id,
            "date_time": date_time,
            "duration_time": duration_time + threshold_time,
            "severity": "Critical",
            "start_time": start_time,
            "end_time": current_time,
            "evidence": "https://example.com/evidence/8896",
            "pilot": "Co-Pilot",
            "alert_type": alert_type
        }

    def check_and_send_alert(self, current_time, date_time, start_time, alert_type, threshold_time):
        if alert_type == "Drowsiness":
            index = self.next_drowsiness_alert_index
            intervals = self.drowsiness_alert_intervals
            duration = self.last_alert_duration_drowsy
        else:
            index = self.next_yawning_alert_index
            intervals = self.yawning_alert_intervals
            duration = self.last_alert_duration_yawn

        if index < len(intervals) and duration >= intervals[index]:
            alert_data = self.build_alert_data(current_time, date_time, int(duration), start_time, alert_type, threshold_time)
            send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
            log_info(f"{alert_type} alert sent. Duration: {int(duration)}s", date_time, self.rabbitmq_host)

            # Update interval strategy
            if alert_type == "Drowsiness":
                self.next_drowsiness_alert_index += 1
                self.drowsiness_alert_intervals.append(intervals[-1] + (self.next_drowsiness_alert_index * 10))
            else:
                self.next_yawning_alert_index += 1
                self.yawning_alert_intervals.append(intervals[-1] + (self.next_yawning_alert_index * 10))

    def process_drowsiness(self, avg_eye, current_time, date_time):
        if avg_eye < EAR_THRESHOLD:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = current_time

            elapsed_time = (current_time - self.drowsy_start_time).total_seconds()
            if elapsed_time >= self.DROWSY_TIME_THRESHOLD:
                if self.drowsy_alert_trigger_time is None:
                    self.drowsy_alert_trigger_time = current_time
                self.last_alert_duration_drowsy = (current_time - self.drowsy_alert_trigger_time).total_seconds()
                self.check_and_send_alert(current_time, date_time, self.drowsy_start_time, "Drowsiness", self.DROWSY_TIME_THRESHOLD)
        else:
            if self.drowsy_alert_trigger_time is not None:
                alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_drowsy), self.drowsy_start_time, "Drowsiness", self.DROWSY_TIME_THRESHOLD)
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"Drowsiness alert ended. Duration: {int(self.last_alert_duration_drowsy)}s", date_time, self.rabbitmq_host)

            # Reset state
            self.drowsy_start_time = None
            self.drowsy_alert_trigger_time = None
            self.last_alert_duration_drowsy = 0
            self.drowsiness_alert_intervals = [0]
            self.next_drowsiness_alert_index = 0

    def process_yawning(self, landmarks, current_time, date_time):
        if self.is_yawning(landmarks):
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time

            elapsed_time = (current_time - self.yawn_start_time).total_seconds()
            if elapsed_time >= self.YAWN_TIME_THRESHOLD:
                if self.yawn_alert_trigger_time is None:
                    self.yawn_alert_trigger_time = current_time
                self.last_alert_duration_yawn = (current_time - self.yawn_alert_trigger_time).total_seconds()
                self.check_and_send_alert(current_time, date_time, self.yawn_start_time, "Yawning", self.YAWN_TIME_THRESHOLD)
        else:
            if self.yawn_alert_trigger_time is not None:
                alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_yawn), self.yawn_start_time, "Yawning", self.YAWN_TIME_THRESHOLD)
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"Yawning alert ended. Duration: {int(self.last_alert_duration_yawn)}s", date_time, self.rabbitmq_host)

            # Reset state
            self.yawn_start_time = None
            self.yawn_alert_trigger_time = None
            self.last_alert_duration_yawn = 0
            self.yawning_alert_intervals = [0]
            self.next_yawning_alert_index = 0

    def frame_process_for_drowsiness(self, frame_rgb, current_time, date_time):
        face_results = face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            left_eye = self.calculate_eye(face_landmarks, LEFT_EYE)
            right_eye = self.calculate_eye(face_landmarks, RIGHT_EYE)
            avg_eye = (left_eye + right_eye) / 2.0

            self.process_drowsiness(avg_eye, current_time, date_time)
            self.process_yawning(face_landmarks, current_time, date_time)

































# class DrowsinessYawnAlertHandler:
#     def __init__(self, camera_id, user_id, exchange_name, rabbitmq_host, DROWSY_TIME_THRESHOLD):
#         self.camera_id = camera_id
#         self.user_id = user_id
#         self.exchange_name = exchange_name
#         self.rabbitmq_host = rabbitmq_host
#         self.DROWSY_TIME_THRESHOLD = DROWSY_TIME_THRESHOLD

#         # For managing alert logic
#         self.drowsy_start_time = None
#         self.drowsy_alert_trigger_time = None
#         self.last_alert_duration_drowsy = 0

#         # Interval strategy
#         self.drowsiness_alert_intervals = [0]  # seconds
#         self.next_drowsiness_alert_index = 0

#     def calculate_eye(self, landmarks, eye_indices):
#         eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
#         vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
#         vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
#         horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
#         ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
#         return ear
    
#     def is_yawning(landmarks, threshold=0.03):
#         upper_lip_y = landmarks[UPPER_LIP].y
#         lower_lip_y = landmarks[LOWER_LIP].y
#         mouth_open_distance = lower_lip_y - upper_lip_y
#         return mouth_open_distance > threshold

#     def build_alert_data(self, current_time, date_time, duration_time, start_time, alert_type):
#         return {
#             "camera_id": self.camera_id,
#             "user_id": self.user_id,
#             "date_time": date_time,
#             "duration_time": duration_time + 120,
#             "severity": "Critical",
#             "start_time": start_time,
#             "end_time": current_time,
#             "evidence": "https://example.com/evidence/8896",
#             "pilot": "Co-Pilot",
#             "alert_type": alert_type
#         }

#     def check_and_send_drowsiness_alert(self, current_time, date_time, start_time, alert_type):
#         index = self.next_drowsiness_alert_index
#         intervals = self.drowsiness_alert_intervals

#         if index < len(intervals) and self.last_alert_duration_drowsy >= intervals[index]:
#             alert_data = self.build_alert_data(
#                 current_time, date_time, int(self.last_alert_duration_drowsy), start_time, alert_type
#                 )
#             send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
#             log_info(f"{alert_type} alert sent. Duration: {int(self.last_alert_duration_drowsy)}s", date_time, self.rabbitmq_host)

#             # Update interval strategy
#             self.next_drowsiness_alert_index += 1
#             next_interval = intervals[-1] + (self.next_drowsiness_alert_index * 10)
#             self.drowsiness_alert_intervals.append(next_interval)

#     def process_drowsiness(self, avg_eye, current_time, date_time):
#         if avg_eye < EAR_THRESHOLD:
#             if self.drowsy_start_time is None:
#                 self.drowsy_start_time = current_time

#             elapsed_time = (current_time - self.drowsy_start_time).total_seconds()

#             if elapsed_time >= self.DROWSY_TIME_THRESHOLD:
#                 if self.drowsy_alert_trigger_time is None:
#                     self.drowsy_alert_trigger_time = current_time
#                 self.last_alert_duration_drowsy = (current_time - self.drowsy_alert_trigger_time).total_seconds()
#                 self.check_and_send_drowsiness_alert(current_time, date_time, self.drowsy_start_time,"Drowsiness")
#                 print("DROWSINESS ALERT!")

#         else:
#             if self.drowsy_alert_trigger_time is not None:
#                 print(f"Last Drowsiness Alert Duration: {int(self.last_alert_duration_drowsy)} sec")
#                 alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_drowsy), self.drowsy_start_time, "Drowsiness")
#                 send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
#                 log_info(f"Drowsiness alert and time duration :{int(self.last_alert_duration_drowsy)}", date_time, self.rabbitmq_host)


#             # Reset drowsiness state
#             self.drowsy_start_time = None
#             self.drowsy_alert_trigger_time = None
#             self.last_alert_duration_drowsy = 0
#             self.drowsiness_alert_intervals = [0]
#             self.next_drowsiness_alert_index = 0

#     def frame_process_for_drowsiness(self, frame_rgb, frame_bgr, current_time, date_time):
#         face_results = face_mesh.process(frame_rgb)
#         if face_results.multi_face_landmarks:
#             face_landmarks = face_results.multi_face_landmarks[0].landmark
#             left_eye = self.calculate_eye(face_landmarks, LEFT_EYE)
#             right_eye = self.calculate_eye(face_landmarks, RIGHT_EYE)
#             avg_eye = (left_eye + right_eye) / 2.0
#             self.process_drowsiness(avg_eye, current_time, date_time)



