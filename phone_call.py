import mediapipe as mp
import numpy as np
import cv2
from db_event_data import send_alert_data
from send_data_into_rabbitmq import log_info, log_exception, log_error

# mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

class PhoneCallDetector:
    def __init__(self, camera_id, user_id, exchange_name, rabbitmq_host, PHONE_TIME_THRESHOLD):
        self.camera_id = camera_id
        self.user_id = user_id
        self.exchange_name = exchange_name
        self.rabbitmq_host = rabbitmq_host

        self.PHONE_TIME_THRESHOLD = PHONE_TIME_THRESHOLD
        self.phone_detected_time = None
        self.phone_alert_trigger_time = None
        self.last_alert_duration_phone_call = 0
        self.phone_call_start_time = None

        self.alert_intervals = {"Phone Call": [self.PHONE_TIME_THRESHOLD]}
        self.next_alert_index = {"Phone Call": 0}

    @staticmethod
    def calculate_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def build_alert_data(self, current_time, date_time, duration_time, start_time, alert_type):
        return {
            "camera_id": self.camera_id,
            "user_id": self.user_id,
            "date_time": date_time,
            "duration_time": duration_time,
            "severity": "Critical",
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "evidence": "https://example.com/evidence/8896",
            "pilot": "Co-Pilot",
            "alert_type": alert_type
        }

    def check_and_send_head_pose_alert(self, alert_type, current_time, date_time, start_time, last_alert_duration):
        index = self.next_alert_index[alert_type]
        intervals = self.alert_intervals[alert_type]

        if index < len(intervals) and last_alert_duration >= intervals[index]:
            alert_data = self.build_alert_data(current_time, date_time, int(last_alert_duration), start_time, alert_type)
            send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
            log_info(f"Driver {alert_type} alert and time duration: {int(last_alert_duration)}", date_time, self.rabbitmq_host)

            self.next_alert_index[alert_type] += 1
            next_interval = intervals[-1] + (self.next_alert_index[alert_type] * 10)
            self.alert_intervals[alert_type].append(next_interval)

    def process_phone_call(self, frame, hand_results, pose_results, img_h, img_w, current_time, date_time):
        hand_in_box = False
        hand_open = False
        person_in_frame = False
        

        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                print("These are hand landmark ")

                finger_tips = {
                    "thumb": mp_hands.HandLandmark.THUMB_TIP,
                    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
                    "pinky": mp_hands.HandLandmark.PINKY_TIP
                }

                coords = {
                    key: (
                        int(landmarks.landmark[val].x * img_w),
                        int(landmarks.landmark[val].y * img_h)
                    )
                    for key, val in finger_tips.items()
                }

                thumb_y = landmarks.landmark[finger_tips["thumb"]].y
                avg_finger_y = sum(landmarks.landmark[finger_tips[f]].y for f in ["index", "middle", "ring", "pinky"]) / 4

                if thumb_y > avg_finger_y:
                    hand_open = True

                # for c in coords.values():
                #     cv2.circle(frame, c, 10, (255, 255, 0), -1)

        if pose_results.pose_landmarks:
            person_in_frame = True
            landmarks = pose_results.pose_landmarks.landmark
            print("These are pose landmark")

            def pos(name):
                pt = mp_pose.PoseLandmark[name]
                return int(landmarks[pt].x * img_w), int(landmarks[pt].y * img_h)

            lw, rw = pos("LEFT_WRIST"), pos("RIGHT_WRIST")
            le, re = pos("LEFT_EAR"), pos("RIGHT_EAR")
            leye, reye = pos("LEFT_EYE"), pos("RIGHT_EYE")
            ls, rs = pos("LEFT_SHOULDER"), pos("RIGHT_SHOULDER")

            shoulder_width = self.calculate_distance(ls, rs)
            dynamic_threshold = shoulder_width * 1.5

            box_left = min(leye[0], reye[0], ls[0], rs[0])
            box_right = max(leye[0], reye[0], ls[0], rs[0])
            box_top = min(leye[1], reye[1], ls[1], rs[1])
            box_bottom = max(ls[1], rs[1])

            #cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

            if self.calculate_distance(lw, le) < dynamic_threshold and box_left < lw[0] < box_right and box_top < lw[1] < box_bottom:
                hand_in_box = True
            if self.calculate_distance(rw, re) < dynamic_threshold and box_left < rw[0] < box_right and box_top < rw[1] < box_bottom:
                hand_in_box = True

        if person_in_frame and hand_open and hand_in_box:
            #cv2.putText(frame, "PERSON ON PHONE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if self.phone_detected_time is None:
                self.phone_detected_time = current_time
                self.phone_call_start_time = current_time

            elapsed_time = (current_time - self.phone_detected_time).total_seconds()

            if elapsed_time >= self.PHONE_TIME_THRESHOLD:
                if self.phone_alert_trigger_time is None:
                    self.phone_alert_trigger_time = current_time

                self.last_alert_duration_phone_call = (current_time - self.phone_alert_trigger_time).total_seconds()

                self.check_and_send_head_pose_alert("Phone Call", current_time, date_time, self.phone_call_start_time, self.last_alert_duration_phone_call)
                print("Phone call detected in if")
        else:
            if self.phone_alert_trigger_time:
                date_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                alert_data = self.build_alert_data(current_time, date_time_str, int(self.last_alert_duration_phone_call), self.phone_call_start_time, "Phone Call")
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time_str)
                log_info(f"Phone alert and time duration: {int(self.last_alert_duration_phone_call)}", date_time_str, self.rabbitmq_host)
                print("Phone call detected in else")
            self.phone_detected_time = None
            self.phone_alert_trigger_time = None
            self.phone_call_start_time = None
            self.last_alert_duration_phone_call = 0

    def frame_process_for_phone_call(self, frame_rgb, current_time, date_time):
        print("This is phone function call .....................")
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)
        img_h, img_w, _ = frame_rgb.shape
        self.process_phone_call(frame_rgb, hand_results, pose_results, img_h, img_w, current_time, date_time)


































# import mediapipe as mp
# import numpy as np
# import cv2
# import pickle
# from db_event_data import send_alert_data
# from send_data_into_rabbitmq import log_info, log_exception, log_error

# mp_face_mesh = mp.solutions.face_mesh  # Initialize MediaPipe face mesh solution for detecting and tracking facial landmarks.
# mp_hands = mp.solutions.hands          # Initialize MediaPipe hands solution for detecting and tracking hand landmarks.
# mp_drawing = mp.solutions.drawing_utils  # Initialize MediaPipe drawing utilities for rendering landmarks (face mesh, hands, pose).
# # Initialize MediaPipe Pose and Drawing utils for no driver
# mp_pose = mp.solutions.pose
# #mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose()

# ''' Create a face mesh detector instance:
#  static_image_mode=False: treats input as a video stream.
#  max_num_faces=1: detects only one face.
#  refine_landmarks=True: enables attention mesh for more accurate iris/contour detection. 
#  '''
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# ''' Create a hand detector instance:
#  static_image_mode=False: treats input as a video stream.
#  max_num_hands=1: detects only one hand.
#  min_detection_confidence=0.5: confidence threshold to filter low-quality detections. 
#  '''
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


# LEFT_EYE = [33, 160, 158, 133, 153, 144]   # Indices for key points around the left eye (used for blink detection, gaze estimation, etc.)
# RIGHT_EYE = [362, 385, 387, 263, 373, 380] # Indices for key points around the right eye (same usage as above
# NOSE_TIP = 1                               # Index of the tip of the nose (often used as a central reference point for face orientation)
# CHIN = 152                                 # Index of the bottom of the chin (can help with head tilt or pose estimation)
# LEFT_EAR_POINT = 234                       # Index of a landmark near the left ear (used in head pose estimation and ear detection)
# RIGHT_EAR_POINT = 454                      # Index of a landmark near the right ear (used similarly as the left)
# UPPER_LIP = 13                             # Index for the upper lip (for mouth state detection, e.g., open/closed, talking, yawning)
# LOWER_LIP = 14                             # Index for the lower lip (used with upper lip for mouth-related analysis)


# class PhoneCallDetector:
#     def __init__(self, camera_id, user_id, PHONE_TIME_THRESHOLD=5):
#         self.camera_id = camera_id
#         self.user_id = user_id
#         self.PHONE_TIME_THRESHOLD = PHONE_TIME_THRESHOLD
#         self.phone_detected_time = None
#         self.phone_alert_trigger_time = None
#         self.last_phone_alert_duration = 0


#     def calculate_distance(p1, p2):
#         """Calculate Euclidean distance between two points for Pose landmarks."""
#         return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  

#     def build_alert_data(self, current_time, date_time, duration_time, start_time, alert_type, threshold_time):
#         return {
#             "camera_id": self.camera_id,
#             "user_id": self.user_id,
#             "date_time": date_time,
#             "duration_time": duration_time + threshold_time,
#             "severity": "Critical",
#             "start_time": start_time,
#             "end_time": current_time,
#             "evidence": "https://example.com/evidence/8896",
#             "pilot": "Co-Pilot",
#             "alert_type": alert_type
#         }  
#     def check_and_send_head_pose_alert(self, alert_type, current_time, date_time, start_time, last_alert_duration):
#         index = self.next_alert_index[alert_type]
#         intervals = self.alert_intervals[alert_type]

#         if index < len(intervals) and last_alert_duration >= intervals[index]:
#             alert_data = self.build_alert_data(current_time, date_time, int(last_alert_duration), start_time, alert_type)
#             send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
#             log_info(f"Driver {alert_type} alert and time duration: {int(last_alert_duration)}", date_time, self.rabbitmq_host)

#             self.next_alert_index[alert_type] += 1
#             next_interval = intervals[-1] + (self.next_alert_index[alert_type] * 10)
#             self.alert_intervals[alert_type].append(next_interval)

#     def process_phone_call(self, frame, hand_results, pose_results, img_w, img_h, current_time, date_time):
#         hand_in_box = False
#         hand_open = False
#         person_in_frame = False

#         # Process Hand Landmarks
#         if hand_results.multi_hand_landmarks:
#             for landmarks in hand_results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
#                 finger_tips = {
#                     "thumb": mp_hands.HandLandmark.THUMB_TIP,
#                     "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
#                     "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
#                     "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
#                     "pinky": mp_hands.HandLandmark.PINKY_TIP
#                 }

#                 coords = {key: (int(landmarks.landmark[val].x * img_w), int(landmarks.landmark[val].y * img_h)) for key, val in finger_tips.items()}
#                 thumb_y = landmarks.landmark[finger_tips["thumb"]].y
#                 avg_finger_y = sum(landmarks.landmark[finger_tips[f]].y for f in ["index", "middle", "ring", "pinky"]) / 4

#                 if thumb_y > avg_finger_y:
#                     hand_open = True

#                 for c in coords.values():
#                     cv2.circle(frame, c, 10, (255, 255, 0), -1)

#         # Process Pose Landmarks
#         if pose_results.pose_landmarks:
#             person_in_frame = True
#             landmarks = pose_results.pose_landmarks.landmark

#             def pos(name):
#                 pt = mp_pose.PoseLandmark[name]
#                 return int(landmarks[pt].x * img_w), int(landmarks[pt].y * img_h)

#             lw, rw = pos("LEFT_WRIST"), pos("RIGHT_WRIST")
#             le, re = pos("LEFT_EAR"), pos("RIGHT_EAR")
#             leye, reye = pos("LEFT_EYE"), pos("RIGHT_EYE")
#             ls, rs = pos("LEFT_SHOULDER"), pos("RIGHT_SHOULDER")

#             shoulder_width = calculate_distance(ls, rs)
#             dynamic_threshold = shoulder_width * 1.5

#             box_left = min(leye[0], reye[0], ls[0], rs[0])
#             box_right = max(leye[0], reye[0], ls[0], rs[0])
#             box_top = min(leye[1], reye[1], ls[1], rs[1])
#             box_bottom = max(ls[1], rs[1])

#             cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

#             if calculate_distance(lw, le) < dynamic_threshold and box_left < lw[0] < box_right and box_top < lw[1] < box_bottom:
#                 hand_in_box = True
#             if calculate_distance(rw, re) < dynamic_threshold and box_left < rw[0] < box_right and box_top < rw[1] < box_bottom:
#                 hand_in_box = True

#         # Alert Logic
#         if person_in_frame and hand_open and hand_in_box:
#             cv2.putText(frame, "PERSON ON PHONE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#             if self.phone_detected_time is None:
#                 self.phone_detected_time = current_time

#             elapsed_time = (current_time - self.phone_detected_time).total_seconds()

#             if elapsed_time >= self.threshold:
#                 if self.phone_alert_trigger_time is None:
#                     self.phone_alert_trigger_time = current_time

#                 self.last_phone_alert_duration = (current_time - self.phone_alert_trigger_time).total_seconds()
#                 self.check_and_send_head_pose_alert("Phone Call", current_time, date_time, self.phone_call_start_time, self.last_alert_duration_phone_call)

#                 print("PERSON ON PHONE ALERT!")
               
#         else:
#             if self.phone_alert_trigger_time:
#                 print(f"Last Phone Alert Duration: {int(self.last_phone_alert_duration)} sec")
#                 date_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
#                 alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_phone_call), self.phone_call_start_time, "Phone Call")
#                 send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
#                 log_info(f"Phone alert and time duration :{int(self.last_alert_duration_phone_call)}", date_time, self.rabbitmq_host)


#             self.phone_detected_time = None
#             self.phone_alert_trigger_time = None

#     def frame_process_for_phone_call(self, frame_rgb,current_time, date_time):
#         hand_results = hands.process(frame_rgb)
#         pose_results = pose.process(frame_rgb)
#         img_h, img_w, _ = frame_rgb.shape
#         self.process_phone_call(frame_rgb,hand_results,pose_results,img_h, img_w,current_time, date_time)