import mediapipe as mp
import numpy as np
import cv2
import pickle

mp_face_mesh = mp.solutions.face_mesh  # Initialize MediaPipe face mesh solution for detecting and tracking facial landmarks.
mp_hands = mp.solutions.hands          # Initialize MediaPipe hands solution for detecting and tracking hand landmarks.
mp_drawing = mp.solutions.drawing_utils  # Initialize MediaPipe drawing utilities for rendering landmarks (face mesh, hands, pose).


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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


LEFT_EYE = [33, 160, 158, 133, 153, 144]   # Indices for key points around the left eye (used for blink detection, gaze estimation, etc.)
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # Indices for key points around the right eye (same usage as above
NOSE_TIP = 1                               # Index of the tip of the nose (often used as a central reference point for face orientation)
CHIN = 152                                 # Index of the bottom of the chin (can help with head tilt or pose estimation)
LEFT_EAR_POINT = 234                       # Index of a landmark near the left ear (used in head pose estimation and ear detection)
RIGHT_EAR_POINT = 454                      # Index of a landmark near the right ear (used similarly as the left)
UPPER_LIP = 13                             # Index for the upper lip (for mouth state detection, e.g., open/closed, talking, yawning)
LOWER_LIP = 14                             # Index for the lower lip (used with upper lip for mouth-related analysis)


# --------------------------------------------------------------------

class HeadPoseAlertHandler:
    def __init__(self, camera_id, user_id, exchange_name, rabbitmq_host, HEAD_POSE_TIME_THRESHOLD):
        self.camera_id = camera_id
        self.user_id = user_id
        self.exchange_name = exchange_name
        self.rabbitmq_host = rabbitmq_host
        self.HEAD_POSE_TIME_THRESHOLD = HEAD_POSE_TIME_THRESHOLD
        self.head_left_start_time = None
        self.head_left_alert_trigger_time = None
        self.last_alert_duration_head_left = None
        self.head_right_start_time = None
        self.head_right_alert_trigger_time = None
        self.last_alert_duration_head_right = None
        self.head_up_start_time = None
        self.head_up_alert_trigger_time = None
        self.last_alert_duration_head_up = None
        self.head_down_start_time = None
        self.head_down_alert_trigger_time = None
        self.last_alert_duration_head_down = None



    def build_alert_data(self, current_time, date_time, duration_time, dis_head_position_time, alert_type):
        return {
            "camera_id": self.camera_id,
            "user_id": self.user_id,
            "date_time": date_time,
            "duration_time": duration_time + 120,
            "severity": "Critical",
            "start_time": dis_head_position_time,
            "end_time": current_time,
            "evidence": "https://example.com/evidence/8896",
            "pilot": "Co-Pilot",
            "alert_type": alert_type
        }

    def estimate_head_pose(self, face_landmarks, img_w, img_h):
        face_3d = []
        face_2d = []

        for idx, lm in enumerate(face_landmarks):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
        return x, y, z
    
    def process_head_alert(self, angle, threshold, direction, frame, current_time, date_time):
        start_attr = f"head_{direction}_start_time"
        alert_attr = f"head_{direction}_alert_trigger_time"
        duration_attr = f"last_alert_duration_head_{direction}"

        if eval(f"self.{start_attr}") is None:
            setattr(self, start_attr, current_time)

        elapsed_time = (current_time - eval(f"self.{start_attr}")).total_seconds()
        if elapsed_time >= threshold:
            if eval(f"self.{alert_attr}") is None:
                setattr(self, alert_attr, current_time)

            duration = (current_time - eval(f"self.{alert_attr}")).total_seconds()
            setattr(self, duration_attr, duration)
            cv2.putText(frame, f"HEAD {direction.upper()} ALERT!", (50, 250 + 50 * ["left", "right", "up", "down"].index(direction)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
           
        else:
            if eval(f"self.{alert_attr}") is not None:
                print(f"Last Head {direction.capitalize()} Alert Duration: {int(eval(f'self.{duration_attr}'))} sec")
                alert_data = self.build_alert_data(current_time, date_time, int(eval(f"self.{duration_attr}")), eval(f"self.{start_attr}"), f"Head {direction.capitalize()}")
                # Send to RabbitMQ
                serialized_data = pickle.dumps(alert_data)
                if not self.processed_channel.is_open:
                    self.processed_connection, self.processed_channel = setup_rabbitmq_connection_for_event(self.exchange_name, self.rabbitmq_host)
                self.processed_channel.basic_publish(exchange=self.exchange_name, routing_key="", body=serialized_data)

            setattr(self, start_attr, None)
            setattr(self, alert_attr, None)

    

        def frame_process_for_driver(self, frame_rgb, current_time, date_time):
            results = face_mesh.process(frame_rgb)
            img_h, img_w, _ = frame_rgb.shape
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                #hand_landmarks = hand_results.multi_hand_landmarks[0] if hand_results.multi_hand_landmarks else None

                # Head pose detection
                x_angle, y_angle, _ = estimate_head_pose(face_landmarks, img_w, img_h)

                self.process_head_alert(y_angle, self.HEAD_POSE_TIME_THRESHOLD, "left", HEAD_LEFT_ALERT_SOUND, frame, current_time, date_time)
                self.process_head_alert(y_angle, self.HEAD_POSE_TIME_THRESHOLD, "right", HEAD_RIGHT_ALERT_SOUND, frame, current_time, date_time)
                self.process_head_alert(x_angle, self.HEAD_POSE_TIME_THRESHOLD, "up", HEAD_UP_ALERT_SOUND, frame, current_time, date_time)
                self.process_head_alert(x_angle, self.HEAD_POSE_TIME_THRESHOLD, "down", HEAD_DOWN_ALERT_SOUND, frame, current_time, date_time)

        


# import mediapipe as mp
# import numpy as np
# import cv2
# import pickle
# from datetime import datetime
# from db_event_data import send_alert_data


# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # Face mesh setup
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Landmark indices
# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# NOSE_TIP = 1
# CHIN = 152
# LEFT_EAR_POINT = 234
# RIGHT_EAR_POINT = 454
# UPPER_LIP = 13
# LOWER_LIP = 14



# class HeadPoseAlertHandler:
#     def __init__(self, camera_id, user_id, exchange_name, rabbitmq_host, HEAD_POSE_TIME_THRESHOLD):
#         self.camera_id = camera_id
#         self.user_id = user_id
#         self.exchange_name = exchange_name
#         self.rabbitmq_host = rabbitmq_host
#         self.HEAD_POSE_TIME_THRESHOLD = HEAD_POSE_TIME_THRESHOLD
#         self.reset_alert_states()
        
#     def reset_alert_states(self):
#         for direction in ['left', 'right', 'up', 'down']:
#             setattr(self, f"head_{direction}_start_time", None)
#             setattr(self, f"head_{direction}_alert_trigger_time", None)
#             setattr(self, f"last_alert_duration_head_{direction}", None)

#     def build_alert_data(self, current_time, date_time, duration_time, dis_head_position_time, alert_type):
#         return {
#             "camera_id": self.camera_id,
#             "user_id": self.user_id,
#             "date_time": date_time,
#             "duration_time": duration_time + 120,
#             "severity": "Critical",
#             "start_time": dis_head_position_time,
#             "end_time": current_time,
#             "evidence": "https://example.com/evidence/8896",
#             "pilot": "Co-Pilot",
#             "alert_type": alert_type
#         }

#     def estimate_head_pose(self, face_landmarks, img_w, img_h):
#         face_3d = []
#         face_2d = []

#         for idx, lm in enumerate(face_landmarks):
#             if idx in [33, 263, 1, 61, 291, 199]:
#                 x, y = int(lm.x * img_w), int(lm.y * img_h)
#                 face_2d.append([x, y])
#                 face_3d.append([x, y, lm.z])

#         face_2d = np.array(face_2d, dtype=np.float64)
#         face_3d = np.array(face_3d, dtype=np.float64)

#         focal_length = 1 * img_w
#         cam_matrix = np.array([
#             [focal_length, 0, img_h / 2],
#             [0, focal_length, img_w / 2],
#             [0, 0, 1]
#         ])

#         dist_matrix = np.zeros((4, 1), dtype=np.float64)

#         success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#         rmat, _ = cv2.Rodrigues(rot_vec)
#         angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

#         x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
#         return x, y, z

#     def process_head_alert(self, angle, threshold, direction, frame, current_time, date_time):
#         start_attr = f"head_{direction}_start_time"
#         alert_attr = f"head_{direction}_alert_trigger_time"
#         duration_attr = f"last_alert_duration_head_{direction}"

#         if getattr(self, start_attr) is None:
#             setattr(self, start_attr, current_time)

#         elapsed_time = (current_time - getattr(self, start_attr)).total_seconds()
#         if elapsed_time >= threshold:
#             if getattr(self, alert_attr) is None:
#                 setattr(self, alert_attr, current_time)

#             duration = (current_time - getattr(self, alert_attr)).total_seconds()
#             setattr(self, duration_attr, duration)
#             cv2.putText(frame, f"HEAD {direction.upper()} ALERT!", (50, 250 + 50 * ["left", "right", "up", "down"].index(direction)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         else:
#             if getattr(self, alert_attr) is not None:
#                 print(f"Last Head {direction.capitalize()} Alert Duration: {int(getattr(self, duration_attr))} sec")
#                 alert_data = self.build_alert_data(current_time, date_time, int(getattr(self, duration_attr)), getattr(self, start_attr), f"Head {direction.capitalize()}")
#                 send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
#                 #serialized_data = pickle.dumps(alert_data)

                
#             setattr(self, start_attr, None)
#             setattr(self, alert_attr, None)

#     def frame_process_for_driver(self, frame_rgb, frame_bgr, current_time, date_time):
#         results = face_mesh.process(frame_rgb)
#         img_h, img_w, _ = frame_rgb.shape

#         if results.multi_face_landmarks:
#             face_landmarks = results.multi_face_landmarks[0].landmark
#             x_angle, y_angle, _ = self.estimate_head_pose(face_landmarks, img_w, img_h)

#             self.process_head_alert(y_angle, self.HEAD_POSE_TIME_THRESHOLD, "left", frame_bgr, current_time, date_time)
#             self.process_head_alert(-y_angle, self.HEAD_POSE_TIME_THRESHOLD, "right", frame_bgr, current_time, date_time)
#             self.process_head_alert(x_angle, self.HEAD_POSE_TIME_THRESHOLD, "up", frame_bgr, current_time, date_time)
#             self.process_head_alert(-x_angle, self.HEAD_POSE_TIME_THRESHOLD, "down", frame_bgr, current_time, date_time)









    


# import mediapipe as mp
# import numpy as np
# import cv2
# import pickle
# from datetime import datetime
# from db_event_data import send_alert_data


# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # Face mesh setup
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Landmark indices
# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# NOSE_TIP = 1
# CHIN = 152
# LEFT_EAR_POINT = 234
# RIGHT_EAR_POINT = 454
# UPPER_LIP = 13
# LOWER_LIP = 14



# import cv2
# import numpy as np

# class HeadPoseAlertHandler:
#     def __init__(self, camera_id, user_id, exchange_name, rabbitmq_host, time_threshold):
#         self.camera_id = camera_id
#         self.user_id = user_id
#         self.exchange_name = exchange_name
#         self.rabbitmq_host = rabbitmq_host
#         self.time_threshold = time_threshold
#         self.reset_all()

#     def reset_all(self):
#         self.directions = ['left', 'right', 'up', 'down']
#         self.state = {}
#         for dir in self.directions:
#             self.state[dir] = {
#                 'start_time': None,
#                 'alert_time': None,
#                 'duration': None,
#                 'intervals': [10, 20, 30],
#                 'next_index': 0
#             }

#     def build_alert_data(self, now, date_time, duration, start_time, alert_type):
#         return {
#             "camera_id": self.camera_id,
#             "user_id": self.user_id,
#             "date_time": date_time,
#             "duration_time": duration + 120,
#             "severity": "Critical",
#             "start_time": start_time,
#             "end_time": now,
#             "evidence": "https://example.com/evidence/8896",
#             "pilot": "Co-Pilot",
#             "alert_type": alert_type
#         }

#     def process_head_alert(self, angle, direction, frame, now, date_time):
#         state = self.state[direction]

#         if state['start_time'] is None:
#             state['start_time'] = now

#         seconds_wrong = (now - state['start_time']).total_seconds()

#         if seconds_wrong >= self.time_threshold:
#             if state['alert_time'] is None:
#                 state['alert_time'] = now

#             seconds_alerting = (now - state['alert_time']).total_seconds()
#             state['duration'] = seconds_alerting

#             cv2.putText(frame, f"HEAD {direction.upper()} ALERT!", (50, 250 + 50 * self.directions.index(direction)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#             if state['next_index'] < len(state['intervals']):
#                 wait_time = state['intervals'][state['next_index']]
#                 if seconds_alerting >= wait_time:
#                     alert = self.build_alert_data(now, date_time, int(seconds_alerting), state['start_time'], f"Head {direction.capitalize()}")
#                     send_alert_data(alert, self.exchange_name, self.rabbitmq_host, date_time)
#                     print(f"🚨 Alert sent for {direction} — {int(seconds_alerting)}s")

#                     state['next_index'] += 1
#                     if state['next_index'] > 2:  # after 30s, increase more
#                         state['intervals'].append(state['intervals'][-1] + 10)
#         else:
#             if state['alert_time'] is not None:
#                 print(f"✅ {direction} OK. Duration was {int(state['duration'])}s")
#             # reset direction state
#             state.update({
#                 'start_time': None,
#                 'alert_time': None,
#                 'duration': None,
#                 'intervals': [10, 20, 30],
#                 'next_index': 0
#             })

#     def estimate_head_pose(self, face_landmarks, img_w, img_h):
#         face_3d = []
#         face_2d = []
#         for idx, lm in enumerate(face_landmarks):
#             if idx in [33, 263, 1, 61, 291, 199]:
#                 x, y = int(lm.x * img_w), int(lm.y * img_h)
#                 face_2d.append([x, y])
#                 face_3d.append([x, y, lm.z])
#         face_2d = np.array(face_2d, dtype=np.float64)
#         face_3d = np.array(face_3d, dtype=np.float64)

#         focal_length = 1 * img_w
#         cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
#         dist_matrix = np.zeros((4, 1), dtype=np.float64)

#         success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#         rmat, _ = cv2.Rodrigues(rot_vec)
#         angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
#         return angles[0] * 360, angles[1] * 360, angles[2] * 360

#     def frame_process(self, frame_rgb, frame_bgr, now, date_time, face_mesh):
#         results = face_mesh.process(frame_rgb)
#         img_h, img_w, _ = frame_rgb.shape

#         if results.multi_face_landmarks:
#             face_landmarks = results.multi_face_landmarks[0].landmark
#             x_angle, y_angle, _ = self.estimate_head_pose(face_landmarks, img_w, img_h)

#             self.process_head_alert(y_angle, 'left', frame_bgr, now, date_time)
#             self.process_head_alert(-y_angle, 'right', frame_bgr, now, date_time)
#             self.process_head_alert(x_angle, 'up', frame_bgr, now, date_time)
#             self.process_head_alert(-x_angle, 'down', frame_bgr, now, date_time)
