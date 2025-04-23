import mediapipe as mp
import numpy as np
import cv2
import pickle
from db_event_data import send_alert_data
from send_data_into_rabbitmq import log_info, log_exception, log_error

mp_face_mesh = mp.solutions.face_mesh  # Initialize MediaPipe face mesh solution for detecting and tracking facial landmarks.

''' Create a face mesh detector instance:
 static_image_mode=False: treats input as a video stream.
 max_num_faces=1: detects only one face.
 refine_landmarks=True: enables attention mesh for more accurate iris/contour detection. 
 '''
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


class HeadPoseAlertHandler:
    def __init__(self, camera_id, user_id, exchange_name, rabbitmq_host, HEAD_POSE_TIME_THRESHOLD):
        self.camera_id = camera_id
        self.user_id = user_id
        self.exchange_name = exchange_name
        self.rabbitmq_host = rabbitmq_host
        self.HEAD_POSE_TIME_THRESHOLD = HEAD_POSE_TIME_THRESHOLD
        self.alert_intervals = {
                                "Head Left": [10],
                                "Head Right": [10],
                                "Head Up": [10],
                                "Head Down": [10]
                            }
        self.next_alert_index = {
                                "Head Left": 0,
                                "Head Right": 0,
                                "Head Up": 0,
                                "Head Down": 0
                            }

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

    def build_alert_data(self, current_time, date_time, duration_time, start_time, alert_type):
        return {
            "camera_id": self.camera_id,
            "user_id": self.user_id,
            "date_time": date_time,
            "duration_time": duration_time + 120,
            "severity": "Critical",
            "start_time": start_time,
            "end_time": current_time,
            "evidence": "https://example.com/evidence/8896",
            "pilot": "Co-Pilot",
            "alert_type": alert_type
        }

    def estimate_head_pose(self, face_landmarks, img_w, img_h):
        face_2d = []
        face_3d = []

        for idx, lm in enumerate(face_landmarks):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
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

    def process_head_left(self, angle, frame, current_time, date_time):
        if angle < -15:
            if self.head_left_start_time is None:
                self.head_left_start_time = current_time
                self.next_alert_index["Head Left"] = 0
                self.alert_intervals["Head Left"] = [10]

            elapsed = (current_time - self.head_left_start_time).total_seconds()
            if elapsed >= self.HEAD_POSE_TIME_THRESHOLD:
                if self.head_left_alert_trigger_time is None:
                    self.head_left_alert_trigger_time = current_time

                self.last_alert_duration_head_left = (current_time - self.head_left_alert_trigger_time).total_seconds()
                #cv2.putText(frame, "HEAD LEFT ALERT!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                self.check_and_send_head_pose_alert("Head Left", current_time, date_time, self.head_left_start_time, self.last_alert_duration_head_left)
            return "Head Left not"
        else:
            if self.head_left_alert_trigger_time is not None:
                print(f"Left alert duration: {int(self.last_alert_duration_head_left)} sec")
                alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_head_left), self.head_left_start_time, "Head Left")
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"Driver head Left alert and time duration :{int(self.last_alert_duration_head_left)}", date_time, self.rabbitmq_host)

                

            self.head_left_start_time = None
            self.head_left_alert_trigger_time = None
            self.next_alert_index["Head Left"] = 0
            self.alert_intervals["Head Left"] = [10]
            return "Head Left"

    def process_head_right(self, angle, frame, current_time, date_time):
        if angle > 15:
            if self.head_right_start_time is None:
                self.head_right_start_time = current_time
                self.next_alert_index["Head Right"] = 0
                self.alert_intervals["Head Right"] = [10]

            elapsed = (current_time - self.head_right_start_time).total_seconds()
            if elapsed >= self.HEAD_POSE_TIME_THRESHOLD:
                if self.head_right_alert_trigger_time is None:
                    self.head_right_alert_trigger_time = current_time

                self.last_alert_duration_head_right = (current_time - self.head_right_alert_trigger_time).total_seconds()
                #cv2.putText(frame, "HEAD RIGHT ALERT!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                self.check_and_send_head_pose_alert("Head Right", current_time, date_time, self.head_right_start_time, self.last_alert_duration_head_right)
            return "Head Right"

        else:
            if self.head_right_alert_trigger_time is not None:
                print(f"Right alert duration: {int(self.last_alert_duration_head_right)} sec")
                alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_head_right), self.head_right_start_time, "Head Right")
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"Driver head Right alert and time duration :{int(self.last_alert_duration_head_right)}", date_time, self.rabbitmq_host)

                
            self.head_right_start_time = None
            self.head_right_alert_trigger_time = None
            self.next_alert_index["Head Right"] = 0
            self.alert_intervals["Head Right"] = [10]
            return "Head Right not"

    def process_head_up(self, angle, frame, current_time, date_time):
        if angle < -10:
            if self.head_up_start_time is None:
                self.head_up_start_time = current_time
                self.next_alert_index["Head Up"] = 0
                self.alert_intervals["Head Up"] = [10]

            elapsed = (current_time - self.head_up_start_time).total_seconds()
            if elapsed >= self.HEAD_POSE_TIME_THRESHOLD:
                if self.head_up_alert_trigger_time is None:
                    self.head_up_alert_trigger_time = current_time

                self.last_alert_duration_head_up = (current_time - self.head_up_alert_trigger_time).total_seconds()
                #cv2.putText(frame, "HEAD UP ALERT!", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                self.check_and_send_head_pose_alert("Head Up", current_time, date_time, self.head_up_start_time, self.last_alert_duration_head_up)
            return "Head Up"

        else:
            if self.head_up_alert_trigger_time is not None:
                print(f"Up alert duration: {int(self.last_alert_duration_head_up)} sec")
                alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_head_up), self.head_up_start_time, "Head Up")
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"Driver head Up alert and time duration :{int(self.last_alert_duration_head_up)}", date_time, self.rabbitmq_host)


            self.head_up_start_time = None
            self.head_up_alert_trigger_time = None
            self.next_alert_index["Head Up"] = 0
            self.alert_intervals["Head Up"] = [10]
            return "Head Up not"

    def process_head_down(self, angle, frame, current_time, date_time):
        if angle > 10:
            if self.head_down_start_time is None:
                self.head_down_start_time = current_time
                self.next_alert_index["Head Down"] = 0
                self.alert_intervals["Head Down"] = [10]

            elapsed = (current_time - self.head_down_start_time).total_seconds()
            if elapsed >= self.HEAD_POSE_TIME_THRESHOLD:
                if self.head_down_alert_trigger_time is None:
                    self.head_down_alert_trigger_time = current_time

                self.last_alert_duration_head_down = (current_time - self.head_down_alert_trigger_time).total_seconds()
                #cv2.putText(frame, "HEAD DOWN ALERT!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                self.check_and_send_head_pose_alert("Head Down", current_time, date_time, self.head_down_start_time, self.last_alert_duration_head_down)
            return "Head Down"

        else:
            if self.head_down_alert_trigger_time is not None:
                print(f"Down alert duration: {int(self.last_alert_duration_head_down)} sec")
                alert_data = self.build_alert_data(current_time, date_time, int(self.last_alert_duration_head_down), self.head_down_start_time, "Head Down")
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"Driver head down alert and time duration :{int(self.last_alert_duration_head_down)}", date_time, self.rabbitmq_host)

                
            self.head_down_start_time = None
            self.head_down_alert_trigger_time = None
            self.next_alert_index["Head Down"] = 0
            self.alert_intervals["Head Down"] = [10]
            return "Head Down not"

    def frame_process_for_distraction(self, frame_rgb, frame_bgr, current_time, date_time):
        results = face_mesh.process(frame_rgb)
        img_h, img_w, _ = frame_rgb.shape

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            x_angle, y_angle, _ = self.estimate_head_pose(face_landmarks, img_w, img_h)

            self.process_head_left(y_angle, frame_bgr, current_time, date_time)
            self.process_head_right(y_angle, frame_bgr, current_time, date_time)
            self.process_head_up(x_angle, frame_bgr, current_time, date_time)
            self.process_head_down(x_angle, frame_bgr, current_time, date_time)
