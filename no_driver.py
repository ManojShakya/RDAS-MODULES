import mediapipe as mp    # MediaPipe is used for detecting human body landmarks like hands, face, and pose
from send_data_into_rabbitmq import log_error, log_info, log_exception
from db_event_data import send_alert_data


mp_pose = mp.solutions.pose   
mp_drawing = mp.solutions.drawing_utils  # Initialize MediaPipe drawing utilities for rendering landmarks (face mesh, hands, pose).           # Initialize MediaPipe pose solution for full-body landmark detection (e.g., for no-driver or fall detection scenarios).
pose = mp_pose.Pose()                    # Create a pose detector instance.

#-----------------------------------------------------


class NoDriverDetector:
    def __init__(self, camera_id, user_id, rabbitmq_host, exchange_name, driver_alert_threshold):
        self.camera_id = camera_id
        self.user_id = user_id
        self.rabbitmq_host = rabbitmq_host
        self.exchange_name = exchange_name
        self.driver_alert_threshold = driver_alert_threshold

        self.no_driver_start_time = None
        self.no_driver_alert_trigger_time = None
        self.last_alert_duration_no_driver = 0

        # For periodic alerting
        self.no_driver_alert_intervals = [driver_alert_threshold]
        self.next_alert_index = 0

    def build_alert_data(self, current_time, date_time, duration_time):
        return {
            "camera_id": self.camera_id,
            "user_id": self.user_id,
            "date_time": date_time,
            "duration_time": duration_time + 40,
            "severity": "Critical",
            "start_time": self.no_driver_start_time,
            "end_time": current_time,
            "evidence": "https://example.com/evidence/8896",
            "pilot": "Co-Pilot",
            "alert_type": "NO Driver"
        }

    def reset(self):
        self.no_driver_start_time = None
        self.no_driver_alert_trigger_time = None
        self.last_alert_duration_no_driver = 0
        self.no_driver_alert_intervals = [self.driver_alert_threshold]
        self.next_alert_index = 0

    def frame_process_for_driver(self, frame_rgb, current_time, date_time):
        driver_results = pose.process(frame_rgb)

        if driver_results.pose_landmarks:
            print("✅ Driver is detected")

            if self.no_driver_start_time is not None:
                self.last_alert_duration_no_driver = (current_time - self.no_driver_start_time).total_seconds()
                duration_time = int(self.last_alert_duration_no_driver)

                alert_data = self.build_alert_data(current_time, date_time, duration_time)
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"👤 Driver returned — Last no-driver alert duration: {duration_time}s", date_time, self.rabbitmq_host)
                print("Sending NO driver alert data :", alert_data)

            self.reset()
            return "Driver"

        else:
            if self.no_driver_start_time is None:
                self.no_driver_start_time = current_time
                print("🚨 No driver detected started timing...")
            else:
                elapsed_time = (current_time - self.no_driver_start_time).total_seconds()

                if elapsed_time >= self.driver_alert_threshold:
                    if self.no_driver_alert_trigger_time is None:
                        self.no_driver_alert_trigger_time = current_time

                    self.last_alert_duration_no_driver = (current_time - self.no_driver_alert_trigger_time).total_seconds()

                    print(f"⏱ No driver alert running — {int(self.last_alert_duration_no_driver)} sec")

                    # Check if it's time for the next alert
                    if self.next_alert_index < len(self.no_driver_alert_intervals) and \
                        self.last_alert_duration_no_driver >= self.no_driver_alert_intervals[self.next_alert_index]:

                        duration_time = int(self.last_alert_duration_no_driver)
                        alert_data = self.build_alert_data(current_time, date_time, duration_time)
                        send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                        log_info(f"🚨 No driver alert sent — Duration: {duration_time}s", date_time, self.rabbitmq_host)

                        # Update alert intervals
                        self.next_alert_index += 1
                        if self.next_alert_index > 0:
                            delta = self.next_alert_index * 10
                            next_time = self.no_driver_alert_intervals[-1] + delta
                            self.no_driver_alert_intervals.append(next_time)

        return "NoDriver"






