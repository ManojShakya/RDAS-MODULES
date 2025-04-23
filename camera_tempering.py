import cv2
import numpy as np
import pickle
from datetime import datetime
from send_data_into_rabbitmq import log_info, log_exception, log_error
from db_event_data import send_alert_data
import time



class CameraTemperingDetector:
    def __init__(self,tempering_alert_threshold=None,
                 camera_id=None, user_id=None,
                 rabbitmq_host=None, exchange_name=None,
                 #setup_rabbitmq_connection=None,
                ):
        """
        Initializes the detector for both obstruction detection and tempering alert management.
        """
        self.edge_threshold = 10
        self.tempering_alert_threshold = tempering_alert_threshold
        self.camera_id = camera_id
        self.user_id = user_id
        self.rabbitmq_host = rabbitmq_host
        self.exchange_name = exchange_name
        #self.setup_rabbitmq_connection = setup_rabbitmq_connection

        # Internal state
        self.no_edge_start_time = None
        self.tempering_alert_trigger_time = None
        self.last_alert_duration_tempering = None
        self.next_tempering_alert_time = None
        self.tempering_alert_intervals = [0]  # 0, 10, 30, ...
        self.next_alert_index = 0

    def is_camera_covered(self, frame):
        """
        Checks if the camera is covered using Canny edge detection.
        Returns: (bool: is_covered, np.ndarray: edges)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.sum(edges > 0)
        #print("detecting camera tempering")
        return edge_pixels < self.edge_threshold, edges
    
    def build_alert_data(self, current_time, date_time, duration_time):
        return {
            "camera_id": self.camera_id,
            "user_id": self.user_id,
            "date_time": date_time,
            "duration_time": duration_time + 10,
            "severity": "Critical",
            "start_time": self.no_edge_start_time,
            "end_time": current_time,
            "evidence": "https://example.com/evidence/8896",
            "pilot": "Co-Pilot",
            "alert_type": "Camera tempering"
        }
    
    def frame_process_for_tempering(self, frame, current_time, date_time):
        """
        Processes the current frame for obstruction and tempering.
        Returns the updated frame and possibly updated processed_channel.
        """
        is_covered, edges = self.is_camera_covered(frame)

        if is_covered:
            # Set start time for tempering detection if it's not set already
            if self.no_edge_start_time is None:
                self.no_edge_start_time = current_time
                print("no_edge_start_time set to:", self.no_edge_start_time)

            # Calculate the elapsed time since the camera was covered
            elapsed = (current_time - self.no_edge_start_time).total_seconds()
            print("this is current time:", current_time)
            #print("Camera tempering alert step 1", elapsed)

            if elapsed >= self.tempering_alert_threshold:
                # Trigger the tempering alert if not already triggered
                if self.tempering_alert_trigger_time is None:
                    self.tempering_alert_trigger_time = current_time
                    self.tempering_alert_intervals = [0]  # Reset for new cycle
                    self.next_alert_index = 0

                # Calculate the duration of the tempering alert
                self.last_alert_duration_tempering = (current_time - self.tempering_alert_trigger_time).total_seconds()

                # # Visual Alert
                # cv2.putText(frame, "CAMERA TEMPERING ALERT!", (50, 400),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                print("Camera tempering alert step 2")

                # Check if it's time for the next alert
                if self.next_alert_index < len(self.tempering_alert_intervals) and \
                self.last_alert_duration_tempering >= self.tempering_alert_intervals[self.next_alert_index]:
                    duration_time = int(self.last_alert_duration_tempering)
                    alert_data = self.build_alert_data(current_time, date_time, duration_time=duration_time)
                    send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                    log_info(f"Camera tempering alert and time duration :{duration_time}", date_time, self.rabbitmq_host)

                    # Update the alert intervals for subsequent alerts
                    self.next_alert_index += 1
                    if self.next_alert_index > 0:
                        # Add next time interval (e.g., 10, 30, 60, 100...)
                        delta = self.next_alert_index * 10
                        next_time = self.tempering_alert_intervals[-1] + delta
                        self.tempering_alert_intervals.append(next_time)

            return "True"        

        else:
            # If previously triggered (camera no longer covered)
            if self.tempering_alert_trigger_time and self.last_alert_duration_tempering:
                print(f"Tempering duration: {int(self.last_alert_duration_tempering)} sec")
                duration_time = int(self.last_alert_duration_tempering)

                # Send final alert once the tempering has been cleared
                alert_data = self.build_alert_data(current_time, date_time, duration_time=duration_time)
                send_alert_data(alert_data, self.exchange_name, self.rabbitmq_host, date_time)
                log_info(f"Camera tempering alert and time duration :{duration_time}", date_time, self.rabbitmq_host)

                print("Sending tempering alert:", alert_data)

            # Reset state
            self.no_edge_start_time = None
            self.tempering_alert_trigger_time = None
            self.last_alert_duration_tempering = None
            self.tempering_alert_intervals = [0]
            self.next_alert_index = 0

            return "False"
