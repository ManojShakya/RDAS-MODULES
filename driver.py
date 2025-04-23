import pika               # Importing the RabbitMQ client library to send and receive messages
import os                 # Provides access to environment variables and file system-related functions
import cv2                # OpenCV is a popular computer vision library used here for image and video processing
import time
import numpy as np       # NumPy is used for numerical operations and array manipulations
from playsound import playsound  # Plays sound alerts; used for audio notifications (e.g., warning sounds)
import threading         # Allows you to run tasks in parallel using multiple threads, helpful for performance
from ultralytics import YOLO   # Importing the YOLO model from Ultralytics for object detection (e.g., people, helmets, etc.)
from send_data_into_rabbitmq import log_info, log_exception, log_error, setup_rabbitmq_connection  
import datetime
import pickle   # Used to serialize (save) and deserialize (load) Python objects, such as video frames
from camera_tempering import CameraTemperingDetector
from no_driver import NoDriverDetector
from side_distraction import HeadPoseAlertHandler
from drowsiness_yawning import DrowsinessYawnAlertHandler
from phone_call import PhoneCallDetector


'''
Set an environment variable to ensure PyTorch loads full model weights instead of "weights-only" loading mode.
  This is useful when you want to guarantee the model architecture is also loaded correctly with the weights. 
  '''
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'


fire_smoke_model = YOLO("fire_smoke_detectorv8.pt")  # Load a custom YOLOv8 model trained to detect fire and smoke scenarios.
fall_model = YOLO("Fall_detectorv8.pt")              # Load a custom YOLOv8 model trained to detect human falls (for fall detection applications).
model = YOLO("yolov8m.pt")                           # Load a general-purpose YOLOv8 medium model (pretrained or fine-tuned), used for common object detection.
helmet_model = YOLO("hemletYoloV8_100epochs.pt")     # Load a custom YOLOv8 model trained for helmet detection, possibly for safety monitoring or traffic violations.

date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Fetch current date and time

# Dictionary to track detectors for each camera
detectors = {}



def process_frame(ch, method, properties, body, exchange_name, rabbitmq_host):
    try:
        # Deserialize the frame and metadata
        frame_data = pickle.loads(body)
        camera_id = frame_data["camera_id"]
        frame = frame_data["frame"]
        user_id = frame_data["user_id"]
        date_time = frame_data["date_time"]
        
        # Convert string to datetime
        current_time = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
        
        # Preprocess frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Create or retrieve tempering detector
        if camera_id not in detectors:
            detectors[camera_id] = {
                    "tempering": CameraTemperingDetector(tempering_alert_threshold=1,camera_id=camera_id,
                                user_id=user_id,rabbitmq_host=rabbitmq_host,exchange_name=exchange_name),
                    "driver": NoDriverDetector(camera_id=camera_id,user_id=user_id,
                                rabbitmq_host=rabbitmq_host,exchange_name=exchange_name,driver_alert_threshold = 40 ),
                    "distraction" : HeadPoseAlertHandler( camera_id=camera_id,user_id=user_id,
                                    exchange_name=exchange_name,rabbitmq_host=rabbitmq_host,HEAD_POSE_TIME_THRESHOLD=10),
                    "drowsi_yawn": DrowsinessYawnAlertHandler(camera_id= camera_id, user_id=user_id,
                                                              exchange_name=exchange_name,rabbitmq_host=rabbitmq_host,
                                                              DROWSY_TIME_THRESHOLD=5, YAWN_TIME_THRESHOLD=3),
                    "phone_call": PhoneCallDetector(camera_id=camera_id,user_id=user_id,exchange_name=exchange_name,
                                                    rabbitmq_host=rabbitmq_host,PHONE_TIME_THRESHOLD=20)                                                            
                                                            }

        tempering_detector = detectors[camera_id]["tempering"]
        driver_detector = detectors[camera_id]["driver"]
        distraction_detector = detectors[camera_id]["distraction"]
        drowsi_yawn_detector = detectors[camera_id]["drowsi_yawn"]
        phone_call_detector = detectors[camera_id]["phone_call"]

        # Check for camera tempering
        temper_status = tempering_detector.frame_process_for_tempering(
            frame=frame, current_time=current_time, date_time=date_time
           )

        if temper_status.upper() == "FALSE":
            status_driver = driver_detector.frame_process_for_driver(
                            frame_rgb=frame_rgb,current_time=current_time,date_time=date_time
                            )
            status_distraction = distraction_detector.frame_process_for_distraction(
                                frame_rgb = frame_rgb,frame_bgr = frame,
                                current_time = current_time,date_time = date_time
                                )
            status_drowsi_yawn = drowsi_yawn_detector.frame_process_for_drowsiness(frame_rgb,current_time, date_time)
            status_phone_call = phone_call_detector.frame_process_for_phone_call(frame, current_time, date_time) 
            print("Driver status:", status_driver)
            print("Side Distraction Status :", status_distraction)

        print("Tempering status:", temper_status)

    except Exception as e:
        log_exception(f"Error processing frame: {e}", date_time, rabbitmq_host)



def main(queue_name="all_frame_media", exchange_name="drowsiness_alert_data", rabbitmq_host="localhost"):
    """
    Main function to set up RabbitMQ connections for receiving and sending frames.

    Args:
        queue_name (str): The RabbitMQ queue to consume frames from. Defaults to 'video_frames'.
        processed_queue_name (str): The RabbitMQ queue to send processed frames to. Defaults to 'processed_frames'.
    """
    # Set up RabbitMQ connection and channel for receiving frames
    receiver_connection, receiver_channel = setup_rabbitmq_connection(queue_name,rabbitmq_host,date_time)

    # # Set up RabbitMQ connection and channel for sending processed frames
    # processed_connection, processed_channel = setup_rabbitmq_connection(exchange_name, rabbitmq_host, date_time)

    # ---------------------------------------------------
    while True:
        try:
            if not receiver_channel.is_open:
                log_error("Receiver channel is closed. Attempting to reconnect.", date_time, rabbitmq_host)
                time.sleep(25)
                receiver_connection, receiver_channel = setup_rabbitmq_connection(queue_name, rabbitmq_host, date_time)
            # if not processed_channel.is_open:
            #     log_error("Receiver channel is closed. Attempting to reconnect.", date_time, rabbitmq_host)
            #     time.sleep(25)
            #     processed_connection, processed_channel = setup_rabbitmq_connection(exchange_name, rabbitmq_host, date_time)
            
            receiver_channel.queue_declare(queue="all_frame_medias")
            receiver_channel.queue_bind(exchange=queue_name, queue="all_frame_medias")
            receiver_channel.basic_consume(
                queue="all_frame_medias", 
                on_message_callback=lambda ch, method, properties, body: process_frame(
                    ch, method, properties, body, exchange_name, rabbitmq_host
                ),
                auto_ack=True
            )
            log_info("Waiting for video frames...", date_time, rabbitmq_host)
            receiver_channel.start_consuming()
        
        except pika.exceptions.ConnectionClosedByBroker as e:
            log_error("Connection closed by broker, reconnecting...", date_time, rabbitmq_host)
            time.sleep(25)
            receiver_connection, receiver_channel = setup_rabbitmq_connection(queue_name, rabbitmq_host, date_time)
        except Exception as e:
            log_exception(f"Unexpected error: {e}", date_time , rabbitmq_host)
            time.sleep(25)
            continue 

if __name__ == "__main__":
    # Start the receiver and sender
    main()