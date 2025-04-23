
from send_data_into_rabbitmq import log_info, log_exception, log_error, setup_rabbitmq_connection 
import pickle

def send_alert_data(data,exchange_name, rabbitmq_host, date_time):
    # Set up RabbitMQ connection and channel for sending processed frames
    processed_connection, processed_channel = setup_rabbitmq_connection(exchange_name, rabbitmq_host, date_time)
    try:
        serialized_data = pickle.dumps(data)
        if not processed_channel.is_open:
            _, processed_channel = setup_rabbitmq_connection(exchange_name, 
                                                                    rabbitmq_host, date_time
                                                                    )
        processed_channel.basic_publish(
                                    exchange=exchange_name,
                                    routing_key="",
                                    body=serialized_data
                                    )
        log_info("Send data into repoter successfully ", date_time, rabbitmq_host)
    except Exception as e:
        log_exception(f"fail data {e} ", date_time, rabbitmq_host)










# import pickle

# class TemperingAlertSender:
#     def __init__(self, camera_id, user_id, rabbitmq_host, exchange_name,
#                  setup_rabbitmq_connection_for_event, log_error=None, pilot="Co-Pilot"):
#         """
#         Initializes the alert sender with RabbitMQ and camera metadata.
#         """
#         self.camera_id = camera_id
#         self.user_id = user_id
#         self.rabbitmq_host = rabbitmq_host
#         self.exchange_name = exchange_name
#         self.pilot = pilot
#         self.setup_rabbitmq_connection_for_event = setup_rabbitmq_connection_for_event
#         self.log_error = log_error

#     def send_tempering_alert(self, processed_channel, start_time, end_time, date_time, duration_time):
#         """
#         Sends the tempering alert with all metadata serialized and published via RabbitMQ.
#         """
#         alert_data = {
#             "camera_id": self.camera_id,
#             "user_id": self.user_id,
#             "date_time": date_time,
#             "duration_time": duration_time + 10,  # Buffer
#             "severity": "Critical",
#             "start_time": start_time,
#             "end_time": end_time,
#             "evidence": "https://example.com/evidence/8896",
#             "pilot": self.pilot,
#             "alert_type": "Camera tempering"
#         }

#         print("Sending tempering alert:", alert_data)

#         try:
#             serialized_data = pickle.dumps(alert_data)

#             if not processed_channel.is_open:
#                 if self.log_error:
#                     self.log_error("Processed channel is closed. Reconnecting...")
#                 _, processed_channel = self.setup_rabbitmq_connection_for_event(
#                     self.exchange_name, self.rabbitmq_host
#                 )

#             processed_channel.basic_publish(
#                 exchange=self.exchange_name,
#                 routing_key="",
#                 body=serialized_data
#             )

#         except Exception as e:
#             if self.log_error:
#                 self.log_error(f"RabbitMQ error: {str(e)}")

#         return processed_channel

