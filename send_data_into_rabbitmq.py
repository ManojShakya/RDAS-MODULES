import pika     # Importing the RabbitMQ client library to send and receive messages
import pickle   # Used to serialize (save) and deserialize (load) Python objects, such as video frames
import struct   # Used for packing and unpacking binary data, like frame sizes for consistent transmission
import datetime
import logging  # Used for logging messages, helpful for debugging and monitoring the program
import time


def send_log_to_rabbitmq(log_message, rabbitmq_host):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host, heartbeat=600))
        channel = connection.channel()
        channel.queue_declare(queue='rdas_logs')  # Declare the queue for logs
        
        # Serialize the log message as JSON and send it to RabbitMQ
        channel.basic_publish(
            exchange='',
            routing_key='rdas_logs',
            body=pickle.dumps(log_message)
        )
        connection.close()
    except Exception as e:
        print(f"Failed to send log to RabbitMQ: {e}")

# Wrapper functions for logging and sending logs to RabbitMQ
def log_info(message, current_date_time, rabbitmq_host):
    logging.info(message)
    message_data = {
        "log_level" : "INFO",
        "Event_Type":"Start threads for send frames",
        "Message":message,
        "datetime" : current_date_time,

    }
    send_log_to_rabbitmq(message_data, rabbitmq_host)

def log_error(message, current_date_time, rabbitmq_host):
    logging.info(message)
    message_data = {
        "log_level" : "ERROR",
        "Event_Type":"Start threads for send frames",
        "Message":message,
        "datetime" : current_date_time,

    }
    send_log_to_rabbitmq(message_data, rabbitmq_host)    

def log_exception(message, current_date_time, rabbitmq_host):
    logging.error(message)
    message_data = {
        "log_level" : "EXCEPTION",
        "Event_Type":"Start threads for send frames",
        "Message":message,
        "datetime" : current_date_time,

    }
    send_log_to_rabbitmq(message_data, rabbitmq_host)


def setup_rabbitmq_connection(queue_name, rabbitmq_host, current_time, retries=5, retry_delay=5):
    """
    Set up a RabbitMQ connection and declare the queue.
    """
    for attempt in range(retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host, heartbeat=600))
            channel = connection.channel()
            channel.exchange_declare(exchange=queue_name, exchange_type="fanout")
            log_info(f"Connected to RabbitMQ at {rabbitmq_host}", current_time, rabbitmq_host)
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            log_error(f"RabbitMQ connection failed (attempt {attempt+1}/{retries}): {e}", current_time, rabbitmq_host)
            time.sleep(retry_delay)
    raise log_exception(f"Could not connect to RabbitMQ after {retries} attempts", current_time, rabbitmq_host)
