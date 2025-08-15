from google.cloud import pubsub_v1
import os
import json

PROJECT_ID = os.getenv('pipeline-ml-wine-dataset')
TOPIC_ID = "status-preprocessing-topic"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

def notify_status_change(event, context):
    file_name = event['name']
    if file_name != "status_preprocessing.json":
        return  # ignora file diversi

    message_json = json.dumps(event).encode("utf-8")
    future = publisher.publish(topic_path, message_json)
    future.result()