from google.cloud import logging

client = logging.Client()

entries = list(client.list_entries(page_size=5))
for entry in entries:
    print(entry.payload)
