from main import preprocessing

class Event:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

def test_preprocessing():
    event = {
        "bucket": "wine-dataset-originali",
        "name": "wine-dataset_unito_60963e3cf51f4efba2844806da68e46b_20250830T220637.csv"
    }
    context = None

    print(">>> Avvio test preprocessing...")
    preprocessing(event, context)
    print(">>> Test completato.")

if __name__ == "__main__":
    test_preprocessing()