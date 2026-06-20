import csv
import os

print("LOGGER FILE LOADED")

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
LOG_FILE = os.path.join(LOG_DIR, "query_logs.csv")

print("BASE_DIR:", BASE_DIR)
print("LOG_FILE:", LOG_FILE)

os.makedirs(LOG_DIR, exist_ok=True)


def log_query(timestamp,
              question,
              answer,
              latency,
              chunks_retrieved):

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "question",
                "answer",
                "latency_sec",
                "chunks_retrieved"
            ])

        writer.writerow([
            timestamp,
            question,
            answer,
            round(latency, 2),
            chunks_retrieved
        ])