import csv
from pathlib import Path

RESULTS_DIR = Path("experiment_results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "metrics.csv"


def initialize_metrics():

    with open(RESULTS_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Round",
            "Client",
            "Loss",
            "Accuracy",
        ])


def save_metrics(round_number, client_name, loss, accuracy):

    with open(RESULTS_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            round_number,
            client_name,
            float(loss),
            float(accuracy),
        ])