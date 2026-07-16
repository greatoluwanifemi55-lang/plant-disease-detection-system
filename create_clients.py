import random
import shutil
from pathlib import Path

# ======================================================
# PATHS
# ======================================================

PROJECT_DIR = Path(__file__).parent

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"

CLIENT_DIR = PROJECT_DIR / "federated_clients"

CLIENTS = [

    "Oyo",

    "Kaduna",

    "Benue"

]

# ======================================================
# DELETE OLD CLIENT FOLDER
# ======================================================

if CLIENT_DIR.exists():

    shutil.rmtree(CLIENT_DIR)

# ======================================================
# CREATE CLIENT FOLDERS
# ======================================================

for client in CLIENTS:

    for disease in TRAIN_DIR.iterdir():

        (CLIENT_DIR / client / disease.name).mkdir(

            parents=True,

            exist_ok=True

        )

print("Client folders created.")

# ======================================================
# SPLIT DATASET
# ======================================================

random.seed(42)

for disease in TRAIN_DIR.iterdir():

    images = list(disease.iterdir())

    random.shuffle(images)

    split = len(images) // 3

    splits = {

        "Oyo": images[:split],

        "Kaduna": images[split:split*2],

        "Benue": images[split*2:]

    }

    for client, files in splits.items():

        for file in files:

            shutil.copy(

                file,

                CLIENT_DIR /

                client /

                disease.name /

                file.name

            )

print()

print("="*50)

print("Federated dataset created successfully!")

print("="*50)

for client in CLIENTS:

    total = 0

    for disease in (CLIENT_DIR / client).iterdir():

        total += len(list(disease.iterdir()))

    print(client, ":", total, "images")