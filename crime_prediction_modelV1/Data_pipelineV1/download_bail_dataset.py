import os
from datasets import load_dataset

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

dataset = load_dataset("SnehaDeshmukh/IndianBailJudgments-1200")

dataset["train"].to_json(
    os.path.join(DATA_DIR, "bail_judgments.json"),
    orient="records",
    lines=True
)

print("dataset downloaded and saved locally")