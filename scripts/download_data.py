
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Dataset UCI ID 697
DEST = Path("data/student_dropout.csv")
DEST.parent.mkdir(parents=True, exist_ok=True)

if DEST.exists() and DEST.stat().st_size > 1000:
    print(f"Dataset already exists: {DEST}")
else:
    print("Downloading UCI Student Dropout Dataset...")

    dataset = fetch_ucirepo(id=697)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    df.to_csv(DEST, index=False)

    print(f"Saved dataset to {DEST}")
    print("Dataset shape:", df.shape)
    print(df.head())

print("Download completed!")
