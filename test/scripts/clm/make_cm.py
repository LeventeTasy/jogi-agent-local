from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PATH_SAVE = BASE_DIR.parent.parent / "results" / "clm" / "CLM_v0.1-8B-v2.csv"
df = pd.read_csv(PATH_SAVE)

df["prediction"] = df["prediction"].str.lower()

p_s = [0, 0.2, 0.3, 0.4, 0.5, 0.6]

for p in p_s:
    CM_PATH_SAVE = BASE_DIR.parent.parent / "results" / "clm" / "ConfusionMatrices" / f"CLM_v0.1-8B-v2_p{str(p).replace('.', '_')}.csv"

    df_c = df[df["prob"] == p]

    cm = confusion_matrix(
        df_c["label"],
        df_c["prediction"],
        labels=["legal", "not_legal"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["legal", "not_legal"])


    disp.plot(values_format="d")
    plt.title("CLM v0.1 8B")
    plt.gcf().text(0.95, 0.95, f"p={p}", ha="right", va="top")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()