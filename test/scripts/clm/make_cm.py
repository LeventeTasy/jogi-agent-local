from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PATH_SAVE = BASE_DIR.parent.parent / "results" / "clm" / "CLM_v0.1-8B-v2.csv"
df = pd.read_csv(PATH_SAVE)

df["prediction"] = df["prediction"].str.lower()
p = 0.3

p_s = [0, 0.2, 0.3, 0.4, 0.5, 0.6]

for p in p_s:
    CM_PATH_SAVE = BASE_DIR.parent.parent / "results" / "clm" / "ConfusionMatrices" / f"CLM_v0.1-8B-v2_p{str(p).replace('.', '_')}.png"

    df_c = df[df["prob"] == p]

    llm_percentage = df_c["is_llm"].mean() * 100
    accuracy = accuracy_score(df_c["label"], df_c["prediction"])

    cm = confusion_matrix(
        df_c["label"],
        df_c["prediction"],
        labels=["legal", "not_legal"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["legal", "not_legal"])


    disp.plot(values_format="d")
    plt.title(
        f"CLM v0.1 8B\n"
        f"p={p} | LLM usage: {llm_percentage:.2f}% | Accuracy: {accuracy:.2%}",
        pad=15
    )

    plt.tight_layout()
    plt.savefig(CM_PATH_SAVE, dpi=300, bbox_inches="tight")
    plt.show()