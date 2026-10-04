import os
import sys
import pandas as pd
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (BASE_DIR.parent/ "datasets"/ "model_comparison"/ "test_questions.csv")
SAVE_PATH = (BASE_DIR.parent/ "results"/ "approved"/ "reviewed_questions.csv")


df = pd.read_csv(INPUT_PATH)

print(f"\nÖsszes kérdés: {len(df)}")
print(f"Input: {INPUT_PATH}")
print(f"Output: {SAVE_PATH}\n")


def save_dataframe():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")


def print_question(index, row):
    print("\n" + "=" * 100)
    print(f" KÉRDÉS {index + 1} / {len(df)}")
    print("=" * 100)

    print(f"\nTörvény:")
    print(row.get("Torveny", ""))

    print(f"\nTípus:")
    print(row.get("Tipus", ""))

    print(f"\nKÉRDÉS:")
    print(row.get("Kerdes", ""))

    print(f"\nQ_CHUNK:")
    print("-" * 100)
    print(row.get("Q_chunk", ""))
    print("-" * 100)

    print(f"\nA_CHUNK:")
    print("-" * 100)
    print(row.get("A_chunk", ""))
    print("-" * 100)


def edit_question(index):
    old_question = str(df.at[index, "Kerdes"])

    print("\nRégi kérdés:")
    print(old_question)

    print("\nÍrd be az új kérdést.")
    print("Üres input esetén megszakítjuk a szerkesztést.")

    new_question = input("\nÚj kérdés: ").strip()

    if new_question == "":
        print("Szerkesztés megszakítva.")
        return False

    df.at[index, "Kerdes"] = new_question

    print("\nKérdés módosítva.")
    return True


# REVIEW

index = 0

while index < len(df):

    row = df.iloc[index]
    jelenlegi_kerdes = row["Kerdes"]

    # Üres kérdés kezelése

    if pd.isna(jelenlegi_kerdes) or str(jelenlegi_kerdes).strip() == "":
        print(f"\nÜres kérdés a(z) {index + 1}. sorban.")

        action = input("[d] törlés / [s] kihagyás / [q] kilépés: ").strip().lower()

        if action == "d":
            df.drop(index=index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            save_dataframe()
            print("✓ Sor törölve.")
            continue

        elif action == "q":
            save_dataframe()
            print("\nMentve. Kilépés.")
            break

        else:
            index += 1
            continue

    print_question(index, row)

    print("\n" + "-" * 100)
    print("MIT SZERETNÉL?")
    print("-" * 100)
    print("[y] Jó → megtartás")
    print("[n] Rossz → törlés")
    print("[e] Szerkesztés")
    print("[s] Kihagyás")
    print("[q] Kilépés + mentés")
    print("-" * 100)

    action = input("Választás: ").strip().lower()

    # JÓ
    if action == "y":
        print("Jóváhagyva.")
        index += 1

    # TÖRLÉS
    elif action == "n":

        confirm = input("Biztosan törlöd ezt a kérdést? [y/n]: ").strip().lower()

        if confirm == "y":

            df.drop(index=index, inplace=True)
            df.reset_index(drop=True, inplace=True)

            save_dataframe()

            print("Kérdés törölve.")
        else:
            print("Törlés megszakítva.")


    # edit
    elif action == "e":
        changed = edit_question(index)

        if changed:
            save_dataframe()

    elif action == "s":
        print("Kihagyva.")
        index += 1

    elif action == "q":
        save_dataframe()
        print("\nAktuális állapot elmentve. Kilépés")
        break

    else:
        print("Ismeretlen parancs.")


save_dataframe()

print("\n" + "=" * 100)
print("REVIEW BEFEJEZVE")
print("=" * 100)

print(f"Fennmaradó kérdések: {len(df)}")
print(f"Mentve ide: {SAVE_PATH}")