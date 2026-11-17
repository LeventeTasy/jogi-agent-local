from datetime import datetime
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

#from jogi_agent.flow import JogiFlow

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR.parent.parent / "results" / "scraping" / "scraped.csv"

COLUMNS = ["Buntett", "Jogalap", "Model_Paragrafus", "Helyesseg", "Valasz", "Runtime", "Timestamp", "Verifier_Agent_Runs", "Question_ID", "Total_Tokens"]
chatID = f"T_{datetime.now().strftime("%Y-%m-%d %H:%M:%S").upper()}"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    #df.to_csv(file_path, index=False, encoding="utf-8-sig")
else:
    df = pd.DataFrame(columns=COLUMNS)
    #df.to_csv(file_path, index=False, encoding="utf-8-sig")

def scraping():
    jogalap = set()

    for i in range(1000):
        url = f"https://www.police.hu/hu/koral/elfogatoparancs-alapjan-korozott-szemelyek?ent_szemely_elfpar_viselt_nev_teljes=&ent_szemely_elfpar_szuletesi_hely=&ent_szemely_elfpar_kore_jogalap_fk_kod_ertekek=All&ent_szemely_elfpar_kore_szerv_fk_kod_ertekek=All&ent_szemely_elfpar_kori_szerv_fk_kod_ertekek=All&min=&max=&ent_szemely_elfpar_nem_fk_kod_ertekek=All&page={i}"
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            captions = soup.find_all("div", class_="caption")

            for caption in captions:
                txt = caption.text.strip()

                if "BTK" in txt:
                    jogalap.update([txt])

            print(len(jogalap))

    print(jogalap)
    jogalap = list(jogalap)

    for i in jogalap:
        splitted = i.split(" - ")

        if splitted[0] not in df["Buntett"].values:
            new_row = {
                "Buntett" : splitted[0],
                "Jogalap" : splitted[1],
                "Helyesseg" : "",
                "Model_Paragrafus" : None,
                "Valasz": None, "Runtime": None, "Timestamp": None, "Verifier_Agent_Runs": None, "Question_ID": None, "Total_Tokens": None
            }
            df.loc[len(df)] = new_row

    df.to_csv(file_path, index=False)

def run_model():
    PROMPT = "Kérlek írd le, hogy a következő elkövetett bűntett milyen BTK szerinti bűncselekmény: BŰNTETT"

    for index, row in df.iterrows():

        m_parag = row['Model_Paragrafus']
        buntett = row["Buntett"]

        parag_ures = pd.isna(m_parag) or str(m_parag).strip() == ""

        if not parag_ures:
            print(f"Sor [{index}] már ki van töltve, kihagyás.")
            continue

        print(f"Sor [{index}]: Feldolgozás alatt... Kérdés: '{buntett[:50]}...'")

        kerdes = PROMPT.replace("BŰNTETT", buntett)
        inputs = {
                'topic': kerdes,
                'pdf_text': "",
                'history': "",
                'details': "",
                'da_questions': "",
                'da_answers': "",
                'username': "S_1",
                "chatID": chatID,
                "questionNumber": index
            }


        start_time = time.perf_counter()
        flow = JogiFlow()
        flow.state["inputs"] = inputs

        answer = str(flow.kickoff())
        elapsed_time = time.perf_counter() - start_time

        print(answer+"\n\n")
        chunks_list = flow.get_chunks()
        parag_string = "\n\n".join(chunks_list)

        verifier_retries = flow.get_verifier_counter()
        questionID = flow.get_question_id()
        metrics = flow.get_metrics()

        df.at[index, 'Total_Tokens'] = metrics["totalTokens"]
        df.at[index, "Verifier_Agent_Runs"] = int(verifier_retries)
        df.at[index, "Question_ID"] = questionID

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df.at[index, 'Valasz'] = answer
        df.at[index, 'Model_Paragrafus'] = parag_string
        df.at[index, 'Runtime'] = elapsed_time
        df.at[index, "Timestamp"] = timestamp

        jogalap = "".join(df.at[index, "Jogalap"].replace(" ", "").lower().replace("btk",""))
        parag = "".join(df.at[index, "Model_Paragrafus"].replace(" ", "").lower().replace("btk",""))

        print(jogalap)
        print(parag)

        if jogalap in parag:
            df.at[index, "Helyesseg"] = 1

        df.to_csv(file_path, index=False, encoding="utf-8-sig")

def eval():
    vc = df["Helyesseg"].value_counts()
    hossz = len(df["Helyesseg"])
    good = vc.get(0,0)
    acc = good/hossz
    s1 = round(vc.get(1, 0)/hossz*100,2)
    s2 = round(vc.get(2, 0)/hossz*100,2)
    s3 = round(vc.get(3, 0)/hossz*100,2)
    s4 = round(vc.get(4, 0)/hossz*100,2)
    s5 = round(vc.get(5, 0)/hossz*100,2)
    print(f"Hossz: {hossz}")
    print(f"Accuracy: {round(acc,2)*100}%")
    print(f"Messzi találat: {round(vc.get(-1, 0)/hossz*100,2)}%")
    print(f"1-es rádiuszban: {s1}%")
    print(f"2-es rádiuszban: {s2}%")
    print(f"3-as rádiuszban: {s3}%")
    print(f"4-es rádiuszban: {s4}%")
    print(f"5-ös rádiuszban: {s5}%")
    print(f"Accuracy ha a keresési rádiusz kibővítve 4-re {round(s1+s2+s3+s4+acc*100)}%")
    print()
    print(vc)

# Helyesség oszlopban: -1=nagyon messze, -2=hiba
eval()
#scraping()
#print("Scraping finished!")
#run_model()