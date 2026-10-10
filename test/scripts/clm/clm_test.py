import os.path
from pathlib import Path

from jogi_agent.utils import ask_clm_llm_choice, ask_llm, ask_clm_choice
from clm import CLMClient
import pandas as pd

"""print(r.answers["department"].choice)         # billing
print(r.answers["department"].probabilities)  # {'billing': 0.93878, 'technical': 0.06122}"""

"""r  = ask_clm_choice("Vásárló: köszönöm szépen a válaszát, a segítséget köszönöm szépen!",
                    "Which team should handle this?", {"billing": "Charges, invoices, refunds",
                                           "technical": "Bugs and outages"})"""

BASE_DIR = Path(__file__).resolve().parent
PATH_SAVE = BASE_DIR.parent.parent / "results" / "clm" / "CLM_v0.1-8B-v2.csv"



questions = [

    {"question": "Mikor vehetem igénybe a 25 éven aluliak adókedvezményét?", "real": "legal"},
    {"question": "Felmondhat-e a munkáltatóm azonnali hatállyal próbaidő alatt?", "real": "legal"},
    {"question": "Mit tehetek, ha a munkáltatóm nem fizeti ki a béremet?", "real": "legal"},
    {"question": "Kötelező-e írásba foglalni a munkaszerződést?", "real": "legal"},
    {"question": "Hány nap szabadság jár egy munkavállalónak évente?", "real": "legal"},
    {"question": "Mikor tagadhatja meg a munkáltató a szabadság kiadását?", "real": "legal"},
    {"question": "Felmondhatok-e azonnali hatállyal, ha nem fizetnek ki?", "real": "legal"},
    {"question": "Milyen feltételekkel jár munkanélküli-ellátás Magyarországon?", "real": "legal"},
    {"question": "Mikor évül el egy polgári jogi követelés?", "real": "legal"},
    {"question": "Mit tehetek, ha valaki nem fizeti vissza a kölcsönadott pénzemet?", "real": "legal"},
    {"question": "Érvényes-e egy szóbeli megállapodás szerződésként?", "real": "legal"},
    {"question": "Mikor állhatok el egy online vásárlástól?", "real": "legal"},
    {"question": "Hány napom van elállni egy interneten vásárolt terméktől?", "real": "legal"},
    {"question": "Mit tehetek, ha a webáruház nem akarja visszafizetni a pénzemet?", "real": "legal"},
    {"question": "Ki felel, ha a bérelt lakásban kárt okozok?", "real": "legal"},
    {"question": "Felmondhatja-e a főbérlő azonnal a lakásbérleti szerződést?", "real": "legal"},
    {"question": "Visszatarthatja-e a főbérlő a teljes kauciómat kiköltözéskor?", "real": "legal"},
    {"question": "Milyen jogaim vannak, ha a szomszédom rendszeresen zajong éjszaka?", "real": "legal"},
    {"question": "Készíthetek-e felvételt valakiről az engedélye nélkül?", "real": "legal"},
    {"question": "Közzétehetem-e más ember fényképét a közösségi médiában az engedélye nélkül?", "real": "legal"},
    {"question": "Milyen jogaim vannak, ha valaki engedély nélkül használja a fotómat?", "real": "legal"},
    {"question": "Milyen következménye lehet a rágalmazásnak?", "real": "legal"},
    {"question": "Mi a különbség a rágalmazás és a becsületsértés között?", "real": "legal"},
    {"question": "Mikor köteles valaki kártérítést fizetni egy baleset után?", "real": "legal"},
    {"question": "Ki felel a parkoló autómban okozott kárért?", "real": "legal"},
    {"question": "Mit tehetek, ha hibás terméket vásároltam?", "real": "legal"},
    {"question": "Mi a különbség a jótállás és a kellékszavatosság között?", "real": "legal"},
    {"question": "Mikor tagadhatja meg egy üzlet a termék visszavételét?", "real": "legal"},
    {"question": "Milyen feltételekkel lehet örökölni végrendelet nélkül?", "real": "legal"},
    {"question": "Hogyan lehet érvényesen végrendeletet készíteni?", "real": "legal"},
    {"question": "Milyen jogai vannak a gyermeknek a szülők válása után?", "real": "legal"},
    {"question": "Hogyan határozzák meg a gyermektartásdíj összegét?", "real": "legal"},
    {"question": "Mikor köteles valaki tartásdíjat fizetni a volt házastársának?", "real": "legal"},
    {"question": "Milyen feltételekkel lehet megváltoztatni a gyermek felügyeletéről szóló döntést?", "real": "legal"},
    {"question": "Mit tehetek, ha egy hatóság elutasította a kérelmemet?", "real": "legal"},
    {"question": "Mennyi időn belül lehet fellebbezni egy hatósági határozat ellen?", "real": "legal"},
    {"question": "Milyen jogaim vannak rendőri igazoltatás során?", "real": "legal"},
    {"question": "Mikor jogosult a rendőrség átvizsgálni a táskámat?", "real": "legal"},
    {"question": "Milyen következményei lehetnek a jogosítvány nélküli vezetésnek?", "real": "legal"},
    {"question": "Milyen feltételekkel igényelhetek személyi jövedelemadó-visszatérítést?", "real": "legal"},


    {"question": "Hogyan írjak Pythonban egy függvényt, amely két számot összead?", "real": "not_legal"},
    {"question": "Mi a különbség a lista és a tuple között Pythonban?", "real": "not_legal"},
    {"question": "Hogyan tudom kijavítani a Git merge conflictot?", "real": "not_legal"},
    {"question": "Miért kapok 404-es hibát egy HTTP-kérésnél?", "real": "not_legal"},
    {"question": "Hogyan lehet egy fájlt beolvasni Pythonban?", "real": "not_legal"},
    {"question": "Mi a különbség a mesterséges intelligencia és a gépi tanulás között?", "real": "not_legal"},
    {"question": "Hogyan működik egy neurális hálózat?", "real": "not_legal"},
    {"question": "Miért fogyaszt sok memóriát a Chrome?", "real": "not_legal"},
    {"question": "Hogyan tudom felgyorsítani a Windows rendszerindítását?", "real": "not_legal"},
    {"question": "Milyen algoritmussal érdemes rendezni egy nagy listát?", "real": "not_legal"},
    {"question": "Miért kék az ég?", "real": "not_legal"},
    {"question": "Hogyan keletkezik a szivárvány?", "real": "not_legal"},
    {"question": "Mi a különbség a vírus és a baktérium között?", "real": "not_legal"},
    {"question": "Miért alszunk naponta több órát?", "real": "not_legal"},
    {"question": "Hogyan működik a fotoszintézis?", "real": "not_legal"},
    {"question": "Melyik bolygó van a legközelebb a Naphoz?", "real": "not_legal"},
    {"question": "Hogyan készíthetek otthon palacsintát?", "real": "not_legal"},
    {"question": "Milyen hőmérsékleten érdemes sütni a csirkét?", "real": "not_legal"},
    {"question": "Hogyan lehet megtanulni gyorsabban gépelni?", "real": "not_legal"},
    {"question": "Milyen gyakorlatokkal lehet fejleszteni a memóriát?", "real": "not_legal"},


    {"question": "Hogyan állíthatom be a jogosultságokat egy Google Drive-mappán?", "real": "not_legal"},
    {"question": "Hogyan ellenőrizhetem, hogy egy API-kérés megfelel-e a specifikációnak?", "real": "not_legal"},
    {"question": "Mit jelent a szerződés egy REST API dokumentációjában?", "real": "not_legal"},
    {"question": "Hogyan kezeljem a hozzáférési jogokat egy SQL-adatbázisban?", "real": "not_legal"},
    {"question": "Hogyan lehet visszavonni egy felhasználó adminisztrátori jogosultságát Linuxon?", "real": "not_legal"},
    {"question": "Milyen büntetés jár a Monopolyban, ha rossz helyre lépek?", "real": "not_legal"},
    {"question": "Mikor jár büntetőrúgás a fociban?", "real": "not_legal"},
    {"question": "Milyen szabálytalanságért jár kétperces kiállítás kézilabdában?", "real": "not_legal"},
    {"question": "Mikor kap büntetőpontot egy versenyző a Forma–1-ben?", "real": "not_legal"},
    {"question": "Milyen büntetés jár, ha sakkban szabálytalan lépést teszek?", "real": "not_legal"},
    {"question": "Hogyan számolja ki a játék a karakteremre kiszabott büntetést?", "real": "not_legal"},
    {"question": "Milyen feltételekkel oldható fel egy letiltott fiók egy videojátékban?", "real": "not_legal"},
    {"question": "Hogyan működik a jogosultságkezelés egy Discord-szerveren?", "real": "not_legal"},
    {"question": "Milyen szabályok vonatkoznak egy sakkverseny időmérésére?", "real": "not_legal"},
    {"question": "Mekkora adót kell fizetnem a Monopolyban, ha a mezőre lépek?", "real": "not_legal"},
    {"question": "Hogyan készíthetek szerződésmintát egy objektumorientált programban?", "real": "not_legal"},
    {"question": "Mit jelent az öröklődés a Java osztályaiban?", "real": "not_legal"},
    {"question": "Hogyan kezelhetem a felhasználói szerepköröket egy webalkalmazásban?", "real": "not_legal"},
    {"question": "Mi történik, ha megszegem a játékon belüli küldetés szabályait?", "real": "not_legal"},
    {"question": "Hogyan lehet egy dokumentumon szerkesztési és olvasási engedélyeket beállítani?", "real": "not_legal"},


    {"question": "Szia!", "real": "not_legal"},
    {"question": "Helló, hogy vagy?", "real": "not_legal"},
    {"question": "Jó reggelt!", "real": "not_legal"},
    {"question": "Jó estét kívánok!", "real": "not_legal"},
    {"question": "Köszi szépen!", "real": "not_legal"},
    {"question": "Nagyon hálás vagyok a segítségedért.", "real": "not_legal"},
    {"question": "Viszlát, további szép napot!", "real": "not_legal"},
    {"question": "Mit csinálsz most?", "real": "not_legal"},
    {"question": "Milyen napod volt?", "real": "not_legal"},
    {"question": "Mi újság veled?", "real": "not_legal"},
    {"question": "Tudsz egy kicsit beszélgetni?", "real": "not_legal"},
    {"question": "Unatkozom, mit csináljak?", "real": "not_legal"},
    {"question": "Ajánlanál egy jó filmet estére?", "real": "not_legal"},
    {"question": "Milyen zenét hallgassak tanuláshoz?", "real": "not_legal"},
    {"question": "Szerinted mit vegyek fel ma?", "real": "not_legal"},
    {"question": "Mit főzzek ma vacsorára?", "real": "not_legal"},
    {"question": "Holnap milyen programot szervezzek a barátaimmal?", "real": "not_legal"},
    {"question": "Elmesélhetek neked valamit?", "real": "not_legal"},
    {"question": "Ez nagyon jól hangzik, mesélj még!", "real": "not_legal"},
    {"question": "Bocsi, ezt nem értettem. Elmagyaráznád újra?", "real": "not_legal"},
]


client = CLMClient()
formatted_history = ""
SYSTEM_PROMPT = f'''
        Te egy szigorú, bináris osztályozó modell (Router) vagy egy magyar jogi AI asszisztens rendszer kapujában.
        A feladatod a beérkező felhasználói üzenet szándékának (intent) vizsgálata és besorolása a korábbi beszélgetési kontextus figyelembevételével.

        Beszélgetési előzmény:
        "{formatted_history}"

        SZABÁLYOK:
        1. Két lehetséges kategória létezik:
           - "LEGAL": Ha a felhasználó kérdése vagy üzenete közvetlenül vagy közvetve magyar jogi szabályozással, törvényekkel (pl. Munka Törvénykönyve, Ptk., Btk., GDPR, Adójog, fogyasztóvédelem), hatósági eljárásokkal, szerződésekkel, munkaviszonnyal vagy jogi vitákkal kapcsolatos.
           - "NOT_LEGAL": Ha az üzenet:
             * Csupán üdvözlés, köszönés, elköszönés (pl. "Szia", "Jó napot", "Helló").
             * Megköszönés vagy udvariassági formula (pl. "Köszönöm", "Szuper, köszi", "Hálás vagyok").
             * Nem jogi témájú általános kérdés (pl. receptek, időjárás, matematika, programozás, popkultúra).
             * Értelmetlen szöveg, karakterhalmaz, tesztelés (pl. "asdasd", "123").
             * Általános csevegés vagy a modell képességeire vonatkozó kérdés (pl. "Ki vagy te?", "Hogy vagy?").

        2. BIZTONSÁGI / DÖNTÉSI IRÁNYELV:
           - Ha az üzenet köznapi megfogalmazású, de egyértelműen jogi helyzetre utal (pl. "Kirúgott a főnököm, mit csináljak?", "Átvert az eladó az interneten", "Nem kaptam fizetést"), akkor a besorolás: "LEGAL".
           - Ha bizonytalan vagy, de nincs benne konkrét jogi vagy jogszabályi elem, a válasz: "NOT_LEGAL".

        3. KIMENETI FORMÁTUM (SZIGORÚ):
           Kizárólag a két kulcsszó egyikét válaszold vissza, bármiféle írásjel, magyarázat, bevezető vagy szóköz nélkül, nagy betűvel írva:
           LEGAL
           vagy
           NOT_LEGAL
        '''

"""llm = LLM(model=self.state["model"])
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": self.state["question"]},
]
response = llm.call(messages)
self.state["total_tokens"] = llm.get_token_usage_summary()"""

client = CLMClient()
instruction = f"""Döntsd el, hogy a felhasználó kérdése jogi vagy nem jogi.
                    A beszélgetési előzmény figyelembevételével:
                    {formatted_history}"""

criteria = {
            "legal": "A kérdés jogi.",
            "not_legal": "A kérdés nem jogi."
        }


llm_used = 0

write_header = (not os.path.exists(PATH_SAVE) or os.path.getsize(PATH_SAVE) == 0)

prob = 0.5

probs = [0, 0.2, 0.3, 0.4, 0.5]

for prob_i in probs:
    for i, q in enumerate(questions):
        ask = ask_clm_llm_choice(q["question"],instruction,criteria,SYSTEM_PROMPT,client,prob_i)

        row = {
            "question": q["question"],
            "label": q["real"],
            "is_llm": ask["is_llm"],
            "prediction": None,
            "confidence": None,
            "legal_prob": None,
            "not_legal_prob": None,
            "prob": prob_i,
            "c_legal": criteria["legal"],
            "c_not_legal": criteria["not_legal"]

        }

        if ask["is_llm"]:
            llm_used += 1
            questions[i]["choice"] = ask["resp"]

            row["prediction"] = ask["resp"]

        else:
            resp = ask["resp"]

            questions[i]["choice"] = resp.choice
            questions[i]["confidence"] = resp.confidence
            questions[i]["probability"] = (resp.probabilities[resp.choice])

            row["prediction"] = resp.choice
            row["confidence"] = resp.confidence
            row["legal_prob"] = resp.probabilities["legal"]
            row["not_legal_prob"] = resp.probabilities["not_legal"]

        pd.DataFrame([row]).to_csv(
            PATH_SAVE,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8"
        )

        write_header = False

        print(questions[i])

    nem_egyezo_db = 0

    for i in questions:
        if i['choice'] != i["real"]:
            nem_egyezo_db += 1

    print(f"Nem egyezik: {nem_egyezo_db}")
    print(f"Used LLM: {llm_used}")

