from jogi_agent.utils import ask_clm_llm_choice, ask_llm, ask_clm_choice
from clm import CLMClient

"""print(r.answers["department"].choice)         # billing
print(r.answers["department"].probabilities)  # {'billing': 0.93878, 'technical': 0.06122}"""



"""r  = ask_clm_choice("Vásárló: köszönöm szépen a válaszát, a segítséget köszönöm szépen!",
                    "Which team should handle this?", {"billing": "Charges, invoices, refunds",
                                           "technical": "Bugs and outages"})"""
questions = [
    {"question": "Felmondhat-e a munkáltató azonnali hatállyal súlyos kötelezettségszegés esetén?", "real": "legal"},
    {"question": "Milyen feltételekkel köthető érvényes adásvételi szerződés?", "real": "legal"},
    {"question": "Kell-e kártérítést fizetni véletlenül okozott kár esetén?", "real": "legal"},
    {"question": "Hány év az általános elévülési idő polgári jogi követeléseknél?", "real": "legal"},
    {"question": "Mikor jogosult a bérlő felmondani a lakásbérleti szerződést?", "real": "legal"},
    {"question": "Mikor minősül egy cselekmény jogos védelemnek?", "real": "legal"},
    {"question": "Kötelező-e írásba foglalni egy ingatlan adásvételi szerződését?", "real": "legal"},
    {"question": "Ki felel azért, ha egy kiskorú másnak kárt okoz?", "real": "legal"},
    {"question": "Milyen jogai vannak az örökösnek az örökhagyó halála után?", "real": "legal"},
    {"question": "Mikor lehet egy szerződést bíróság előtt megtámadni?", "real": "legal"},
    {"question": "Milyen feltételekkel lehet próbaidő alatt felmondani?", "real": "legal"},
    {"question": "Mikor valósul meg a lopás bűncselekménye?", "real": "legal"},
    {"question": "Jogosult-e a munkavállaló szabadságra a munkaviszony megszűnése előtt?", "real": "legal"},
    {"question": "Milyen feltételek mellett lehet végrendeletet érvényesen elkészíteni?", "real": "legal"},
    {"question": "Milyen jogkövetkezménye lehet a titoktartási kötelezettség megszegésének?", "real": "legal"},
    {"question": "Felel-e a vállalkozás az alkalmazottja által okozott kárért?", "real": "legal"},
    {"question": "Milyen feltételekkel lehet felbontani egy szerződést szerződésszegés miatt?", "real": "legal"},
    {"question": "Mit jelent az ártatlanság vélelme a büntetőeljárásban?", "real": "legal"},
    {"question": "Mikor követ el valaki sikkasztást?", "real": "legal"},
    {"question": "Követelhet-e a fogyasztó visszatérítést hibás teljesítés esetén?", "real": "legal"},
    {"question": "Milyen feltételekkel lehet adatokat jogszerűen kezelni?", "real": "legal"},
    {"question": "Milyen esetekben rendelhet el a bíróság letartóztatást?", "real": "legal"},
    {"question": "Ki örököl, ha az elhunytnak nincs végrendelete?", "real": "legal"},
    {"question": "Milyen feltételekkel lehet egy munkavállalót túlórára kötelezni?", "real": "legal"},
    {"question": "Milyen jogorvoslati lehetőségei vannak egy sérelmes elsőfokú ítélet esetén?", "real": "legal"},
    {"question": "Hogyan készül a hagyományos magyar gulyásleves?", "real": "not_legal"},
    {"question": "Miért kék az ég nappal?", "real": "not_legal"},
    {"question": "Hány gramm fehérje van 100 gramm csirkemellben?", "real": "not_legal"},
    {"question": "Melyik bolygó van legközelebb a Naphoz?", "real": "not_legal"},
    {"question": "Hogyan lehet Pythonban eltávolítani egy elemet egy listából?", "real": "not_legal"},
    {"question": "Milyen bor illik legjobban egy camembert sajthoz?", "real": "not_legal"},
    {"question": "Mi a különbség a processzor és a videokártya között?", "real": "not_legal"},
    {"question": "Hogyan működik a fotoszintézis?", "real": "not_legal"},
    {"question": "Mennyi idő alatt fő meg egy kemény tojás?", "real": "not_legal"},
    {"question": "Melyik a legnagyobb óceán a Földön?", "real": "not_legal"},
    {"question": "Miért lesz lassabb egy számítógép kevés szabad tárhely esetén?", "real": "not_legal"},
    {"question": "Hogyan lehet C#-ban objektumot létrehozni egy osztályból?", "real": "not_legal"},
    {"question": "Milyen előnyei vannak a rendszeres testmozgásnak?", "real": "not_legal"},
    {"question": "Miért változik a Hold látható alakja?", "real": "not_legal"},
    {"question": "Hogyan működik a ZIP fájlok tömörítése?", "real": "not_legal"},
    {"question": "Melyik francia város híres az Eiffel-toronyról?", "real": "not_legal"},
    {"question": "Hogyan lehet egy pandas DataFrame adott indexű sorát lekérni?", "real": "not_legal"},
    {"question": "Mi okozza a szivárvány kialakulását?", "real": "not_legal"},
    {"question": "Milyen ételeket lehet készíteni édesburgonyából?", "real": "not_legal"},
    {"question": "Mi a különbség a vörösbor és a fehérbor készítése között?", "real": "not_legal"},
    {"question": "Hogyan működik egy neurális hálózat tanítása?", "real": "not_legal"},
    {"question": "Mennyi idő alatt jut el a fény a Naptól a Földig?", "real": "not_legal"},
    {"question": "Hogyan lehet Gitben új branchet létrehozni?", "real": "not_legal"},
    {"question": "Miért rozsdásodik meg a vas?", "real": "not_legal"},
    {"question": "Milyen alapanyagok kellenek egy egyszerű ratatouille-hoz?", "real": "not_legal"}
]

client = CLMClient()

instruction = "Döntsd el, hogy a kérdés jogi vagy nem jogi."
criteria = {
        "legal": "A kérdés jogi.",
        "not_legal": "A kérdés nem jogi."
    }

system_prompt = ("Döntsd el, hogy a kérdés jogi vagy nem jogi."
                 "A kimenet csak 'legal' vagy 'not_legal' lehet")

r = ask_clm_llm_choice("Milyen szabályok alapján működik a sakkban a sáncolás?", instruction, criteria, system_prompt, client)
llm_used = 0
print(r)

for i, q in enumerate(questions):
    ask = ask_clm_llm_choice(q["question"], instruction, criteria, system_prompt, client, 0.35)

    if ask["is_llm"]:
        llm_used += 1
        questions[i]["choice"] = ask["resp"]
    else:
        questions[i]['choice'] = ask["resp"].choice
        questions[i]["confidence"] =ask["resp"].confidence
        questions[i]['probability'] = ask["resp"].probabilities[ask["resp"].choice]


    print(questions[i])

nem_egyezo_db = 0

for i in questions:
    if i['choice'] != i["real"]:
        nem_egyezo_db += 1

print(f"Nem egyezik: {nem_egyezo_db}")
print(f"Used LLM: {llm_used}")

