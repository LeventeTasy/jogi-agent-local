import uuid

from jogi_agent.router import RouterFlow
from jogi_agent.utils import format_history_for_prompt

history : list[dict[str, str]] = []
questions = ["mikor lehet igenybe venni a 25 éven aluliak adokedvezmenyet?", "mikor nem?", "köszönöm"]

chatID = f"T_{uuid.uuid4()}".upper()
for counter, question in enumerate(questions):
        formatted_history = format_history_for_prompt(history)

        try:
            inputs = {
                'topic': question,
                'pdf_text' : "",
                'history': formatted_history,
                'details': "",
                'da_questions': "",
                'da_answers': "",
                'username': "T_1",
                "chatID": chatID,
                "questionNumber": counter
            }

            flow = RouterFlow()
            flow.state["inputs"] = inputs
            flow.state["history"] = history

            resp = str(flow.kickoff())

            if "### RÖVID VÁLASZ" in resp:
                print("jogi válasz")
            else:
                print("nem jogi válasz")

            history = flow.get_history()


        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")