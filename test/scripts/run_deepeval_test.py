import os
import time
import json
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

from deepeval.models import GeminiModel, AzureOpenAIModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    SummarizationMetric,
    ToxicityMetric,
    BiasMetric,
    GEval
)

from clm import CLMClient, Noul, Score

load_dotenv()

# EVALUATION MODEL
USE_CLM = True

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

CLM_BASE_URL = os.getenv("CLM_BASE_URL", "http://127.0.0.1:8700")
CLM_MODEL = "clm-latest"

clm_client = None

if USE_CLM:
    clm_client = CLMClient(base_url=CLM_BASE_URL, model=CLM_MODEL)
    if not clm_client.health():
        raise RuntimeError(f"CLM szerver nem érhető el: {CLM_BASE_URL}")
    print(f"CLM használata: {CLM_BASE_URL}")
    print(f"CLM modell: {CLM_MODEL}")


# CLM EVALUATION
def evaluate_with_clm(input_text, actual_output, retrieval_context):
    if clm_client is None:
        raise RuntimeError("CLM kliens nincs inicializálva.")

    if retrieval_context is None:
        retrieval_context = []

    retrieval_context = [str(chunk) for chunk in retrieval_context if chunk is not None and str(chunk).strip() != ""]
    context_text = "\n\n".join(f"[Retrieved context {i + 1}]\n{chunk}" for i, chunk in enumerate(retrieval_context))

    if not context_text:
        context_text = "[No retrieved context available]"

    state = f"""
        QUESTION:
        {input_text}
        
        RETRIEVED CONTEXT:
        {context_text}
        
        ANSWER:
        {actual_output}
        """.strip()

    response = clm_client.system_one(
        state=state,
        questions={
            "faithfulness": Score(
                instructions="How well is the ANSWER supported by the RETRIEVED CONTEXT?",
                criteria=["Not supported", "Fully supported"]
            ),
            "answer_relevancy": Score(
                instructions="How well does the ANSWER address the QUESTION?",
                criteria=["Not relevant", "Fully relevant"]
            ),
            "context_relevancy": Score(
                instructions="How relevant is the RETRIEVED CONTEXT to the QUESTION?",
                criteria=["Not relevant", "Fully relevant"]
            )
        }
    )
    faithfulness_score = float(response.answers["faithfulness"].probabilities["1"])
    answer_relevancy_score = float(response.answers["answer_relevancy"].probabilities["1"])
    context_relevancy_score = float(response.answers["context_relevancy"].probabilities["1"])

    faithfulness_reason = f"CLM probability of full faithfulness: {faithfulness_score:.4f}"
    answer_relevancy_reason = f"CLM probability of answer relevance: {answer_relevancy_score:.4f}"
    context_relevancy_reason = f"CLM probability of context relevance: {context_relevancy_score:.4f}"

    print(f"Faithfulness: {faithfulness_score}")
    print(faithfulness_reason)
    print(f"Answer_Relevancy: {answer_relevancy_score}")
    print(answer_relevancy_reason)
    print(f"Context_Relevancy: {context_relevancy_score}")
    print(context_relevancy_reason)

    if response.latency_ms is not None:
        print(f"CLM latency: {response.latency_ms:.2f} ms")
    else:
        print("CLM latency: N/A")

    return {
        "Faithfulness": faithfulness_score,
        "Faithfulness_Reason": faithfulness_reason,
        "Answer_Relevancy": answer_relevancy_score,
        "Answer_Relevancy_Reason": answer_relevancy_reason,
        "Context_Relevancy": context_relevancy_score,
        "Context_Relevancy_Reason": context_relevancy_reason
    }


# GPT EVALUATION
def evaluate_with_gpt(input_text, actual_output, retrieval_context):
    model = AzureOpenAIModel(
        model="gpt-5-mini",
        deployment_name="gpt-5-mini",
        api_key=API_KEY,
        api_version="2025-01-01-preview",
        base_url=BASE_URL,
        temperature=0.5
    )

    faithfulness = FaithfulnessMetric(threshold=0.5, model=model)
    answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model=model)
    context_relevancy = ContextualRelevancyMetric(threshold=0.5, model=model)

    try:
        test_case = LLMTestCase(input=input_text, actual_output=actual_output, retrieval_context=retrieval_context)

        metrics = [faithfulness, answer_relevancy, context_relevancy]
        metric_names = ["Faithfulness", "Answer_Relevancy", "Context_Relevancy"]

        for metric, metric_name in zip(metrics, metric_names):
            metric.measure(test_case)

            try:
                print(metric_name + ": " + str(metric.score))
                print(metric_name + ": " + str(metric.reason))
            except Exception:
                print("Hiba a kiiratásnál!")

    except TypeError:
        test_case = LLMTestCase(input=input_text, actual_output=actual_output, retrieval_context=None)

        metrics = [answer_relevancy]
        metric_names = ["Answer_Relevancy"]

        for metric, metric_name in zip(metrics, metric_names):
            metric.measure(test_case)

            try:
                print(metric_name + ": " + str(metric.score))
                print(metric_name + ": " + str(metric.reason))
            except Exception:
                print("Hiba a kiiratásnál!")

    return {
        "Faithfulness": faithfulness.score,
        "Faithfulness_Reason": faithfulness.reason,
        "Answer_Relevancy": answer_relevancy.score,
        "Answer_Relevancy_Reason": answer_relevancy.reason,
        "Context_Relevancy": context_relevancy.score,
        "Context_Relevancy_Reason": context_relevancy.reason
    }


# EVALUATION
def evaluate_multi_agent_system(input_text, actual_output, retrieval_context):
    if USE_CLM:
        return evaluate_with_clm(input_text, actual_output, retrieval_context)
    return evaluate_with_gpt(input_text, actual_output, retrieval_context)


AGENT_COLUMNS = [
    "Torveny", "Tipus", "Kerdes", "Q_chunk", "A_chunk", "Valasz",
    "Faithfulness", "Faithfulness_Reason",
    "Answer_Relevancy", "Answer_Relevancy_Reason",
    "Context_Relevancy", "Context_Relevancy_Reason",
    "Verifier_Agent_Runs", "Runtime", "Question_ID", "Timestamp",
    "Total_Tokens", "Prompt_Tokens", "Completion_Tokens", "Successful_Requests"
]

RAG_COLUMNS = [
    "Torveny", "Tipus", "Kerdes", "Q_chunk", "A_chunk", "Valasz",
    "Faithfulness", "Faithfulness_Reason",
    "Answer_Relevancy", "Answer_Relevancy_Reason",
    "Context_Relevancy", "Context_Relevancy_Reason", "Runtime"
]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    PATH = BASE_DIR.parent / "results" / "answered_questions_clm_eval.csv"

    limit = 25
    ind = 0

    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=object)

        for index, row in df.iterrows():
            if ind >= limit:
                break

            jelenlegi_valasz = row["Valasz"]
            jelenlegi_achunk = row["A_chunk"]
            faithfulness_reason = row["Faithfulness_Reason"]

            valasz_ures = pd.isna(jelenlegi_valasz) or str(jelenlegi_valasz).strip() == ""
            achunk_ures = pd.isna(jelenlegi_achunk) or str(jelenlegi_achunk).strip() == ""
            faithfulness_reason_ures = pd.isna(faithfulness_reason) or str(faithfulness_reason).strip() == ""

            if valasz_ures and achunk_ures:
                print(f"Sor [{index}] még nincs megválaszolva, kihagyás.")
                continue

            if not faithfulness_reason_ures:
                print(f"Sor [{index}] már ki van töltve, kihagyás.")
                continue

            input_q = row["Kerdes"]
            agent_output = row["Valasz"]
            retrieved_rag_chunks = [row["A_chunk"]]

            print("-" * 30)
            print(f"{index}: Kérdés feldolgozása: {input_q}")
            print(f"Evaluation model: {'CLM-v0.1-8B' if USE_CLM else 'GPT-5-mini'}")

            eredmenyek = evaluate_multi_agent_system(input_q, agent_output, retrieved_rag_chunks)

            for key in eredmenyek.keys():
                df.at[index, key] = eredmenyek[key]

            #df.to_csv(PATH, index=False, encoding="utf-8-sig")

            print("-" * 30)
            print("\n\n")

            ind += 1

    else:
        raise FileNotFoundError(PATH)