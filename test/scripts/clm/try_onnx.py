
import time

from jogi_agent.utils import classify_onnx

labels = {
    "legal": "a legal question about law, legal rights or legal obligations",
    "not_legal": "a non-legal question unrelated to law"
}

questions = [
    "A munkáltatóm nem fizette ki a béremet. Mit tehetek?",
    "Hogyan írjak egy Python-függvényt?",
]

for question in questions:
    start = time.perf_counter()
    try:
        label, score, c_scores = classify_onnx(question, labels)
        print(
            f"{time.perf_counter() - start:.3f}s | "
            f"{label} | {score:.4f} | {c_scores}"
            f" | {abs(c_scores['legal']-c_scores['not_legal'])}"
        )
    except Exception:
        raise
