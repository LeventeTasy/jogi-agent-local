import time
from transformers import pipeline

clf = pipeline(
    "zero-shot-classification",
    model="Horizon-Labs/multilingual-zeroshot-base",
    device=0,
    model_kwargs={"attn_implementation": "sdpa"},
)

labels = [
    "a question about law, legal rights or legal obligations",
    "a question unrelated to law",
]

questions = [
    "A munkáltatóm nem fizette ki a béremet. Mit tehetek?",
    "Hogyan írjak egy Python-függvényt?",
]

for question in questions:
    start = time.perf_counter()
    result = clf(
        question,
        candidate_labels=labels,
        hypothesis_template="This is {}.",
    )
    print(f"{time.perf_counter() - start:.2f}s")
    print(result["labels"][0], result["scores"][0])