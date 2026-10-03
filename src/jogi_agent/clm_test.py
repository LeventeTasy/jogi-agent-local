from clm import CLMClient, Choice, Noul, Score

client = CLMClient()  # http://127.0.0.1:8700 by default
r = client.system_one(
    state="Vásárló: köszönöm szépen a válaszát, a segítséget köszönöm szépen!",
    questions={
        "urgency": Noul(instructions="Is this urgent?"),
        "department": Choice(instructions="Which team should handle this?",
                             criteria={"billing": "Charges, invoices, refunds",
                                       "technical": "Bugs and outages"}),
        "frustration": Score(instructions="Mennyire frusztrált a vásárló?",
                             criteria=["Nyugodt", "Frusztrált", "Nagyon mérges"]),
    },
)
"""print(r.answers["department"].choice)         # billing
print(r.answers["department"].probabilities)  # {'billing': 0.93878, 'technical': 0.06122}"""

for i in r.answers:
    print(f"{i} : {r.answers[i]}")

"""# Projekt gyökérkönyvtár
cd /data_shared/tasyl/jogi-agent-rag

# Qwen3-8B indítása vLLM-mel embedding módban
./vllm-env/bin/vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --task embed \
  --dtype half \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8090

# Projekt gyökérkönyvtár
cd /data_shared/tasyl/jogi-agent-rag

# CLM szerver indítása
# A Qwen3 embedding API-jához csatlakozik a 8090-es porton
./.venv/bin/clm-serve \
  --port 8700 \
  --emb-url http://127.0.0.1:8090/v1/embeddings
"""