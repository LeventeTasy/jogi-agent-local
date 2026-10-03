from clm import CLMClient, Choice, Noul, Score

client = CLMClient()  # http://127.0.0.1:8700 by default
r = client.system_one(
    state="Customer: my invoice was charged twice and nobody answers the phone!",
    questions={
        "urgency": Noul(instructions="Is this urgent?"),
        "department": Choice(instructions="Which team should handle this?",
                             criteria={"billing": "Charges, invoices, refunds",
                                       "technical": "Bugs and outages"}),
        "frustration": Score(instructions="How frustrated is the customer?",
                             criteria=["Calm", "Frustrated", "Very angry"]),
    },
)
print(r.answers["urgency"].choice)         # billing
print(r.answers["urgency"].probabilities)  # {'billing': 0.93878, 'technical': 0.06122}

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

# Qwen3-8B indítása vLLM-mel embedding módban
./vllm-env/bin/vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --task embed \
  --dtype half \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8090"""