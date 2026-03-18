import json
import os
import faiss
import numpy as np
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "recipe_database.json")

client = OpenAI()

# 加载数据
with open(DB_PATH, "r", encoding="utf-8") as f:
    recipes = json.load(f)

texts = []
metadata = []

for r in recipes:
    text = f"{r['name']} {r.get('clinical_definition','')}"
    texts.append(text)
    metadata.append(r)


# 生成embedding
def embed(text):
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(res.data[0].embedding, dtype="float32")


vectors = np.array([embed(t) for t in texts])

# 创建FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)


def search(query, top_k=3):
    q = embed(query).reshape(1, -1)
    D, I = index.search(q, top_k)

    results = []
    for idx in I[0]:
        results.append(metadata[idx])

    return results
