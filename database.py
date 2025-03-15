import json
import logging
import re
from tqdm import tqdm
import numpy as np
from torch import tensor
from sentence_transformers import SentenceTransformer

def add_embeddings_to_json(json_file, output_json_file, batch_size=32, device='mps'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
    bgb_pattern = re.compile(r'§\s*\d+\s*[a-zA-Z]*\s*BGB', re.IGNORECASE)

    for i in tqdm(range(0, len(data), batch_size), desc="Embedding documents"):
        texts = []
        indices = range(i, min(i + batch_size, len(data)))
        for j in indices:
            try:
                with open(data[j]['cleaned_content_path'], 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logging.warning(f"Error reading file {data[j]['cleaned_content_path']}: {e}")
                text = ""
            texts.append(text)
            data[j]['bgb_references'] = bgb_pattern.findall(text)

        embeddings = model.encode(texts, show_progress_bar=False)
        for j, embedding in zip(indices, embeddings):
            data[j]['embedding'] = embedding.tolist()

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_embeddings_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    embeddings = [item.pop('embedding') for item in data]
    return np.array(embeddings).astype('float32'), data

def build_faiss_index(embeddings):
    import faiss
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def get_similar_documents(embeddings, query, top_k=5, device='mps'):
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
    faiss_index = build_faiss_index(embeddings.copy())
    query_embedding = model.encode(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)

    distances, indices = faiss_index.search(query_embedding, top_k)
    return indices.flatten().tolist()

def main():
    json_file = "data/extracted_metadata.json"
    output_json_file = "data/embeddings.json"
    add_embeddings_to_json(json_file, output_json_file)
    embeddings, data = load_embeddings_from_json(output_json_file)
    query = "Wie melde ich Heirat an?"
    top_indices = get_similar_documents(embeddings, query)
    for idx in top_indices:
        print(data[idx]['Title'])

if __name__ == "__main__":
    main()
