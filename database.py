import json
import logging
import re
from tqdm import tqdm
import numpy as np
import torch 
from sentence_transformers import SentenceTransformer
import os

def add_embeddings_to_json(json_file, output_json_file, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
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

def extract_metadata_from_documents(path='data/cleaned_bgb'):
    metadata = []
    for file_name in os.listdir(path):
        if "weggefallen" in file_name:
            continue
        input_path = os.path.join(path, file_name)
        if os.path.isfile(input_path):
            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()
            metadata.append({
                'Title': file_name.replace('.md', ''),
                'cleaned_content_path': input_path,
                'page_content': content
            })
    return metadata

def main():
    json_file = "data/extracted_metadata.json"
    if not os.path.exists(json_file):
        metadata = extract_metadata_from_documents()
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    else:
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    # Check if the embeddings already exist
    if os.path.exists("data/embeddings.json"):
        with open("data/embeddings.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        if len(data) == len(metadata):
            print("Embeddings already exist.")
            return
    else:
        print("Embeddings do not exist. Generating new embeddings.")

    output_json_file = "data/embeddings.json"
    add_embeddings_to_json(json_file, output_json_file)
    embeddings, data = load_embeddings_from_json(output_json_file)
    query = "Wie melde ich Heirat an?"
    top_indices = get_similar_documents(embeddings, query)
    for idx in top_indices:
        print(data[idx]['Title'])

if __name__ == "__main__":
    main()
