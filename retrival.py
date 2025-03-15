import json
import logging
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from torch import tensor
import faiss
from database import load_embeddings_from_json
from langchain_ollama import ChatOllama

class Retriever:
    """
    Enhanced Retriever for German legal documents with BGB-specific optimizations.
    """
    def __init__(self, data_path='data/embeddings.json', llm_model='llama3.1',
                 encoder_model='intfloat/multilingual-e5-large', device='mps'):
        self.model = SentenceTransformer(encoder_model, device=device)
        self.embeddings, self.data = load_embeddings_from_json(data_path)
        self.device = device
        self.llm = ChatOllama(model=llm_model)

        self.bgb_paragraph_pattern = re.compile(r'§\s*\d+\s*[a-zA-Z]*\s*BGB', re.IGNORECASE)

        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.bgb_index = {}
        for idx, doc in enumerate(self.data):
            for ref in doc.get('bgb_references', []):
                self.bgb_index.setdefault(ref, []).append(idx)

    def _extract_bgb_references(self, text: str) -> list:
        """Extract explicit BGB paragraph references from text."""
        return self.bgb_paragraph_pattern.findall(text)

    def optimize_legal_query(self, query: str) -> str:
        """
        Optimize the legal query by ensuring key legal concepts and BGB references are included.
        """
        optimization_prompt = """
        Sie sind ein juristischer Expertensystem für deutsches Zivilrecht. Optimieren Sie die Anfrage für die Suche im BGB-Kommentar:
        1. Identifizieren Sie Kernbegriffe: Rechtsgebiet, spezifische Paragraphen, Schlüsselkonzepte.
        2. Präzisieren Sie unklare Formulierungen.
        3. Fügen Sie implizite BGB-Bezüge hinzu, wo angemessen.
        4. Erhalten Sie die ursprüngliche Intention.
        
        Ausgabeformat: [Optimierte Suchanfrage]
        
        Beispiele:
        Original: "Was passiert wenn der Vermieter die Heizung nicht repariert?"
        Optimiert: "Pflichten des Vermieters zur Instandhaltung der Mietssache gemäß § 535 Abs. 1 BGB bei defekter Heizungsanlage"
        
        Original: "Kann ich vom Kaufvertrag zurücktreten?"
        Optimiert: "Widerrufsrecht und Rücktrittsmöglichkeiten nach § 323 BGB bei Kaufverträgen"
        """
        messages = [
            ('system', optimization_prompt),
            ('user', f"Originalanfrage: {query}"),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip("[]")

    def retrieve_documents(self, query: str, top_k: int = 25) -> list:
        """
        Retrieve candidate documents by combining direct BGB reference lookups with FAISS-based semantic search.
        """
        print("Processing legal query...")
        optimized_query = self.optimize_legal_query(query)

        bgb_paragraphs = self._extract_bgb_references(optimized_query)
        direct_matches = []
        if bgb_paragraphs:
            print(f"Detected BGB references: {bgb_paragraphs}")
            for ref in bgb_paragraphs:
                direct_matches.extend(self.bgb_index.get(ref, []))

        query_embedding = self.model.encode(optimized_query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        distances, semantic_indices = self.index.search(query_embedding, top_k)
        semantic_indices = semantic_indices.flatten().tolist()

        combined = list(set(direct_matches + semantic_indices))
        return combined

    def legal_rerank_prompt(self, query: str) -> str:
        """
        Generate a prompt to instruct the LLM to rerank documents based on legal relevance.
        """
        return f"""
        Sie sind ein Rechtsreferendar, der juristische Texte bewertet. Bewerten Sie die Relevanz für diese Anfrage:
        Anfrage: {query}
        
        Bewertungskriterien:
        1. Explizite Erwähnung von BGB-Paragraphen.
        2. Bezug zum Rechtsgebiet der Anfrage.
        3. Behandlung vergleichbarer Fallkonstellationen.
        4. Aktualität der Rechtsprechungsreferenzen.
        
        Bewertungsskala:
        1: Höchstrelevant (direkte Beantwortung)
        5: Teilrelevant (allgemeine Informationen)
        10: Irrelevant
        
        Bitte geben Sie **nur** das JSON-Objekt zurück (ohne Kommentare oder zusätzlichen Text) im folgenden Format:
        {{
          "bewertungen": {{
            "Text 1": <Note>,
            "Text 2": <Note>,
            ...
          }}
        }}
        
        Zu bewertende Texte:
        """

    def rerank_documents(self, query: str, candidates: list, batch_size: int = 3) -> list:
        """
        Rerank candidate documents using an LLM-driven legal relevance assessment.
        """
        print("Reranking retrieved documents...")
        scores = {}
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            prompt = self.legal_rerank_prompt(query)
            messages = [
                ('system', prompt),
                ('user', "\n".join(
                    [f"Text {j+1}: {self.data[idx]}" for j, idx in enumerate(batch)]
                ))
            ]
            response = self.llm.invoke(messages)
            try:
                response_json = json.loads(response.content)
                batch_scores = response_json.get('bewertungen', {})
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON response: {response.content}")
                batch_scores = {f"Text {j+1}": 10 for j in range(len(batch))}
            for text_ref, score in batch_scores.items():
                try:
                    idx = batch[int(text_ref.split()[-1]) - 1]
                    scores[idx] = min(max(int(score), 1), 10)  # Clamp score between 1 and 10
                except (ValueError, IndexError):
                    continue
        return sorted(scores.items(), key=lambda x: x[1])

    def get_results(self, user_query: str, top_k: int = 5) -> list:
        """
        Full retrieval pipeline that first finds candidate documents then reranks them.
        """
        candidate_indices = self.retrieve_documents(user_query)
        if len(candidate_indices) > 5:
            ranked = self.rerank_documents(user_query, candidate_indices)
            return [self.data[idx] for idx, _ in ranked[:top_k]]
        return [self.data[idx] for idx in candidate_indices[:top_k]]
