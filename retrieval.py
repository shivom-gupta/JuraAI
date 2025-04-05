import json
import logging
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import fuzz
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from database import load_embeddings_from_json
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Retriever:
    """
    Enhanced Retriever for German BGB legal documents implementing a hybrid
    retrieval (semantic + BGB reference lookup) and LLM-based reranking strategy.
    """
    def __init__(self,
                 llm: str|BaseChatModel = 'llama3.1',
                 encoder_model: Union[SentenceTransformer, str] = 'intfloat/multilingual-e5-large',
                 data_path: str = 'data/embeddings.json',
                 device: str = 'mps',
                 retrieval_top_k: int = 25,
                 rerank_batch_size: int = 5,
                 rerank_trigger_threshold: int = 10, # Rerank if more candidates than this
                 fuzzy_match_threshold: int = 85, # Stricter threshold for matching
                 ):
        """
        Initializes the LegalRetrieverBGB.

        Args:
            llm: An instance of a Langchain compatible Chat Model (e.g., ChatOllama).
            encoder_model: A SentenceTransformer model instance or the name of a model
                           to load (e.g., 'intfloat/multilingual-e5-large').
            data_path: Path to the JSON file containing pre-computed embeddings and data.
            device: The device to run the encoder model on ('cpu', 'cuda', 'mps').
            retrieval_top_k: The number of initial candidates to retrieve.
            rerank_batch_size: Number of documents to send to the LLM for reranking in one batch.
            rerank_trigger_threshold: Only run LLM reranking if the number of initial
                                      candidates exceeds this threshold.
            fuzzy_match_threshold: Minimum score (0-100) for fuzzy matching BGB references.
        """
        logging.info("Initializing Retriever...")
        self.device = device
        self.llm = ChatOllama(model=llm, device=device) if isinstance(llm, str) else llm
        if not isinstance(self.llm, BaseChatModel):
            raise ValueError("The provided LLM must be an instance of a Langchain compatible Chat Model.")
        self.retrieval_top_k = retrieval_top_k
        self.rerank_batch_size = rerank_batch_size
        self.rerank_trigger_threshold = rerank_trigger_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold

        if isinstance(encoder_model, str):
            logging.info(f"Loading SentenceTransformer model: {encoder_model}")
            self.encoder_model = SentenceTransformer(encoder_model, device=self.device)
        else:
            logging.info("Using pre-loaded SentenceTransformer model.")
            self.encoder_model = encoder_model.to(self.device)

        self.embeddings, self.data = load_embeddings_from_json(data_path)
        if self.embeddings.shape[0] != len(self.data):
            raise ValueError("Mismatch between number of embeddings and data entries.")

        self.bgb_paragraph_pattern = re.compile(r'§\s*(\d+\s*[a-zA-Z]?(?:\s*Abs\s*\.\s*\d+)?)\s*BGB', re.IGNORECASE)

        logging.info("Setting up FAISS index...")
        embeddings_normalized = self.embeddings.copy()
        faiss.normalize_L2(embeddings_normalized) # Normalize for cosine similarity using IndexFlatIP
        self.index = faiss.IndexFlatIP(embeddings_normalized.shape[1])
        self.index.add(embeddings_normalized)
        logging.info(f"FAISS index created with {self.index.ntotal} vectors.")

        logging.info("Building BGB reference index...")
        self.bgb_index: Dict[str, List[int]] = {}
        self.known_bgb_refs: set[str] = set()
        for idx, doc in enumerate(self.data):
            references = doc.get('bgb_references', [])
            if not isinstance(references, list):
                 logging.warning(f"Document index {idx} has invalid 'bgb_references' type: {type(references)}. Skipping.")
                 continue

            for ref in references:
                if isinstance(ref, str):
                    normalized_ref = self._normalize_bgb_ref(ref)
                    if normalized_ref: # Ensure normalization was successful
                        self.bgb_index.setdefault(normalized_ref, []).append(idx)
                        self.known_bgb_refs.add(normalized_ref)
                else:
                    logging.warning(f"Non-string BGB reference '{ref}' found in document index {idx}. Skipping.")
        logging.info(f"BGB reference index built with {len(self.bgb_index)} unique normalized references.")


    def _normalize_bgb_ref(self, ref: str) -> Optional[str]:
        """Normalizes a BGB reference string for consistent matching."""
        ref = ref.strip().upper()
        ref = re.sub(r'\s+', ' ', ref)
        ref = re.sub(r'§\s+', '§', ref)
        if re.match(r'§\d+', ref):
             return ref
        return None

    def _find_closest_bgb_ref(self, extracted_ref_normalized: str) -> Optional[str]:
        """Finds the best matching known BGB reference using fuzzy matching."""
        best_match, best_score = None, 0
        if extracted_ref_normalized in self.known_bgb_refs:
            return extracted_ref_normalized

        for known_ref in self.known_bgb_refs:
            score = fuzz.ratio(extracted_ref_normalized, known_ref)
            if score > best_score:
                best_match, best_score = known_ref, score

        if best_score >= self.fuzzy_match_threshold:
             logging.info(f"Fuzzy matched '{extracted_ref_normalized}' to known ref '{best_match}' with score {best_score}")
             return best_match
        else:
             logging.debug(f"No suitable fuzzy match found for '{extracted_ref_normalized}' (best score: {best_score} < {self.fuzzy_match_threshold})")
             return None

    def optimize_legal_query(self, query: str) -> str:
        """
        Optimizes the user query using the LLM for better legal search term precision,
        *without* adding BGB references to prevent hallucination.
        """
        logging.info(f"Optimizing query (strict no-BGB-addition mode): {query}")

        optimization_prompt = """
        You are a legal expert system specializing in German Civil Law (BGB). Your task is to refine a user's query to make it more precise for searching legal documents, focusing *only* on clarifying the language and legal concepts based strictly on the input.
        **Allowed Modifications**:
        - Clarify ambiguous terms using standard legal vocabulary based *only* on the input query's content.
        - Focus the query on the key legal action or question *without adding new substantive information or legal codes*.
        - Rephrase for conciseness and clarity using precise legal terminology where appropriate based on the input.
        - Maintain the exact original search intent and scope.

        **Not Allowed**:
        - **Critically: You MUST NOT add any BGB paragraph numbers (§ XXX BGB) or any other legal statute references that were not explicitly present in the original user query.** Your role is strictly refinement of the existing text, not augmentation with specific legal codes you infer.
        - Fundamentally changing the meaning or scope of the request.
        - Adding facts, details, or legal scenarios not mentioned in the original query.

        **Output Format**: Return *only* the optimized search query, enclosed in square brackets. Example: [Optimized Query Text Here]

        **Examples**:
        Original: "What happens if the landlord doesn't fix the heating?"
        Optimized: [Pflichten des Vermieters zur Instandhaltung der Mietsache bei defekter Heizungsanlage] 

        Original: "Can I cancel the purchase contract?"
        Optimized: [Möglichkeiten zum Widerruf oder Rücktritt von einem Kaufvertrag]

        Original: "Neighbor's tree blocks sunlight"
        Optimized: [Rechtliche Ansprüche bei Beeinträchtigung durch Schattenwurf von Nachbarbaum]
        """
        messages = [
            SystemMessage(content=optimization_prompt),
            HumanMessage(content=f"Original query: {query}"),
        ]
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            match = re.search(r'\[(.*)\]', content, re.DOTALL)
            if match:
                optimized_query = match.group(1).strip()
                if self.bgb_paragraph_pattern.search(optimized_query):
                     logging.warning(f"Optimized query unexpectedly contains BGB reference despite instructions: {optimized_query}")
                else:
                     logging.info(f"Optimized query received (no BGB added): {optimized_query}")
                return optimized_query
            else:
                logging.warning(f"LLM response for query optimization did not contain expected brackets: {content}. Using original query.")
                return query

        except Exception as e:
            logging.error(f"LLM invocation failed during query optimization: {e}")
            return query

    def retrieve_documents(self, query: str) -> List[int]:
        """
        Retrieves candidate document indices using a hybrid approach:
        1. Extracts BGB references from the optimized query.
        2. Looks up documents containing those exact (or fuzzy-matched) references.
        3. Performs semantic search using the optimized query embedding via FAISS.
        4. Combines and deduplicates results.

        Args:
            query: The original user query.

        Returns:
            A list of unique document indices sorted primarily by BGB match source,
            then potentially by semantic score (though order isn't guaranteed after set).
        """
        st.spinner("Starting document retrieval process...")
        optimized_query = self.optimize_legal_query(query)

        direct_matches_indices = set()
        extracted_refs_raw = self.bgb_paragraph_pattern.findall(optimized_query)

        if extracted_refs_raw:
            logging.info(f"Found potential BGB references in optimized query: {extracted_refs_raw}")
            for raw_ref_text in extracted_refs_raw:
                full_ref_text = f"§ {raw_ref_text} BGB"
                normalized_extracted_ref = self._normalize_bgb_ref(full_ref_text)

                if normalized_extracted_ref:
                    closest_known_ref = self._find_closest_bgb_ref(normalized_extracted_ref)
                    if closest_known_ref:
                        indices = self.bgb_index.get(closest_known_ref, [])
                        if indices:
                            logging.info(f"Found {len(indices)} direct matches for matched BGB ref: '{closest_known_ref}' (from extracted '{full_ref_text}')")
                            direct_matches_indices.update(indices)
                        else:
                             logging.warning(f"BGB index key '{closest_known_ref}' exists but has no associated indices!")

        st.spinner(f"Found {len(direct_matches_indices)} unique document indices via BGB reference matching.")

        st.spinner("Performing semantic search with FAISS...")
        query_embedding = self.encoder_model.encode(optimized_query, convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        distances, semantic_indices = self.index.search(query_embedding, self.retrieval_top_k)
        semantic_indices = semantic_indices.flatten().tolist()
        semantic_indices = [idx for idx in semantic_indices if idx != -1]
        st.spinner(f"Semantic search retrieved {len(semantic_indices)} indices.")

        combined_indices = list(direct_matches_indices.union(set(semantic_indices)))
        st.spinner(f"Total unique candidate indices after combination: {len(combined_indices)}")
        return combined_indices


    def _legal_rerank_prompt_template(self, query, texts) -> str:
        """Generates the system prompt template for the LLM reranker."""
        return f"""
        You are a highly meticulous German legal assistant (Rechtsreferendar) tasked with evaluating the legal relevance of exactly the provided text snippets in relation to the user query. Your goal is to assign a relevance score to each text snippet.

        **User Query**: {query}

        **Evaluation Criteria** (Score based on how well each text addresses the query):
        1. **Direct BGB Reference Match**: Does the text explicitly mention and discuss BGB paragraphs that are highly relevant to the query? (Highest Importance)
        2. **Legal Topic Relevance**: Does the text cover the core legal issue raised in the query?
        3. **Case Similarity**: Does the text describe legal principles or case outcomes applicable to the query?
        4. **Actuality**: Does the text reference current law or recent, relevant case law (Rechtsprechung)? Outdated information is less relevant.

        **Scoring Scale** (Lower is better):
        - 1-2: Highly Relevant (Directly addresses the core legal question with specific BGB references or principles).
        - 3-5: Relevant (Covers the legal area or related principles, providing useful context).
        - 6-8: Partially Relevant (Mentions the topic vaguely or discusses only tangentially related BGB sections).
        - 9-10: Irrelevant (Completely off-topic or outdated).

        **Output Format**:  
        Return **only** a single JSON object with exactly one key, `"bewertungen"`, whose value is another object that maps each provided text identifier to its integer score. Do not include any additional keys, explanations, or markdown formatting. Evaluate **only** the texts provided below.

        Example JSON Output (if exactly three texts are provided):
        ```json
        {{
        "bewertungen": {{
            "Text 1": 2,
            "Text 2": 5,
            "Text 3": 1
        }}
        }}

        **Text Snippets for Evaluation**:
        {texts}
        """

    def _parse_rerank_response(self, response_content: str, batch_indices: List[int]) -> Dict[int, float]:
        """Parses the LLM's JSON response for reranking scores."""
        scores: Dict[int, float] = {}
        try:
            json_match = re.search(r'```json\s*(\{.*\})\s*```|\{.*\}', response_content, re.DOTALL)
            if not json_match:
                 logging.warning(f"Could not find JSON object in LLM rerank response: {response_content[:500]}...")
                 return {}

            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            response_json = json.loads(json_str)

            batch_scores = response_json.get('bewertungen', {})
            if not isinstance(batch_scores, dict):
                 logging.warning(f"LLM rerank response 'bewertungen' is not a dictionary: {batch_scores}")
                 return {}

            logging.debug(f"Raw batch scores received: {batch_scores}")

            for text_key, score_value in batch_scores.items():
                match = re.match(r'Text\s+(\d+)', text_key)
                if not match:
                    logging.warning(f"Could not parse text identifier '{text_key}' in rerank response.")
                    continue

                try:
                    batch_offset = int(match.group(1)) - 1
                    if 0 <= batch_offset < len(batch_indices):
                        original_doc_idx = batch_indices[batch_offset]

                        score = score_value
                        if isinstance(score_value, dict):
                            score = score_value.get('score', score_value.get('Score', 10.0))

                        try:
                            score_float = float(score)
                            clamped_score = min(max(score_float, 1.0), 10.0)
                            scores[original_doc_idx] = clamped_score
                            logging.debug(f"Assigned score {clamped_score} to doc index {original_doc_idx} (from key {text_key})")
                        except (ValueError, TypeError):
                             logging.warning(f"Invalid score value '{score}' for {text_key}. Assigning default score 10.")
                             scores[original_doc_idx] = 10.0 # Assign default bad score
                    else:
                        logging.warning(f"Text identifier '{text_key}' index out of range for current batch.")

                except IndexError:
                     logging.warning(f"Error mapping {text_key} to batch index.")
                except Exception as e:
                     logging.warning(f"Unexpected error processing score for {text_key}: {e}")


        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON in LLM rerank response: {e}. Response snippet: {response_content[:500]}...")
        except Exception as e:
            logging.error(f"Unexpected error parsing rerank response: {e}")

        return scores


    def rerank_documents(self, query: str, candidate_indices: List[int]) -> List[Tuple[int, float]]:
        """
        Reranks the candidate documents using the LLM based on legal relevance to the query.

        Args:
            query: The original user query.
            candidate_indices: List of document indices retrieved initially.

        Returns:
            A list of tuples `(doc_index, score)`, sorted by score (ascending, lower is better).
        """
        logging.info(f"Reranking {len(candidate_indices)} candidate documents...")
        if not candidate_indices:
            return []

        all_scores: Dict[int, float] = {}

        for i in range(0, len(candidate_indices), self.rerank_batch_size):
            batch_indices = candidate_indices[i : i + self.rerank_batch_size]
            logging.info(f"Reranking batch {i // self.rerank_batch_size + 1} ({len(batch_indices)} documents)...")

            # Prepare texts for the prompt
            texts_for_prompt = []
            for j, idx in enumerate(batch_indices):
                doc_path = self.data[idx]['cleaned_content_path']
                doc_text = open(doc_path, 'r', encoding='utf-8').read()
                max_len = 1000
                snippet = doc_text[:max_len] + ('...' if len(doc_text) > max_len else '')
                texts_for_prompt.append(f"Text {j+1}:\n {snippet}")
            
            formatted_texts = "\n\n".join(texts_for_prompt)

            batch_prompt = self._legal_rerank_prompt_template(query=query, texts=formatted_texts)

            messages = [
                SystemMessage(content="You are evaluating legal text relevance according to the following instructions."), # Context setting
                HumanMessage(content=batch_prompt)
            ]

            try:
                response = self.llm.invoke(messages)
                batch_scores = self._parse_rerank_response(response.content, batch_indices)
                all_scores.update(batch_scores)

                missing_indices = set(batch_indices) - set(batch_scores.keys())
                if missing_indices:
                    logging.warning(f"LLM did not return scores for {len(missing_indices)} documents in the batch: {missing_indices}. Assigning default score 10.")
                    for missing_idx in missing_indices:
                         all_scores[missing_idx] = 10.0

            except Exception as e:
                logging.error(f"LLM invocation or parsing failed for reranking batch starting at index {i}: {e}")
                for idx in batch_indices:
                    if idx not in all_scores:
                         all_scores[idx] = 10.0
        for idx in candidate_indices:
            if idx not in all_scores:
                logging.warning(f"Candidate index {idx} was not scored during reranking. Assigning default score 10.")
                all_scores[idx] = 10.0

        sorted_results = sorted(all_scores.items(), key=lambda item: item[1])
        logging.info(f"Reranking complete. Returning {len(sorted_results)} scored documents.")
        return sorted_results


    def get_results(self, user_query: str, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes the full retrieval and reranking pipeline.

        Args:
            user_query: The original query from the user.
            top_n: The final number of documents to return after reranking.
                   Defaults to `self.retrieval_top_k` if not specified.

        Returns:
            A list of the top N relevant document data dictionaries,
            sorted by relevance (after potential reranking).
        """
        if top_n is None:
            top_n = self.retrieval_top_k

        logging.info(f"--- Starting full retrieval for query: '{user_query}' ---")

        candidate_indices = self.retrieve_documents(user_query)

        if not candidate_indices:
            logging.info("No candidate documents found.")
            return []

        final_indices: List[int]
        if len(candidate_indices) > self.rerank_trigger_threshold:
            logging.info(f"Number of candidates ({len(candidate_indices)}) exceeds threshold ({self.rerank_trigger_threshold}). Proceeding with LLM reranking.")
            ranked_results = self.rerank_documents(user_query, candidate_indices)
            final_indices = [idx for idx, score in ranked_results]
        else:
            logging.info(f"Number of candidates ({len(candidate_indices)}) is below threshold ({self.rerank_trigger_threshold}). Skipping LLM reranking.")

            final_indices = candidate_indices
        top_indices = final_indices[:top_n]

        results_data = [self.data[idx] for idx in top_indices]

        logging.info(f"--- Retrieval complete. Returning {len(results_data)} final documents. ---")
        return results_data

if __name__ == '__main__':

    DATA_FILE = 'data/embeddings.json'
    LLM_MODEL_NAME = 'llama3.1'
    ENCODER_MODEL_NAME = 'intfloat/multilingual-e5-large'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    try:
        llm = ChatOllama(model=LLM_MODEL_NAME, device=DEVICE)

        print("Initializing Retriever...")
        retriever = Retriever(
            llm=llm,
            encoder_model=ENCODER_MODEL_NAME,
            data_path=DATA_FILE,
            device=DEVICE,
            retrieval_top_k=20,
            rerank_trigger_threshold=8,
            rerank_batch_size=4
        )
        print("Retriever initialized successfully.")
        user_query = "Can I cancel the purchase contract if the product is defected?"

        print(f"\nExecuting query: '{user_query}'")
        final_results = retriever.get_results(user_query, top_n=5)

        print("\n--- Top 5 Results ---")
        if final_results:
            for i, doc in enumerate(final_results):
                 print(f"\nResult {i+1}:")
                 print(f"  Source: {doc.get('cleaned_content_path', 'N/A')}")
                 refs = doc.get('bgb_references')
                 if refs:
                     print(f"  BGB Refs: {', '.join(refs)}")
                 print(f"  Text: {doc.get('page_content', '')[:300]}...")
        else:
            print("No relevant documents found.")

    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{DATA_FILE}'. Please update the path.")
    except ImportError:
         print("ERROR: Required libraries (e.g., langchain_ollama, sentence_transformers, faiss) might be missing.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        logging.exception("Detailed traceback:") # Log the full traceback