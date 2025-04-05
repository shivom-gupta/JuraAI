import streamlit as st
import langdetect
import pycountry
import torch
import logging
import textwrap

from langchain_ollama import ChatOllama
from retrieval import Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RETRIEVER_MODEL = "llama3.1"
LLM_MODEL = "gemma3:12b"

def load_models_and_retriever():
    """Loads both LLMs and the Retriever instance once."""
    with st.spinner("Loading models and retriever..."):
        try:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logging.info(f"Using device: {device}")

            with st.spinner(f"Loading LLM for Retriever ({RETRIEVER_MODEL})..."):
                retriever_llm = ChatOllama(model=RETRIEVER_MODEL, device=device)

            with st.spinner("Initializing Retriever..."):
                retriever_instance = Retriever(
                    llm=retriever_llm,
                    device=device,
                )

            with st.spinner(f"Loading main LLM for response generation ({LLM_MODEL})..."):
                main_llm = ChatOllama(model=LLM_MODEL)

            return retriever_instance, main_llm

        except Exception as e:
            st.error(f"Error loading models or retriever: {e}")
            logging.error("Initialization failed", exc_info=True)
            st.stop()

st.set_page_config(page_title="German Legal Chatbot (BGB)", layout="wide")
st.title("🇩🇪 German Legal Chatbot (BGB)")
st.caption("Ask questions about German Civil Law (BGB). This bot uses RAG to retrieve relevant BGB sections.")

try:
    retriever, main_llm = load_models_and_retriever()
except Exception as e:
    st.error("Failed to initialize models.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process new user input
if prompt := st.chat_input("Ihre Frage zum BGB... (Your question about BGB...)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Detect user's language
            try:
                user_lang_code = langdetect.detect(prompt)
                user_lang = pycountry.languages.get(alpha_2=user_lang_code).name
            except langdetect.lang_detect_exception.LangDetectException:
                user_lang = "German"
                st.warning("Could not detect language, defaulting to German.")
            except AttributeError:
                user_lang = "German"
                st.warning(f"Could not find language name for code '{user_lang_code}', defaulting to German.")

            # Retrieve relevant documents
            with st.spinner("Retrieving relevant BGB sections..."):
                top_k = 5
                retrieved_docs_data = retriever.get_results(prompt, top_n=top_k)
                retrieved_texts = [doc.get('page_content', 'Missing text') for doc in retrieved_docs_data]

            # Prepare conversation history by excluding the latest user query to avoid redundancy.
            if len(st.session_state.messages) > 1:
                conversation_history = "\n".join(
                    f"{msg['role'].capitalize()}: {msg['content']}" 
                    for msg in st.session_state.messages[:-1]
                )
            else:
                conversation_history = "No prior conversation."

            # Build final prompt with clear separation:
            if not retrieved_texts:
                final_prompt_str = textwrap.dedent(f"""
                    Conversation History:
                    {conversation_history}
                    
                    Retrieved Relevant Texts: None found.
                    
                    User's Question:
                    {prompt}
                    
                    Note: No specific laws or texts were found. Please answer the question using general legal knowledge.
                    Answer in {user_lang}.
                """)
            else:
                context_str = "\n".join([f"{i+1}. {text}" for i, text in enumerate(retrieved_texts)])
                final_prompt_str = textwrap.dedent(f"""
                    Conversation History:
                    {conversation_history}
                    
                    Retrieved Relevant Texts:
                    {context_str}
                    
                    User's Question:
                    {prompt}
                    
                    Based *only* on the retrieved texts and general legal knowledge if necessary, answer the user's question accurately.
                    If the texts do not contain the answer, state that clearly. Do not invent information not present in the texts.
                    Answer in {user_lang}.
                """)

            logging.info(f"Final prompt for main LLM:\n{final_prompt_str}")

            # Generate and stream the response
            with st.spinner(f"Generating response in {user_lang}..."):
                stream_generator = main_llm.stream(final_prompt_str)
                full_response = message_placeholder.write_stream(stream_generator)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            logging.error("Error during response generation", exc_info=True)
            full_response = "Es tut mir leid, bei der Bearbeitung Ihrer Anfrage ist ein Fehler aufgetreten. (Sorry, an error occurred while processing your request.)"
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
