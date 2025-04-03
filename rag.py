from retrival import Retriever
from langchain_ollama import ChatOllama
import langdetect
import pycountry
import torch


class Rag:
    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        self.retriever = Retriever(llm='llama3.1', device=device)
        self.llm = ChatOllama(model='gemma3:27b')
        self.history = []

    def response(self, user_query: str, top_k: int = 5, rerank: bool = True) -> str:
        """
        Get the response based on the user query.
        """
        user_lang = pycountry.languages.get(alpha_2=langdetect.detect(user_query))

        top_k_texts = self.retriever.get_results(user_query, top_k)
        prompt = f"""
        Frage: {user_query}
        
        Mögliche Gesetze im Zusammenhang mit der Benutzeranfrage:
        """ + "\n".join([f"{i+1}. {text}" for i, text in enumerate(top_k_texts)])

        prompt += f"\nBitte antworte auf {user_lang}."

        self.history.append({'role': 'user', 'content': user_query})

        print(f"Generating response...")
        response = ""

        for chunk in self.llm.stream(prompt):
            response += chunk.content
            print(chunk.content, end='', flush=True)

        self.history.append({'role': 'assistant', 'content': response})
        print()

        return response

    def chat(self):
        """
        Method for continuous chat interaction.
        """
        print("Welcome to the Legal Chatbot! Type 'exit' to end the conversation.")
        while True:
            user_query = input("You: ")

            if user_query.lower() == 'exit':
                print("Ending the conversation.")
                break
            response = self.response(user_query)
            print(f"Bot: {response}")


def main():
    rag = Rag()
    rag.chat()


if __name__ == "__main__":
    main()
