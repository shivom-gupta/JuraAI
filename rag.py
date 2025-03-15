from retrival import Retriever
from langchain_ollama import ChatOllama
import langdetect
import pycountry


class Rag:
    def __init__(self):
        super().__init__()
        self.retriever = Retriever()
        self.llm = ChatOllama(model='qwq')

    def response(self, user_query: str, top_k: int = 5, rerank: bool = True) -> str:
        """
        Get the response based on the user query.
        """
        user_lang = pycountry.languages.get(alpha_2 = langdetect.detect(user_query))

        top_k_texts = self.retriever.get_results(user_query, top_k)
        prompt = f"""
        Frage: {user_query}
        
        Mögliche Gesetze im Zusammenhang mit der Benutzeranfrage:
        """ + "\n".join([f"{i+1}. {text}" for i, text in enumerate(top_k_texts)])

        prompt += f"\nBitte antworte auf {user_lang}."

        print(f"Generating response...")
        response = ""
        for chunk in self.llm.stream(prompt):
            response += chunk.content
            print(chunk.content, end='', flush=True)
        print()
        return response

def main():
    rag = Rag()
    user_query = "what happens in case of unjustified subletting"
    response = rag.response(user_query)
    print(response)

if __name__ == "__main__":
    main()
