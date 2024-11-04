import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from groq import Groq
from dotenv import load_dotenv

class Models:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # ollama pull mxbai-embed-large
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        # ollama pull llama3.2
        self.model_ollama = ChatOllama(
            model="llama3.2",
            temperature=0,
        )

        # Groq AI client with API key
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Groq AI embeddings
        self.embeddings_groq = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": "Initialize embeddings"}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Groq AI chat model
        self.model_groq = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": "Initialize chat model"}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
