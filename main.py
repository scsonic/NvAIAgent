from datasets import load_dataset
import torch
import cohere
from llama_cpp import Llama
from langchain.retrievers import ContextualCompressionRetriever, CohereRagRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# Doomsday Civilization Codex for NVIDIA AI
# 1. load pre-embedding wiki model from cohere about 2G
# 2. execute query with phi3

from langchain_community.llms import LlamaCpp
phi3 = LlamaCpp(
      model_path="Phi-3-mini-128k-instruct.gguf",
      n_gpu_layers=1000,
      n_ctx=128000,
      verbose=True,
)

cohere_embeddings = CohereEmbeddings(cohere_api_key="", model="Cohere/wikipedia-22-12-simple-embeddings")

while True:
    prirnt("Hi, I am Doomsday Civilization Codex, you can ask any question, for glory of mankind!")

    user_query = input()
    rag = CohereRagRetriever(llm=phi3)
    docs = rag.get_relevant_documents(user_query)

    for doc in docs[:-1]:
        print(doc.metadata)
        print(doc.page_content)
        print("\n")
    answer = docs[-1].page_content
    print(answer)