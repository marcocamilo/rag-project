
from dotenv import load_dotenv
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

from utils.functions import arxiv_pdf_reader
from utils.nlp import preprocessing

load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)
# model_name = "gemini-1.5-flash"
# llm = ChatGoogleGenerativeAI(model=model_name)

url = "https://arxiv.org/pdf/2402.07927"
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(text_chunks, embeddings, name):
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore.save_local(f"./data/0-external/{name}")

def vectorize_document(url, name, embeddings):
    pdf_text = arxiv_pdf_reader(url)
    text = preprocessing(pdf_text)
    chunks = get_text_chunks(text)
    vectorstore = create_vectorstore(chunks, embeddings, name)
    return vectorstore

def load_database(name, embeddings):
    database = FAISS.load_local(
        f"./data/0-external/{name}", embeddings, allow_dangerous_deserialization=True
    )
    return database

def create_retrievers(text_chunks, vectorstore, k):
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    keyword_retriever = BM25Retriever.from_texts(text_chunks)
    keyword_retriever.k =  k
    return vector_retriever, keyword_retriever

def ensemble_retriever(query, vector_retriever, keyword_retriever):
    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever],
                                       weights=[0.5, 0.5])
    matches = ensemble_retriever.get_relevant_documents(query)
    return matches



def main():

