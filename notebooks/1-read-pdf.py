import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model_name = "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(model=model_name)


def pdf_read(file):
    text = ""
    # for pdf in pdf_library:
    #     pdf_reader = PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, embedding_model):
    embeddings = embedding_model
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore.save_local("./data/faiss_db")


def main():
    print("Reading PDF")
    text = pdf_read("./data/Jurafsky-chp-5.pdf")
    print("Getting chunks")
    chunks = get_text_chunks(text)
    print("Configuring model")
    embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    print("Getting vector store")
    get_vectorstore(chunks, embeddings_model)
    print("Done!")

if __name__ == "__main__":
    main()
