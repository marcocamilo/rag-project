import os

import google.generativeai as genai
from dotenv import load_dotenv, main
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.functions import arxiv_pdf_reader
from utils.nlp import preprocessing

load_dotenv()

# GOAL: Parse contents from arXiv PDF URL

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model_name = "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(model=model_name)

url = "https://arxiv.org/pdf/2402.07927"
pdf_text = arxiv_pdf_reader(url)
text = preprocessing(pdf_text)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore.save_local("./data/0-external/faiss_db")


def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperatue=0.5)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = create_stuff_documents_chain(model, prompt)

    return chain


def user_input(user_question):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    new_db = FAISS.load_local(
        "/kaggle/working/faiss_db/", embeddings, allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain.invoke({"context": docs, "question": user_question})

    print(response)

def main():
    user_input("List the prompt engineering techniques mentioned across the article")

if __name__ == "__main__":
    main()
