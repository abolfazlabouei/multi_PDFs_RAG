import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(document):
    text = ""
    for pdf in document:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,  
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def truncate_text(text, max_length=400):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=max_length,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks[0] if chunks else text


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    pipe = pipeline(
        "text2text-generation",
        model="t5-small",
        device=-1,  # CPU
        max_length=512,
        truncation=True,  
        clean_up_tokenization_spaces=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=message_history
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    
    st.set_page_config(page_title="Multiple PDF RAG", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your PDFs")

    if user_question and st.session_state.conversation:
        truncated_question = truncate_text(user_question, max_length=400)
        response = st.session_state.conversation.invoke({"question": truncated_question})
        st.session_state.chat_history = response["chat_history"]
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your PDFs")
        document = st.file_uploader("Upload your PDFs and click on process", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your PDFs..."):
                raw_text = get_pdf_text(document)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
