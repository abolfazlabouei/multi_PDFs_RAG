import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import RetrievalQA
from langchain_core.runnables.history import RunnableWithMessageHistory
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(document):
    text = ""
    for pdf in document:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" + "\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=30,
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
    llm = OllamaLLM(
        model="mistral",
        temperature=0.7,
        max_tokens=512
    )
    history = ChatMessageHistory()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    conversation_chain = RunnableWithMessageHistory(
        runnable=qa_chain,
        get_session_history=lambda session_id: history,
        input_messages_key="query",
        output_messages_key="result",
        history_messages_key="chat_history"
    )
    return conversation_chain, history

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDF RAG", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "history" not in st.session_state:
        st.session_state.history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your PDFs")

    if user_question and st.session_state.conversation:
        truncated_question = truncate_text(user_question, max_length=400)
        response = st.session_state.conversation.invoke(
            {"query": truncated_question},
            config={"configurable": {"session_id": "default"}}
        )
        st.session_state.chat_history = st.session_state.history.messages
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
                st.session_state.conversation, st.session_state.history = get_conversation_chain(vector_store)

if __name__ == "__main__":
    main()