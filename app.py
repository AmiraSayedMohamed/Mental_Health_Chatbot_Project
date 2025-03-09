import os
import streamlit as st
from chatbot import initialize_llm, setup_qa_chain
from vector_db import load_or_create_vector_db
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Streamlit UI
st.title("🧠 Mental Health Chatbot 🤖")
st.write("A chatbot designed to assist with mental well-being. If you need professional help, seek a licensed expert.")

# Load or create the vector database
llm = initialize_llm()
vector_db = load_or_create_vector_db()
qa_chain = setup_qa_chain(vector_db, llm)

# Chatbot interface using Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything about mental health...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get chatbot response
    response = qa_chain.run(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
