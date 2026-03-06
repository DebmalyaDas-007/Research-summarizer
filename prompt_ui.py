from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

st.header("Text Summarizer")

user_input = st.text_area("Enter your text here")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

if st.button("Summarize"):
    if user_input:
        result = model.invoke("Summarize the following text: " + user_input)
        st.write(result.content)
        print(result.content)
    else:
        st.warning("Please enter some text.")