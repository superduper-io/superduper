import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Technical Guide - PDF RAG")


def add_logo(width=250):
    LOGO_URL = "https://superduperdb-public-demo.s3.amazonaws.com/superduper_logo.svg"
    st.markdown(
        f"""
        <style>
        .fixed-logo img {{
            position: fixed;
            top: 1rem;
            left: 1rem;
            z-index: 9999999;
        }}
        </style>
        <div class="fixed-logo">
            <img src="{LOGO_URL}" alt="Logo" style="width: {width}px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )


add_logo()

st.markdown(
    "#### Superduper Demo (Volvo) App: <br>Technical Guide - PDF RAG", unsafe_allow_html=True
)


def init_db():
    from superduper import superduper

    db = superduper()
    model_rag = db.load("model", "rag")
    return db, model_rag


def load_questions():
    return ["What is sparse-vector retrieval?", "How to perform Query Optimization?"]


db, model_rag = st.cache_resource(init_db)()
questions = st.cache_resource(load_questions)()


def get_user_input(input_mode, input_key, questions):
    """
    A function to get user input based on the input mode
    """
    if input_mode == "Free text":
        return st.text_input(
            "Enter your text", placeholder="Type here...", key=input_key
        )
    else:  # Question Selection
        return st.selectbox("Choose a question:", questions, key=input_key)


qa_mode = st.radio(
    "Choose your input type:",
    ["Predefined question", "Free text"],
    key="qa_mode",
    horizontal=True,
)
query = get_user_input(qa_mode, "qa_query", questions)

submit_button = st.button("Search", key="qa")
if submit_button:
    st.markdown("#### Input/Query")
    st.markdown(query)
    result = model_rag.predict(query, top_k=5, format_result=True)
    st.markdown("#### Answer:")
    st.markdown(result["answer"])

    st.markdown("#### Related Documents:")
    for text, img in result["images"]:
        st.markdown(text)
        if img:
            st.image(img)
