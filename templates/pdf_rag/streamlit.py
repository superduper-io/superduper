import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Volvo Technical Guide - RAG")


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
    "#### Superduper Demo App: <br> Volvo Technical Guide - RAG", unsafe_allow_html=True
)


def init_db():
    from superduper import superduper

    db = superduper("mongodb://localhost:27017/test_db")
    model_rag = db.load("model", "rag")
    return db, model_rag


def load_questions():
    return [
        "When is the air suspension system activated?",
        "What happens if Active Grip Control and the Traction Control System were off when the truck starts again?",
        "How is the instrument lighting automatically adjusted according to the ambient light?",
        "How should the new filter be screwed on?",
        "What are the options available in the radio player?",
        "Why is it important to clean the radiator with extreme caution?",
        "How can I find the Distance Alert setting?",
        "What conditions need to be fulfilled in order to start manual regeneration?",
        "What are the four positions of the nozzles on the driver's side?",
        "What does the driveline warranty cover?",
        "How can the trucks parking brake and the service brake on any connected trailer be braked gradually while driving?",
        "What are the different units of measurement for fuel consumption in the instrument display and the side display?",
        "What is the maximum freezing-point depression for concentrated coolant?",
        "How should the oil be filled in the gearbox?",
        "What must be the function mode of the electrical system in order to disconnect the batteries?",
        "How can the cargo weight be reset to zero? ",
        "How do you generate a new report when automatic pre-trip check is disabled?",
        "What is the purpose of turning the hydraulic valve anticlockwise to the stop position?",
        "What functions does the control panel for the infotainment system have?",
        "What components are included in the exhaust aftertreatment system?",
    ]


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
