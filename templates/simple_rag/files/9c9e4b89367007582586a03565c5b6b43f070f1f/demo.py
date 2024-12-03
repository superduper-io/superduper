def demo_func(db):
    import streamlit as st
    import openai

    openai.api_key = "your_openai_api_key"

    st.title("Chat with the Superduper docs!")

    user_input = st.text_input("Your Question:", key="question_input")

    rag = db.load('model', 'simple_rag')

    if st.button("Get Answer") and user_input:
        with st.spinner("Thinking..."):
            answer = rag.predict(user_input)
            st.write(f"**Answer:** {answer}")
