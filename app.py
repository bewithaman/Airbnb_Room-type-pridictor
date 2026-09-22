import streamlit as st
import joblib

model = joblib.load("LogisticRegression_wine (1).pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Text Classifier",
    page_icon="🤖"
)

st.title("🤖 Text Classification")
st.write("Enter text below and let the model predict the category.")

text = st.text_area(
    "Enter your text",
    placeholder="Type something here..."
)

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        text_vector = tfidf.transform([text])

        prediction = model.predict(text_vector)[0]

        st.success(f"Prediction: {prediction}")stre