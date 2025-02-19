import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit App Title
st.title("AI-Powered Healthcare Assistant")

# ✅ Load AI Model
@st.cache_resource()
def load_model():
    return pipeline("text-generation", model="distilgpt2")

chatbot = load_model()  # Load model once

# Define chatbot function
def healthcare_chatbot(user_input):
    """
    Generates responses based on user input.
    - Uses predefined responses for specific medical topics.
    - If no predefined response is available, uses AI to generate text.
    """
    user_input_lower = user_input.lower()

    # ✅ Improved Rule-Based Responses with More Details
    predefined_responses = {
        "covid-19 symptoms": "Common symptoms of COVID-19 include fever, cough, shortness of breath, fatigue, body aches, sore throat, and loss of taste or smell. Severe cases may lead to difficulty breathing, chest pain, and confusion. If you experience these symptoms, seek medical attention immediately.",
        "flu symptoms": "Influenza (flu) symptoms include fever, chills, cough, sore throat, runny nose, muscle aches, headaches, and fatigue. In severe cases, it can cause pneumonia or hospitalization.",
        "medication guidelines": "Always take medications as prescribed by your doctor. Do not exceed the recommended dosage. If you experience side effects, contact your healthcare provider.",
        "healthy diet": "A healthy diet includes a balance of proteins, carbohydrates, fats, vitamins, and minerals. Consume fresh fruits, vegetables, lean proteins, and whole grains while reducing processed foods and sugars."
    }

    # ✅ Check if a predefined response exists
    for key in predefined_responses:
        if key in user_input_lower:
            return predefined_responses[key]  # Return predefined response

    # ✅ If no predefined response, use AI to generate one
    response = chatbot(user_input, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

# User Input Field
user_input = st.text_input("How can I assist you today?", "")

# Submit Button Logic
if st.button("Submit"):
    if user_input:
        response = healthcare_chatbot(user_input)
        st.write("**User:**", user_input)  # Display User Input
        st.write("**Healthcare Assistant:**", response)  # Display AI Response
    else:
        st.warning("Please enter a query before submitting.")  # Ensure no empty input
