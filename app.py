import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Load model & vectorizer
with open("best_model_france.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer_france.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Wide layout
st.set_page_config(page_title=" FRANCE Sentiment Analyzer", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        body, .main {
            background-color: #121212;
            color: #f5f5f5;
        }
        h1 {
            font-size: 88px !important;
            text-align: center;
            color: #f39c12;
        }
        .big-text-area textarea {
            font-size: 28px !important;
            height: 250px !important;
            width: 60% !important;
            background-color: #1e1e1e !important;
            color: white !important;
            border-radius: 10px !important;
        }
        .stButton button {
            background-color: #2e86de;
            color: white;
            font-size: 28px;
            border-radius: 10px;
            padding: 10px 25px;
        }
        .centered-div {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-top: 50px;
        }
        .container {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 50px;
        }
        .graph-container {
            width: 70%;
            height: 400px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🇫🇷 France Wikipedia Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size: 20px;'>Discover sentiment insights from French Wikipedia-style content</p>", unsafe_allow_html=True)
st.markdown("---")

# Create two columns for input and graph display
col1, col2 = st.columns([2, 1])  # Adjust column width as needed

# Centered input in column 1
with col1:
    st.markdown("### ✍️ Enter Text for Sentiment Analysis", unsafe_allow_html=True)
    user_input = st.text_area("", placeholder="Type or paste French text here...", key="text_input", label_visibility="collapsed")

# Predict Button
predict_button = st.button("Predict Sentiment")

# Sentiment functions
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

def predict_sentiment(text):
    sentiment = get_sentiment(text)
    vectorized = vectorizer.transform([text])
    if sentiment == "Neutral":
        return {"Neutral": 1.0, "Positive": 0.0, "Negative": 0.0}
    prediction = model.predict_proba(vectorized)[0]
    return {"Neutral": 0.0, "Positive": prediction[1], "Negative": prediction[0]}

# Display result + word cloud when predict button is pressed
if predict_button and user_input:
    with col2:
        st.markdown("### 📊 Sentiment Prediction")
        result = predict_sentiment(user_input)
        result_df = pd.DataFrame(result.items(), columns=["Sentiment", "Probability"])

        # Create a smaller graph container
        fig, ax = plt.subplots(figsize=(8, 6))  # Limiting the size of the graph
        colors = ['#F8766D', '#00BA38', '#619CFF']
        wedges, texts, autotexts = ax.pie(
            result_df["Probability"],
            labels=result_df["Sentiment"],
            autopct='%1.1f%%',
            startangle=150,
            colors=colors,
            wedgeprops=dict(width=0.4)
        )
        ax.set_title("Sentiment Prediction", fontsize=50, color="white")
        
        # Display the graph within a limited size
        st.pyplot(fig, use_container_width=True)

        st.markdown("### ☁ Word Cloud")
        st.image("france_wordcloud.png", caption="Word Frequency from France Wikipedia", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size: 30px;'>🔍 Created with ❤️ by Maddi Jahnavi</p>", unsafe_allow_html=True)
