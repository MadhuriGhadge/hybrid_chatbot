import os
# Disable TensorFlow/Keras
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_JAX"] = "0"

import streamlit as st
import json
import PyPDF2
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# ---------------- Load FAQ JSON ----------------
with open("faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data["faqs"]]
answers   = [item["answer"] for item in faq_data["faqs"]]

# ---------------- Load PDF ----------------
def load_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

pdf_text = load_pdf("department_pdf_data.pdf")
chunks = pdf_text.split("\n\n")  # split paragraphs

# ---------------- Embeddings with Caching ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def get_faq_embeddings():
    model = load_model()
    return model.encode(questions)

@st.cache_data
def get_pdf_embeddings():
    model = load_model()
    return model.encode(chunks)

model = load_model()
faq_embeddings = get_faq_embeddings()
pdf_embeddings = get_pdf_embeddings()

def analyze_sentiment(text):
    """Analyze user sentiment"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def save_chat(user_input, bot_reply, sentiment):
    """Save chat history with sentiment"""
    import os
    
    # Create CSV if it doesn't exist
    if not os.path.exists("chat_history.csv"):
        pd.DataFrame(columns=["timestamp", "user_input", "bot_reply", "sentiment"]).to_csv("chat_history.csv", index=False)
    
    # Append new chat
    log = pd.DataFrame([[datetime.now(), user_input, bot_reply, sentiment]], 
                       columns=["timestamp", "user_input", "bot_reply", "sentiment"])
    log.to_csv("chat_history.csv", mode="a", header=False, index=False)

def get_response(user_input):
    query_vec = model.encode([user_input])

    # Compare with FAQ
    faq_sims = cosine_similarity(query_vec, faq_embeddings)
    faq_idx = faq_sims.argmax()
    faq_score = faq_sims[0, faq_idx]

    # Compare with PDF
    pdf_sims = cosine_similarity(query_vec, pdf_embeddings)
    pdf_idx = pdf_sims.argmax()
    pdf_score = pdf_sims[0, pdf_idx]

    # Decide best source with confidence display
    confidence = max(faq_score, pdf_score)
    
    if faq_score > pdf_score and faq_score > 0.35:
        source = "FAQ"
        response = answers[faq_idx]
    elif pdf_score > 0.35:
        source = "Document"
        response = chunks[pdf_idx]
    else:
        return "Sorry, I don't understand that yet. Try rephrasing your question."
    
    return f"{response}\n\n*Source: {source} | Confidence: {confidence:.1%}*"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Smart Hybrid Chatbot", page_icon="🤖")
st.title("Hey curious Cat")
st.write("Ask me about courses or department information!")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"{message['content']}")
            if "sentiment" in message:
                st.caption(f"Sentiment: {message['sentiment']}")
        else:
            st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask me something..."):
    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    
    # Add user message with sentiment
    st.session_state.history.append({
        "role": "user", 
        "content": user_input,
        "sentiment": sentiment
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"Sentiment: {sentiment}")
    
    # Get bot response
    bot_reply = get_response(user_input)
    
    # Add bot response
    st.session_state.history.append({
        "role": "assistant",
        "content": bot_reply
    })
    
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    
    # Save to CSV
    save_chat(user_input, bot_reply, sentiment)

# Sidebar with analytics
with st.sidebar:
    st.header("Chat Analytics")
    
    # Show current session stats
    if st.session_state.history:
        user_messages = [msg for msg in st.session_state.history if msg["role"] == "user"]
        if user_messages:
            sentiments = [msg.get("sentiment", "Neutral") for msg in user_messages]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            st.write("**Current Session Sentiment:**")
            for sentiment, count in sentiment_counts.items():
                st.write(f"{sentiment}: {count}") 
            
            st.write(f"**Total Messages:** {len(user_messages)}")
        else:
            st.write("No messages yet!")
    else:
        st.write("Start chatting to see analytics!")
