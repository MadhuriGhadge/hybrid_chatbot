import streamlit as st
import json
import PyPDF2
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import os

# ---------------- Configuration ----------------
st.set_page_config(page_title="Smart Hybrid Chatbot", page_icon="🤖")

# ---------------- Helper Functions ----------------
def load_faq_data():
    """Load FAQ data with error handling"""
    try:
        if os.path.exists("faq.json"):
            with open("faq.json", "r", encoding="utf-8") as f:
                faq_data = json.load(f)
            return faq_data.get("faqs", [])
        else:
            st.error("faq.json file not found!")
            return []
    except Exception as e:
        st.error(f"Error loading FAQ data: {str(e)}")
        return []

def load_pdf(file_path):
    """Load PDF with error handling"""
    try:
        if not os.path.exists(file_path):
            st.error(f"PDF file '{file_path}' not found!")
            return ""
        
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return ""

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return "Neutral"

def save_chat(user_input, bot_reply, sentiment):
    """Save chat to CSV with error handling"""
    try:
        if not os.path.exists("chat_history.csv"):
            df = pd.DataFrame(columns=["timestamp", "user_input", "bot_reply", "sentiment"])
            df.to_csv("chat_history.csv", index=False)

        log = pd.DataFrame([[datetime.now(), user_input, bot_reply, sentiment]],
                          columns=["timestamp", "user_input", "bot_reply", "sentiment"])
        log.to_csv("chat_history.csv", mode="a", header=False, index=False)
    except Exception as e:
        st.error(f"Error saving chat: {str(e)}")

# ---------------- Cached Resources ----------------
@st.cache_resource
def load_model():
    """Load sentence transformer model"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def get_faq_embeddings(questions):
    """Get FAQ embeddings"""
    if not questions:
        return []
    
    model = load_model()
    if model is None:
        return []
    
    try:
        return model.encode(questions)
    except Exception as e:
        st.error(f"Error creating FAQ embeddings: {str(e)}")
        return []

@st.cache_data
def get_pdf_embeddings(chunks):
    """Get PDF embeddings"""
    if not chunks:
        return []
    
    model = load_model()
    if model is None:
        return []
    
    try:
        return model.encode(chunks)
    except Exception as e:
        st.error(f"Error creating PDF embeddings: {str(e)}")
        return []

# ---------------- Data Loading ----------------
def initialize_data():
    """Initialize all data and embeddings"""
    # Load FAQ data
    faq_list = load_faq_data()
    questions = [item.get("question", "") for item in faq_list if "question" in item]
    answers = [item.get("answer", "") for item in faq_list if "answer" in item]
    
    # Load PDF data
    pdf_text = load_pdf("department_pdf_data.pdf")
    chunks = []
    if pdf_text:
        chunks = [chunk.strip() for chunk in pdf_text.split("\n\n") if chunk.strip()]
    
    # Get embeddings
    faq_embeddings = get_faq_embeddings(questions) if questions else []
    pdf_embeddings = get_pdf_embeddings(chunks) if chunks else []
    
    return questions, answers, chunks, faq_embeddings, pdf_embeddings

# ---------------- Response Function ----------------
def get_response(user_input, questions, answers, chunks, faq_embeddings, pdf_embeddings):
    """Generate response based on user input"""
    model = load_model()
    if model is None:
        return "Sorry, the model is not available. Please try again later."
    
    if not user_input.strip():
        return "Please enter a valid question."
    
    try:
        query_vec = model.encode([user_input])
        
        faq_score = 0
        pdf_score = 0
        faq_idx = -1
        pdf_idx = -1
        
        # Compare with FAQ if available
        if len(faq_embeddings) > 0 and len(questions) > 0:
            faq_sims = cosine_similarity(query_vec, faq_embeddings)
            faq_idx = faq_sims.argmax()
            faq_score = faq_sims[0, faq_idx]
        
        # Compare with PDF if available
        if len(pdf_embeddings) > 0 and len(chunks) > 0:
            pdf_sims = cosine_similarity(query_vec, pdf_embeddings)
            pdf_idx = pdf_sims.argmax()
            pdf_score = pdf_sims[0, pdf_idx]
        
        confidence = max(faq_score, pdf_score)
        
        if faq_score > pdf_score and faq_score > 0.35 and faq_idx >= 0:
            source = "FAQ"
            response = answers[faq_idx] if faq_idx < len(answers) else "Answer not found."
        elif pdf_score > 0.35 and pdf_idx >= 0:
            source = "Document"
            response = chunks[pdf_idx] if pdf_idx < len(chunks) else "Content not found."
        else:
            return "Sorry, I don't have enough information to answer that question. Try rephrasing or ask about something else."
        
        return f"{response}\n\n*Source: {source} | Confidence: {confidence:.1%}*"
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, there was an error processing your request."

# ---------------- Main App ----------------
def main():
    st.title("Hey Curious Cat")
    st.write("Ask me about courses or department information!")
    
    # Initialize data
    with st.spinner("Loading data and models..."):
        questions, answers, chunks, faq_embeddings, pdf_embeddings = initialize_data()
    
    # Show data status
    with st.expander("Data Status", expanded=False):
        st.write(f"FAQ Questions: {len(questions)}")
        st.write(f"PDF Chunks: {len(chunks)}")
        st.write(f"FAQ Embeddings: {'Ready' if len(faq_embeddings) > 0 else 'Not available'}")
        st.write(f"PDF Embeddings: {'Ready' if len(pdf_embeddings) > 0 else 'Not available'}")
    
    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Display chat history
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "user" and "sentiment" in message:
                st.caption(f"Sentiment: {message['sentiment']}")
    
    # Chat input
    if user_input := st.chat_input("Ask me something..."):
        sentiment = analyze_sentiment(user_input)
        
        # User message
        st.session_state.history.append({"role": "user", "content": user_input, "sentiment": sentiment})
        with st.chat_message("user"):
            st.markdown(user_input)
            st.caption(f"Sentiment: {sentiment}")
        
        # Bot response
        with st.spinner("Thinking..."):
            bot_reply = get_response(user_input, questions, answers, chunks, faq_embeddings, pdf_embeddings)
        
        st.session_state.history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        
        # Save to CSV
        save_chat(user_input, bot_reply, sentiment)
    
    # Sidebar with analytics
    with st.sidebar:
        st.header("Chat Analytics")
        if st.session_state.history:
            user_msgs = [m for m in st.session_state.history if m["role"] == "user"]
            if user_msgs:
                sentiments = [m.get("sentiment", "Unknown") for m in user_msgs]
                sentiment_counts = pd.Series(sentiments).value_counts()
                st.write("**Current Session Sentiment:**")
                for s, c in sentiment_counts.items():
                    st.write(f"{s}: {c}")
                st.write(f"**Total Messages:** {len(user_msgs)}")
            else:
                st.write("No user messages yet.")
        else:
            st.write("Start chatting to see analytics!")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()