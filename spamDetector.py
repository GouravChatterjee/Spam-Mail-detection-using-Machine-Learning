import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vec.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# 🔧 Modern UI Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0D1117;
        color: #C9D1D9;
    }

    h1 {
        color: #58A6FF;
        text-align: center;
        font-weight: 700;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #8B949E;
        margin-bottom: 2rem;
    }

    textarea {
        background-color: #161B22 !important;
        color: #C9D1D9 !important;
        border: 2px solid #58A6FF !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }

    .stTextArea label {
        color: #C9D1D9 !important;
        font-weight: 600;
    }

    .stButton > button {
        background-color: #58A6FF;
        color: #0D1117;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.8rem;
        margin-top: 1rem;
        font-size: 1.05rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #1F6FEB;
        transform: scale(1.05);
        color: white;
    }

    .result {
        margin-top: 2rem;
        padding: 1.2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        text-align: center;
        font-weight: 600;
    }

    .success {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid #28a745;
        color: #28a745;
    }

    .error {
        background-color: rgba(248, 81, 73, 0.1);
        border: 1px solid #f85149;
        color: #f85149;
    }

    footer, header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# 🚀 Main App
def main():
    st.markdown("<h1>📨 Spam Detector</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-powered email spam classification with a modern UI</div>", unsafe_allow_html=True)

    # 📥 Email input
    user_input = st.text_area("📬 Enter your email content:", height=160, placeholder="e.g. Congratulations! You've been selected for a free iPhone...")

    # 🔍 Predict
    if st.button("🔍 Classify"):
        if user_input.strip():
            vec = cv.transform([user_input]).toarray()
            result = model.predict(vec)

            if result[0] == 0:
                st.markdown('<div class="result success">✅ This is <strong>NOT</strong> a Spam Email.</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="result error">🚫 This is a <strong>SPAM</strong> Email. Be cautious!</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some content to classify.")

if __name__ == "__main__":
    main()
