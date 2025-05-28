import streamlit as st
import requests
from datetime import datetime

# Configure the Streamlit page
st.set_page_config(
    page_title="Legal Document Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and instructions
st.header("Legal Document Chatbot")
st.write("""
Welcome to the Legal Document Chatbot! This tool allows you to input legal case reports and 
classify them into areas of law using either a TF-IDF or BERT model. Enter your text, select 
a model, and get instant predictions in a conversational format.
""")

# Sidebar for settings
with st.sidebar:
    st.subheader("Chat Settings")
    selected_model = st.selectbox(
        "Choose Classification Model",
        ["tfidf", "bert"],
        index=0,
        help="Select 'tfidf' for Logistic Regression (faster, more accurate) or 'bert' for Transformer-based classification."
    )
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Input form for legal text
with st.form("legal_input", clear_on_submit=True):
    legal_text = st.text_area(
        "Legal Case Report (min 50 characters)",
        placeholder="Paste or type your legal document here...",
        height=200,
        key="input_field"
    )
    submit = st.form_submit_button("Analyze Document")

# Process input and display chat
if submit and legal_text:
    # Validate input length
    if len(legal_text.strip()) < 50:
        st.error("Please provide a legal report with at least 50 characters.")
    else:
        # Record user input
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": legal_text, "time": current_time})

        # Call the API
        try:
            api_response = requests.post(
                url=f"http://localhost:8000/predict?model_type={selected_model}",
                json={"full_report": legal_text},
                headers={"Content-Type": "application/json"}
            )
            api_response.raise_for_status()
            prediction = api_response.json()

            # Format bot response with HTML for line breaks
            bot_response = (
                f"<b>Area of Law:</b> {prediction['area_of_law']}<br>"
                f"<b>Confidence:</b> {round(float(prediction['confidence']) * 100, 2)}%<br>"
                f"<b>Model:</b> {prediction['model_used']}<br>"
                f"<b>Text Length:</b> {prediction['input_length']} characters"
            )
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "time": current_time})
        except requests.RequestException as e:
            error_msg = f"Failed to connect to API: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "time": current_time})
            st.error(error_msg)

# Display chat history
st.subheader("Chat History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            f"**{message['role'].capitalize()} at {message['time']}**: {message['content']}",
            unsafe_allow_html=True if message["role"] == "assistant" else False
        )

# Add some basic styling
st.markdown("""
<style>
.stChatMessage { border-radius: 8px; padding: 10px; margin: 5px 0; }
.stChatMessage.user { background-color: #e6f3fa; }
.stChatMessage.assistant { background-color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)