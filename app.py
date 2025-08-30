import streamlit as st
import tempfile
import os
import openai
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Speech / Translation
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from langdetect import detect
import asyncio
import edge_tts

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or "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"

# Streamlit page config
st.set_page_config(page_title="Healthcare Voice Bot", page_icon="🩺", layout="wide")
st.title("🩺 Real-Time Voice-Activated Healthcare Bot")

# Initialize session state for answer
if "answer" not in st.session_state:
    st.session_state.answer = None

    
# ---------------- PDF Upload ----------------
uploaded_files = st.file_uploader(
    "📂 Upload guidelines & patient reports (multiple PDFs)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- Callback for Streaming ----------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(f"### 📝 Answer:\n{self.text}")

# Initialize session state variables
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "auto_speak" not in st.session_state:
    st.session_state.auto_speak = False

if uploaded_files:
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    # Split & create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ConversationalRetrievalChain for streaming
    chat_model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=False,
        verbose=False
    )

    st.success("✅ Documents processed and indexed!")

    # ---------------- User Input ----------------
    st.subheader("Ask a question")
    previous_method = st.session_state.get("input_method", None)
    input_method = st.radio("Choose input method:", ["Type", "Speak"])
    query = ""

    # Reset audio whenever input method changes
    if previous_method and previous_method != input_method:
        st.session_state.audio_path = None
        st.session_state.last_answer = ""
    st.session_state.input_method = input_method

    # Flag to check if auto-speaking is needed
    st.session_state.auto_speak = False

    if input_method == "Type":
        query = st.text_input("💬 Type your question here:")

    elif input_method == "Speak":
        if st.button("🎤 Record Question"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Listening... Speak now!")
                audio = recognizer.listen(source)
                try:
                    query = recognizer.recognize_google(audio)
                    st.success(f"You said: {query}")
                    st.session_state.auto_speak = True  # Enable auto-speak for voice input
                except Exception as e:
                    st.error(f"Could not recognize speech: {e}")

    # ---------------- Generate Answer ----------------
    if query:
     # Check if it's a new query, avoid re-running unnecessarily
        if st.session_state.get("last_query") != query:
            st.markdown("### 📝 Answer:")
            container = st.empty()
            callback = StreamHandler(container=container)

            with st.spinner("Generating answer..."):
                result = qa_chain.run({"question": query, "chat_history": []}, callbacks=[callback])

            # Save answer and reset audio
            st.session_state.last_answer = result
            st.session_state.last_query = query
            st.session_state.audio_path = None

# Always display the saved answer if available
if st.session_state.get("last_answer"):
        st.write(st.session_state.last_answer)


   # ---------------- Speak Answer ----------------
if st.session_state.get("last_answer"):   # <-- Make sure this starts at 0 indent
    speak_answer = st.checkbox("🔊 Speak Answer")

    if speak_answer:
        # Generate audio only once per answer
        if "audio_path" not in st.session_state or not st.session_state.audio_path:
            try:
                from gtts import gTTS
                import tempfile
                from langdetect import detect
                lang_code = detect(st.session_state.last_answer)
            except:
                lang_code = "en"

            # Create temporary audio file
            tts = gTTS(st.session_state.last_answer, lang=lang_code)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                st.session_state.audio_path = f.name

        # Instantly play saved audio
        st.audio(st.session_state.audio_path, format="audio/mp3")


