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

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or "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"

# Streamlit page config
st.set_page_config(page_title="Healthcare Voice Bot", page_icon="🩺", layout="wide")
st.title("🩺 Real-Time Voice-Activated Healthcare Bot")

# ---------------- PDF Upload ----------------
uploaded_files = st.file_uploader(
    "📂 Upload guidelines & patient reports (multiple PDFs)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- Callback for Streaming ----------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, tts_enabled=False, lang="en"):
        self.container = container
        self.tts_enabled = tts_enabled
        self.lang = lang
        self.text = ""
        self.audio_file = None

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(f"### 📝 Answer:\n{self.text}")
        
        # Generate speech progressively if TTS is enabled
        if self.tts_enabled and len(self.text) % 30 == 0:
            try:
                tts = gTTS(self.text, lang=self.lang)
                tts_file = BytesIO()
                tts.write_to_fp(tts_file)
                tts_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    f.write(tts_file.read())
                    self.audio_file = f.name
                st.audio(self.audio_file, format="audio/mp3")
            except Exception:
                pass

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

    # Use ConversationalRetrievalChain instead of RetrievalQA for streaming
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
    input_method = st.radio("Choose input method:", ["Type", "Speak"])
    query = ""

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
                except Exception as e:
                    st.error(f"Could not recognize speech: {e}")

    # ---------------- Real-Time Streaming Answer ----------------
    if query:
        st.markdown("### 📝 Answer:")
        container = st.empty()

        # Detect language for TTS
        try:
            lang_code = detect(query)
        except:
            lang_code = "en"

        enable_tts = st.checkbox("🔊 Speak Answer in Real-Time")

        callback = StreamHandler(container=container, tts_enabled=enable_tts, lang=lang_code)
        with st.spinner("Generating answer..."):
            qa_chain.run({"question": query, "chat_history": []}, callbacks=[callback])
