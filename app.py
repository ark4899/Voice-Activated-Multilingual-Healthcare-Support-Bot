import streamlit as st
import tempfile
import os
import openai
import shutil
import faiss
import pandas as pd
import pytesseract
from PIL import Image

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Try to find tesseract automatically
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Fallback to default path
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\akansha.khandare\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"



# Streamlit page config
st.set_page_config(page_title="Healthcare Voice Bot", page_icon="🩺", layout="wide")
st.title("🩺 Real-Time Voice-Activated Healthcare Bot")

# ----------------- Extract Text from Any File -----------------
def extract_text_from_file(file):
    ext = os.path.splitext(file.name)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        # PDF files
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])

        # DOCX files
        elif ext == ".docx":
            from docx import Document
            doc = Document(tmp_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        # TXT files
        elif ext == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()

        # CSV files
        elif ext == ".csv":
            df = pd.read_csv(tmp_path)
            return df.to_string()

        # Excel files
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(tmp_path)
            return df.to_string()

        # Images (JPG, JPEG, PNG)
        elif ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(tmp_path)
            return pytesseract.image_to_string(image)

        else:
            st.warning(f"⚠️ Unsupported file type: {ext}")
            return ""

    except Exception as e:
        st.error(f"❌ Error reading file {file.name}: {e}")
        return ""

# ---------------- Initialize session state ----------------
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "auto_speak" not in st.session_state:
    st.session_state.auto_speak = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------------- File Upload ----------------
uploaded_files = st.file_uploader(
    "📂 Upload guidelines, reports, images, spreadsheets (multiple files)",
    type=["pdf", "docx", "txt", "csv", "xls", "xlsx", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

query = ""
if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        content = extract_text_from_file(file)
        if content:
            all_text += content + "\n"

    if all_text.strip():
        # Split & create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.create_documents([all_text])
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # ConversationalRetrievalChain
        chat_model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            return_source_documents=False,
            verbose=False
        )
        st.success("✅ All files processed and indexed!")

        # ---------------- User Input ----------------
        st.subheader("Ask a question")
        input_method = st.radio("Choose input method:", ["Type", "Speak"])
        if input_method == "Type":
            query = st.text_input("💬 Type your question here:")
        elif input_method == "Speak":
            if st.button("🎤 Record Question"):
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.write("🎙️ Listening...")
                    audio = recognizer.listen(source)
                try:
                    query = recognizer.recognize_google(audio)
                    st.success(f"You said: {query}")
                    st.session_state.auto_speak = True
                except Exception as e:
                    st.error(f"Could not recognize speech: {e}")

        # ---------------- Generate Answer ----------------
        if query:
            if st.session_state.get("last_query") != query:
                with st.spinner("Generating answer..."):
                    result = qa_chain.run({"question": query, "chat_history": []})
                st.session_state.last_answer = result
                st.session_state.last_query = query
                st.session_state.audio_path = None

        # ---------------- Show Answer & Audio ----------------
        if st.session_state.get("last_answer"):
            st.markdown("### 📝 Answer:")
            st.write(st.session_state.last_answer)

            # Speak Answer
            speak_answer = st.checkbox("🔊 Speak Answer", key="speak_checkbox")
            if st.session_state.get("input_method") == "Speak":
                speak_answer = True

            if speak_answer:
                if not st.session_state.audio_path:
                    try:
                        lang_code = detect(st.session_state.last_answer)
                    except:
                        lang_code = "en"
                    tts = gTTS(st.session_state.last_answer, lang=lang_code)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        tts.save(f.name)
                        st.session_state.audio_path = f.name
                st.audio(st.session_state.audio_path, format="audio/mp3")
