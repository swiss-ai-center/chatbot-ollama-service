import os
import zipfile
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from chatcsia.llm.chain import ask_chain, prepare_chain

load_dotenv()


VECTORSTORES_DIR = os.environ.get("VECTORSTORES_DIR", None)

if not os.path.exists(VECTORSTORES_DIR):
    os.makedirs(VECTORSTORES_DIR)

STATIC_FOLDERS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)


def reset_session():
    st.session_state.messages = []
    st.session_state.chain = None
    st.session_state.zipfile_processed = False
    if "uploaded_file" in st.session_state:
        del st.session_state["uploaded_file"]
    st.rerun()


def extract_zip(uploaded_file, path_to_extract):
    if os.path.exists(os.path.join(path_to_extract, "vectorstore")):
        shutil.rmtree(os.path.join(path_to_extract, "vectorstore"))
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall(path_to_extract)
    st.session_state.zipfile_processed = True


def setup_chain(language: str = "en"):
    """
    Setup the chain.

    Args:
        machine (str): name of the machine

    Returns:
        RetrievalQA: chain
    """
    with st.spinner("Setting up chain..."):
        return prepare_chain(language=language)


def is_chain_empty() -> bool:
    """
    Check if the chain is empty and needs to be setup.

    Returns:
        bool: True if the chain is empty
    """
    return ("chain" not in st.session_state or st.session_state.chain is None) and (
        "chain_setup_in_progress" not in st.session_state
        or not st.session_state.chain_setup_in_progress
    )


logo_path = os.path.join(STATIC_FOLDERS, "img", "logo_full_white_sec.png")
st.image(Image.open(logo_path), width=300)

st.title(os.getenv("CHATBOT_NAME"))
st.set_page_config(
    page_title=os.getenv("CHATBOT_NAME"),
)

st.markdown(
    """
    Before using this chatbot go to the Swiss AI Center's
    [App](https://frontend-core-engine-swiss-ai-center.kube.isc.heia-fr.ch/showcase/service/document-vectorizer)
    and vectorize your document with the **Document Vectorizer service**.\n

    **Data disclaimer :** The data you upload to this chatbot is temporarily stored on the Swiss AI Center's server
    and deleted after the end of the session.\n
    The content of the conversation you have with the chatbot is not stored.\n

    **Usage disclaimer:** This chatbot is for testing and demo purposes. It does not provide service delivery guarantee.
    """
)


with st.expander("Extra info"):
    st.markdown(
        f"""
        model: {os.getenv("LLM_MODEL_NAME")}\n
        Number of parameters: 7 billions\n
        Created by: [mistral.ai](https://mistral.ai/)\n
        temperature: {os.getenv("LLM_TEMPERATURE")}\n
    """
    )

if "uploaded_file" not in st.session_state:
    st.text("Select a language")
    st.session_state.language = st.selectbox("Language", ["en", "fr", "de", "it"])
    temp_uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
    if temp_uploaded_file is not None:
        st.session_state.uploaded_file = temp_uploaded_file
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, st.session_state.uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(st.session_state.uploaded_file.getbuffer())

            extract_zip(zip_path, VECTORSTORES_DIR)
            st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "zipfile_processed" not in st.session_state:
    st.session_state.zipfile_processed = False
if "uploaded_file" in st.session_state:
    if is_chain_empty():
        chain = setup_chain(language=st.session_state.language)
    else:
        chain = st.session_state.chain

if st.session_state.zipfile_processed:
    st.session_state.chain_setup_in_progress = True
    st.session_state.chain = chain
    st.session_state.chain_setup_in_progress = False


if reset := st.button("Reset"):
    reset_session()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

if "uploaded_file" in st.session_state:
    if prompt := st.chat_input(""):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "text": prompt})

        with st.spinner("Thinking..."):
            response = ask_chain(prompt, chain)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            with st.expander("Sources"):
                for source in response["sources"]:
                    st.write(source)

            st.session_state.messages.append(
                {"role": "assistant", "text": response["answer"]}
            )
