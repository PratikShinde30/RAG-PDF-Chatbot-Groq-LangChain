import streamlit as st
import os
import tempfile
import shutil

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:

    st.error(
        "❌ GROQ_API_KEY not found.\n\n"
        "Please add GROQ_API_KEY=your_key to your .env file."
    )

    st.stop()


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📚 RAG Document Chatbot")

st.write(
    "Upload one or more PDF documents and ask questions "
    "about their contents."
)


# =========================================================
# SESSION STATE
# =========================================================

if "rag_chain" not in st.session_state:

    st.session_state.rag_chain = None


if "vector_database" not in st.session_state:

    st.session_state.vector_database = None


if "documents_loaded" not in st.session_state:

    st.session_state.documents_loaded = False


if "uploaded_file_names" not in st.session_state:

    st.session_state.uploaded_file_names = []


# =========================================================
# PDF UPLOAD
# =========================================================

st.subheader("📁 Select PDF Documents")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================================
# SHOW SELECTED FILES
# =========================================================

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} PDF file(s) selected."
    )

    for file in uploaded_files:

        st.write(
            f"📄 {file.name}"
        )

else:

    st.info(
        "👆 Click Browse files and select your PDF documents."
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📄 Selected Documents")


if uploaded_files:

    for file in uploaded_files:

        st.sidebar.write(
            f"📄 {file.name}"
        )

else:

    st.sidebar.info(
        "No PDF files selected."
    )


# =========================================================
# LOAD PDF DOCUMENTS
# =========================================================

def load_uploaded_documents(files):

    all_documents = []

    temp_directory = tempfile.mkdtemp()

    try:

        for uploaded_file in files:

            file_path = os.path.join(
                temp_directory,
                uploaded_file.name
            )

            with open(
                file_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )


            loader = PyPDFLoader(
                file_path
            )

            documents = loader.load()


            # Store original file name
            for document in documents:

                document.metadata["source"] = (
                    uploaded_file.name
                )

            all_documents.extend(
                documents
            )

        return all_documents

    finally:

        # Delete temporary files
        shutil.rmtree(
            temp_directory,
            ignore_errors=True
        )


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

@st.cache_resource
def create_embeddings():

    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )

    return embeddings


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vector_database(documents):

    if not documents:

        return None


    chunks = create_chunks(
        documents
    )


    if not chunks:

        return None


    embeddings = create_embeddings()


    vector_database = FAISS.from_documents(

        chunks,

        embeddings

    )


    return vector_database


# =========================================================
# CREATE GROQ LLM
# =========================================================

@st.cache_resource
def create_llm():

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    groq_api_key=groq_api_key
)

    return llm


# =========================================================
# CREATE RAG CHAIN
# =========================================================

def create_rag_chain(vector_database):

    if vector_database is None:

        return None


    # =====================================================
    # RETRIEVER
    # =====================================================

    retriever = vector_database.as_retriever(

        search_kwargs={
            "k": 4
        }

    )


    # =====================================================
    # LLM
    # =====================================================

    llm = create_llm()


    # =====================================================
    # PROMPT
    # =====================================================

    prompt = ChatPromptTemplate.from_messages(

        [

            (

                "system",

                """
You are a helpful document-based question answering assistant.

Answer the user's question using ONLY the information
provided in the context.

Rules:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not make up information.
4. If the answer is not available in the context, say:

"I could not find the answer in the provided documents."

5. Give a clear and concise answer.
6. If multiple parts of the context are relevant,
combine them logically.

Context:

{context}
"""

            ),

            (

                "human",

                "{input}"

            )

        ]

    )


    # =====================================================
    # DOCUMENT CHAIN
    # =====================================================

    document_chain = create_stuff_documents_chain(

        llm,

        prompt

    )


    # =====================================================
    # RETRIEVAL CHAIN
    # =====================================================

    rag_chain = create_retrieval_chain(

        retriever,

        document_chain

    )


    return rag_chain


# =========================================================
# LOAD DOCUMENTS BUTTON
# =========================================================

st.divider()

load_button = st.button(
    "🔄 Load Documents",
    use_container_width=True
)


if load_button:

    if not uploaded_files:

        st.error(
            "❌ Please select at least one PDF file."
        )

        st.stop()


    with st.spinner(
        "📚 Loading PDF documents..."
    ):

        try:

            # ---------------------------------------------
            # Load PDFs
            # ---------------------------------------------

            documents = load_uploaded_documents(
                uploaded_files
            )


            if not documents:

                st.error(
                    "❌ No text could be extracted from the PDFs."
                )

                st.stop()


            st.info(
                f"📄 Loaded {len(documents)} PDF page(s)."
            )


            # ---------------------------------------------
            # Create vector database
            # ---------------------------------------------

            st.write(
                "🔢 Creating document embeddings..."
            )

            vector_database = create_vector_database(
                documents
            )


            if vector_database is None:

                st.error(
                    "❌ Could not create the vector database."
                )

                st.stop()


            # ---------------------------------------------
            # Create RAG chain
            # ---------------------------------------------

            st.write(
                "🤖 Creating RAG chain..."
            )

            rag_chain = create_rag_chain(
                vector_database
            )


            if rag_chain is None:

                st.error(
                    "❌ Could not create RAG chain."
                )

                st.stop()


            # ---------------------------------------------
            # Save in session
            # ---------------------------------------------

            st.session_state.vector_database = (
                vector_database
            )

            st.session_state.rag_chain = (
                rag_chain
            )

            st.session_state.documents_loaded = True

            st.session_state.uploaded_file_names = [

                file.name

                for file in uploaded_files

            ]


            st.success(
                "✅ Documents loaded successfully!"
            )


        except Exception as e:

            st.error(
                "❌ Error while loading documents:"
            )

            st.exception(e)


# =========================================================
# STATUS
# =========================================================

st.divider()

if st.session_state.documents_loaded:

    st.success(
        "🟢 RAG system is ready. You can ask questions."
    )

else:

    st.info(
        "🔵 Select PDF files and click "
        "**Load Documents** before asking questions."
    )


# =========================================================
# QUESTION SECTION
# =========================================================

st.subheader("🔎 Ask a Question")


question = st.text_input(

    "Enter your question:",

    placeholder=(
        "Example: What is the main topic of the document?"
    )

)


# =========================================================
# ASK QUESTION
# =========================================================

if question:

    if st.session_state.rag_chain is None:

        st.warning(
            "⚠️ Please click **Load Documents** first."
        )

    else:

        with st.spinner(
            "🔍 Searching your documents..."
        ):

            try:

                response = (
                    st.session_state.rag_chain.invoke(
                        {
                            "input": question
                        }
                    )
                )


                # =================================================
                # ANSWER
                # =================================================

                st.subheader("🤖 Answer")

                st.write(
                    response["answer"]
                )


                # =================================================
                # RETRIEVED DOCUMENTS
                # =================================================

                st.subheader(
                    "📄 Retrieved Information"
                )


                retrieved_documents = response.get(
                    "context",
                    []
                )


                if retrieved_documents:

                    for i, document in enumerate(
                        retrieved_documents
                    ):

                        with st.expander(
                            f"📄 Source {i + 1}"
                        ):


                            # -------------------------------------
                            # SOURCE FILE
                            # -------------------------------------

                            source = document.metadata.get(
                                "source",
                                "Unknown"
                            )


                            st.markdown(
                                f"**📁 File:** `{source}`"
                            )


                            # -------------------------------------
                            # PAGE NUMBER
                            # -------------------------------------

                            page = document.metadata.get(
                                "page"
                            )


                            if page is not None:

                                st.markdown(
                                    f"**📄 Page:** `{page + 1}`"
                                )


                            # -------------------------------------
                            # RETRIEVED TEXT
                            # -------------------------------------

                            st.markdown(
                                "**Retrieved Content:**"
                            )


                            st.write(
                                document.page_content
                            )


                else:

                    st.info(
                        "No relevant information was retrieved."
                    )


            except Exception as e:

                st.error(
                    "❌ Error while generating the answer:"
                )

                st.exception(e)


# =========================================================
# SIDEBAR STATUS
# =========================================================

st.sidebar.divider()

if st.session_state.documents_loaded:

    st.sidebar.success(
        "🟢 RAG Ready"
    )

else:

    st.sidebar.info(
        "🔵 Waiting for documents"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📚 RAG Chatbot | Groq + LangChain + "
    "HuggingFace Embeddings + FAISS"
)