import hashlib
import streamlit as st
from pdf_reader import extract_text_from_bytes
from text_chunker import split_text_into_chunks
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

def process_pdf(pdf_bytes):
    pages = extract_text_from_bytes(
        pdf_bytes
    )
    chunks = split_text_into_chunks(
        pages,
        chunk_size=500,
        overlap=100
    )

    if not chunks:
        raise ValueError(
            "PDF-ul nu conține text care poate fi extras."
        )
    vector_store = VectorStore()
    vector_store.create_index(
        chunks
    )
    rag_pipeline = RAGPipeline(
        vector_store
    )
    return (
        rag_pipeline,
        len(pages),
        len(chunks)
    )
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader(
        "Încarcă un document PDF",
        type=["pdf"]
    )
    st.divider()
    st.caption(
        "AI local: Ollama + FAISS + PyMuPDF"
    )
st.title("AI PDF Chatbot")
st.write(
    "Încarcă un PDF și pune întrebări "
    "despre conținutul său."
)

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()
    current_document_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()
    if (
        current_document_id
        != st.session_state.document_id
    ):
        st.session_state.rag_pipeline = None
        st.session_state.messages = []
        st.session_state.document_id = None
        st.session_state.document_name = None
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        with st.spinner(
            "Procesăm documentul PDF..."
        ):

            try:

                (
                    pipeline,
                    page_count,
                    chunk_count
                ) = process_pdf(pdf_bytes)

                st.session_state.rag_pipeline = pipeline
                st.session_state.document_id = current_document_id
                st.session_state.document_name = uploaded_file.name
                st.session_state.page_count = page_count
                st.session_state.chunk_count = chunk_count

            except Exception as error:

                st.error(
                    "Eroare la procesarea PDF-ului: "
                    f"{error}"
                )

else:
    st.session_state.rag_pipeline = None
    st.session_state.messages = []
    st.session_state.document_id = None
    st.session_state.document_name = None
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0

if st.session_state.rag_pipeline is not None:
    st.success(
        "Document pregătit: "
        f"{st.session_state.document_name}"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Pagini",
            st.session_state.page_count
        )
    with col2:
        st.metric(
            "Chunk-uri",
            st.session_state.chunk_count
        )
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )
            if message.get("sources"):
                with st.expander(
                    "Surse PDF"
                ):
                    for source in message["sources"]:
                        st.write(
                            f"Pagina {source['page']} "
                            f"– Similaritate: "
                            f"{source['score']:.4f}"
                        )
                        st.caption(
                            source["text"]
                        )
    question = st.chat_input(
        "Scrie o întrebare despre PDF..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner(
                "AI-ul generează răspunsul..."
            ):
                try:
                    result = (
                        st.session_state.rag_pipeline.ask(
                            question
                        )
                    )
                    answer = result["answer"]
                    st.markdown(answer)
                    with st.expander(
                        "Surse PDF"
                    ):
                        for source in result["sources"]:
                            st.write(
                                f"Pagina {source['page']} "
                                f"– Similaritate: "
                                f"{source['score']:.4f}"
                            )
                            st.caption(
                                source["text"]
                            )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"]
                    })
                except Exception as error:
                    error_message = str(error)
                    if "timed out" in error_message.lower():
                        st.error(
                            "Ollama a depășit timpul de "
                            "așteptare. Modelul poate "
                            "rula prea lent pe calculator."
                        )
                    elif "connection" in error_message.lower():
                        st.error(
                            "Nu ne putem conecta la Ollama. "
                            "Verifică dacă aplicația "
                            "Ollama rulează."
                        )
                    else:
                        st.error(
                            "Eroare la generarea răspunsului: "
                            f"{error}"
                        )
else:
    st.info(
        "Încarcă un PDF pentru a începe conversația."
    )
