
from pdf_reader import extract_text_from_pdf
from text_chunker import split_text_into_chunks
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


PDF_PATH = "data/sample.pdf"


def main():

    print("Loading PDF...")

    pages = extract_text_from_pdf(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")

    chunks = split_text_into_chunks(
        pages,
        chunk_size=500,
        overlap=100
    )

    print(f"Chunks created: {len(chunks)}")

    print("\nCreating FAISS index...")

    vector_store = VectorStore()
    vector_store.create_index(chunks)

    chatbot = RAGPipeline(vector_store)

    print("\nAI PDF Chatbot is ready!")

    while True:

        question = input(
            "\nAsk a question (or 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nGenerating answer...")

        try:
            result = chatbot.ask(question)

            print("\nAI Answer:")
            print(result["answer"])

            print("\nRetrieved sources:")

            for source in result["sources"]:
                print(
                    f"Page {source['page']} "
                    f"(similarity: {source['score']:.4f})"
                )

        except Exception as error:
            print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
