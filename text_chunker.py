def split_text_into_chunks(pages, chunk_size=500, overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size"
        )
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })
            if end == len(text):
                break
            start = end - overlap

    return chunks