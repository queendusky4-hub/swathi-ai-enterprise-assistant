def chunk_text(
    pages,
    chunk_size=500,
    overlap=100,
):
    chunks = []

    for page_number, page in enumerate(pages, start=1):

        words = page.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            text = " ".join(words[start:end])

            chunks.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

            start += chunk_size - overlap

    return chunks