from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed(texts: list[str]):
    return MODEL.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )