from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .policy import parse_docx_to_texts_metadatas
from .utils import sha256_file


try: 
    from langchain_core.embeddings import Embeddings as _Base
except Exception: 
    _Base = object


class MultilingualE5Embeddings(_Base):
    """Notebook class, unchanged: mean-pooled, L2-normalised e5 embeddings with 'passage:'/'query:' prefixes."""

    def __init__(self, model_name="intfloat/multilingual-e5-small", device=None, max_length=512, revision=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(self.device)
        self.model.eval()

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed(self, texts, prefix):
        import torch.nn.functional as F
        if isinstance(texts, str):
            texts = [texts]
        texts = [f"{prefix}: {t}" for t in texts]
        batch_dict = self.tokenizer(texts, max_length=self.max_length, padding=True,
                                    truncation=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed_documents(self, texts):
        return self._embed(texts, prefix="passage").tolist()

    def embed_query(self, text):
        return self._embed([text], prefix="query")[0].tolist()

    def __call__(self, text):
        return self.embed_query(text)

    def count_truncated(self, texts: List[str]) -> int:
        """How many 'passage: <text>' inputs exceed max_length (the notebook embedder truncates them)."""
        n = 0
        for t in texts:
            ids = self.tokenizer(f"passage: {t}", truncation=False)["input_ids"]
            n += int(len(ids) > self.max_length)
        return n

    def release(self):
        import gc
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


@dataclass
class Retriever:
    vectorstore: object
    embedder: object
    corpus_path: str
    corpus_sha256: str
    embedding_model: str
    n_chunks: int
    n_chunks_truncated_by_embedder: int
    embedding_max_length: int
    chunking: str

    def retrieve(self, query: str, k: int) -> List[dict]:
        """Top-k chunks, ranked as in the notebook's retriever.invoke (FAISS similarity_search)."""
        hits = self.vectorstore.similarity_search_with_score(query, k=k)
        out = []
        for rank, (doc, score) in enumerate(hits, 1):
            out.append({"rank": rank, "title": doc.metadata.get("title"),
                        "score_l2_distance": float(score), "text": doc.page_content})
        return out

    def info(self) -> dict:
        return {"corpus_path": self.corpus_path, "corpus_sha256": self.corpus_sha256,
                "embedding_model": self.embedding_model, "embedding_max_length": self.embedding_max_length,
                "chunking": self.chunking, "n_chunks": self.n_chunks,
                "n_chunks_truncated_by_embedder": self.n_chunks_truncated_by_embedder,
                "embedding_revision": self.embedder.revision,
                "vector_store": "FAISS (langchain_community, default IndexFlatL2)"}


def build_retriever(retrieval_cfg: dict) -> Retriever:
    from langchain_community.vectorstores import FAISS
    corpus = Path(retrieval_cfg["corpus_path"])
    if not corpus.exists():
        raise FileNotFoundError(f"PDPL corpus not found: {corpus} (see data/README.md)")
    name = retrieval_cfg["embedding_model"]
    max_len = int(retrieval_cfg["embedding_max_length"])
    texts, metadatas = parse_docx_to_texts_metadatas(corpus)
    if not texts:
        raise ValueError(f"No chunks parsed from PDPL corpus: {corpus}")
    embedder = MultilingualE5Embeddings(model_name=name, device=retrieval_cfg.get("embedding_device"),
                                       max_length=max_len, revision=retrieval_cfg.get("embedding_revision"))
    try:
        vs = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
        return Retriever(vectorstore=vs, embedder=embedder, corpus_path=str(corpus),
                         corpus_sha256=sha256_file(corpus), embedding_model=name, n_chunks=len(texts),
                         n_chunks_truncated_by_embedder=embedder.count_truncated(texts),
                         embedding_max_length=max_len, chunking=retrieval_cfg.get("chunking", "docx_heading_sections"))
    except BaseException:
        embedder.release()
        raise
