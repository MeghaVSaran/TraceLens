"""
Vector Index — ChromaDB wrapper for dense retrieval.

See docs/2_system_architecture.md §5 for spec.
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

import chromadb
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

COLLECTION_NAME = "debugaid_code_chunks"


class IndexNotFoundError(Exception):
    """Raised when querying an index that has not been built."""
    pass


class VectorIndex:
    """Persistent ChromaDB vector index for code chunks.

    Stores 768-dim embeddings alongside chunk metadata and supports
    cosine-similarity nearest-neighbour queries.
    """

    def __init__(self, persist_dir: Path):
        """Initialise the index wrapper.

        The ChromaDB client is created immediately but no collection
        is loaded until :meth:`build` or :meth:`query` is called.

        Args:
            persist_dir: Directory where ChromaDB stores its data.
                         Typically: ``repo_root/.debugaid/chroma/``
        """
        self._persist_dir = Path(persist_dir)
        self._client: chromadb.ClientAPI = chromadb.PersistentClient(
            path=str(self._persist_dir)
        )
        self._collection: Optional[chromadb.Collection] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, chunks: List, embeddings: List) -> None:
        """Build or overwrite the index from parallel chunk and vector lists."""
        self.reset()
        self.add(chunks, embeddings)
        logger.info("Built index: %d chunks in %s", len(chunks), self._persist_dir)

    def reset(self) -> None:
        """Delete and recreate the collection before an index rebuild."""
        try:
            self._client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List, embeddings: List) -> None:
        """Add one bounded batch of chunks to the current collection."""
        if self._collection is None:
            self.reset()
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")

        ids: List[str] = []
        metadatas: List[Dict] = []
        emb_lists: List[List[float]] = []

        for chunk, emb in zip(chunks, embeddings):
            ids.append(chunk.chunk_id)
            metadatas.append({
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "function_name": chunk.function_name,
                "symbol_name": getattr(chunk, "symbol_name", ""),
                "class_name": getattr(chunk, "class_name", ""),
                "namespace": getattr(chunk, "namespace", ""),
                "signature": getattr(chunk, "signature", ""),
                "start_line": chunk.start_line,
            })
            if isinstance(emb, np.ndarray):
                emb_lists.append(emb.tolist())
            else:
                emb_lists.append(list(emb))

        batch_size = 5000
        for start in tqdm(range(0, len(ids), batch_size), desc="Upserting chunks", unit="batch"):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=emb_lists[start:end],
                metadatas=metadatas[start:end],
            )

    def query(
        self,
        log_embedding: List[float],
        top_k: int = 20,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Query index for nearest neighbours.

        Args:
            log_embedding: 768-dim query vector.
            top_k: Number of results to return.
            where: Optional ChromaDB ``where`` filter dict for metadata
                filtering, e.g. ``{"file_path": {"$in": ["a.cpp", "b.cpp"]}}``.

        Returns:
            List of dicts, each containing:
            ``{chunk_id, file_path, function_name, start_line, score}``
            where ``score = 1 - cosine_distance`` (higher is better).

        Raises:
            IndexNotFoundError: If the collection has not been built.
        """
        self._ensure_collection()

        # Convert numpy to list if needed.
        if isinstance(log_embedding, np.ndarray):
            log_embedding = log_embedding.tolist()

        query_kwargs = {
            "query_embeddings": [log_embedding],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        # Unpack ChromaDB's nested list structure.
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        output: List[Dict] = []
        for chunk_id, meta, dist in zip(ids, metadatas, distances):
            output.append({
                "chunk_id": chunk_id,
                "file_path": meta["file_path"],
                "function_name": meta["function_name"],
                "symbol_name": meta.get("symbol_name", ""),
                "class_name": meta.get("class_name", ""),
                "namespace": meta.get("namespace", ""),
                "signature": meta.get("signature", ""),
                "start_line": meta["start_line"],
                "score": 1.0 - dist,
            })

        return output

    def exists(self) -> bool:
        """Return True if the collection exists and has documents."""
        try:
            col = self._client.get_collection(COLLECTION_NAME)
            return col.count() > 0
        except (ValueError, Exception):
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Load the collection or raise ``IndexNotFoundError``."""
        if self._collection is not None:
            return
        try:
            self._collection = self._client.get_collection(COLLECTION_NAME)
        except (ValueError, Exception) as exc:
            raise IndexNotFoundError(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Run 'build()' first.  ({exc})"
            ) from exc
