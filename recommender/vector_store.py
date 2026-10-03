"""ChromaDB vector store module for regional item recommendations.

Handles persistent indexing, metadata filtering, and semantic similarity search
for regional items across food, fashion, travel, and diaspora categories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resolve_torch_device() -> str:
    """Resolve the optimal compute device across CUDA, MPS, and CPU.

    Returns:
        str: Target device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RegionalVectorStore:
    """Persistent ChromaDB vector store for regional catalog items."""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "regional_catalog",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the vector store client and embedding function.

        Args:
            db_path: Directory path where ChromaDB persists data.
            collection_name: Name of the target collection.
            embedding_model_name: Hugging Face model for text embeddings.
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.device = resolve_torch_device()

        # Initialize embedding function with explicit device mapping
        try:
            self.embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model_name,
                    device=self.device,
                )
            )
            logger.info(
                "Initialized SentenceTransformerEmbeddingFunction (%s) on %s",
                embedding_model_name,
                self.device,
            )
        except Exception as exc:
            logger.warning(
                "SentenceTransformer init failed (%s), falling back to default ChromaDB embedder.",
                exc,
            )
            self.embedding_function = (
                embedding_functions.DefaultEmbeddingFunction()
            )

        # Initialize persistent ChromaDB client
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client: ClientAPI = chromadb.PersistentClient(
                path=str(self.db_path)
            )
            self._collection: Collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(
                "Connected to ChromaDB at '%s', collection '%s' (item count: %d)",
                self.db_path,
                self.collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB client: %s", exc)
            raise RuntimeError(f"ChromaDB initialization failed: {exc}") from exc

    @property
    def collection(self) -> Collection:
        """Get the active collection, refreshing if recreated by another process."""
        try:
            _ = self._collection.count()
            return self._collection
        except Exception:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            return self._collection

    def load_catalog(
        self,
        catalog_path: str = "data/regional_catalog.json",
        force_reload: bool = False,
    ) -> int:
        """Load regional catalog data from JSON and index documents in ChromaDB.

        Args:
            catalog_path: Path to the JSON catalog file.
            force_reload: Whether to clear existing records and reindex.

        Returns:
            int: Number of items successfully indexed.
        """
        path = Path(catalog_path)
        if not path.is_file():
            logger.error("Catalog file not found: %s", catalog_path)
            raise FileNotFoundError(f"Catalog file does not exist: {catalog_path}")

        try:
            with open(path, "r", encoding="utf-8") as file:
                items: list[dict[str, Any]] = json.load(file)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON catalog: %s", exc)
            raise ValueError(f"Malformed JSON in catalog: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error reading catalog: %s", exc)
            raise

        if not items:
            logger.warning("Catalog at %s is empty.", catalog_path)
            return 0

        # If already indexed and not forcing reload, return existing count
        current_count = self.collection.count()
        if current_count > 0 and not force_reload:
            logger.info(
                "Collection '%s' already contains %d records. Skipping re-indexing.",
                self.collection_name,
                current_count,
            )
            return current_count

        if force_reload and current_count > 0:
            logger.info("Force reload requested. Deleting existing records...")
            self.client.delete_collection(name=self.collection_name)
            self._collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for item in items:
            item_id = str(item.get("id", ""))
            title = str(item.get("title", ""))
            region = str(item.get("region", "")).lower()
            category = str(item.get("category", "")).lower()
            description = str(item.get("description", ""))
            tags_list = item.get("tags", [])
            tags_str = ", ".join(tags_list) if isinstance(tags_list, list) else str(tags_list)
            location = str(item.get("location", ""))
            highlights = item.get("highlights", [])
            highlights_str = "; ".join(highlights) if isinstance(highlights, list) else str(highlights)

            # Combined content text for semantic retrieval
            document_text = (
                f"Title: {title}. "
                f"Region: {region}. "
                f"Category: {category}. "
                f"Description: {description} "
                f"Tags: {tags_str}. "
                f"Highlights: {highlights_str}."
            )

            # ChromaDB metadata supports str, int, float, bool
            metadata = {
                "id": item_id,
                "title": title,
                "region": region,
                "category": category,
                "location": location,
                "tags": tags_str,
                "highlights": highlights_str,
                "description": description,
            }

            documents.append(document_text)
            metadatas.append(metadata)
            ids.append(item_id)

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Successfully indexed %d items into ChromaDB.", len(ids))
            return len(ids)
        except Exception as exc:
            logger.error("Failed to add items to ChromaDB: %s", exc)
            raise RuntimeError(f"ChromaDB indexing error: {exc}") from exc

    def retrieve(
        self,
        query: str,
        region: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant items using semantic similarity and metadata filtering.

        Args:
            query: The search or user intent text.
            region: Optional regional filter (e.g., 'telangana').
            category: Optional category filter (e.g., 'food', 'fashion', 'travel', 'diaspora').
            top_k: Number of nearest items to return.

        Returns:
            list[dict[str, Any]]: Ranked list of matching recommendations with metadata.
        """
        if not query.strip():
            logger.warning("Empty query passed to retrieve.")
            return []

        where_filter: Optional[dict[str, Any]] = None
        filter_conditions: list[dict[str, str]] = []

        if region and region.strip():
            filter_conditions.append({"region": region.strip().lower()})
        if category and category.strip():
            filter_conditions.append({"category": category.strip().lower()})

        if len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            where_filter = filter_conditions[0]

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as exc:
            logger.error("Error executing ChromaDB query: %s", exc)
            return []

        recommendations: list[dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return recommendations

        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        docs = results.get("documents", [[]])[0] if results.get("documents") else []

        for idx, item_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else None
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            document = docs[idx] if idx < len(docs) else ""

            # Convert distance to similarity confidence score (cosine distance approx: 1 - dist)
            confidence: Optional[float] = None
            if distance is not None:
                confidence = max(0.0, min(1.0, 1.0 - float(distance)))

            recommendations.append(
                {
                    "id": item_id,
                    "title": metadata.get("title", ""),
                    "region": metadata.get("region", ""),
                    "category": metadata.get("category", ""),
                    "description": metadata.get("description", ""),
                    "location": metadata.get("location", ""),
                    "tags": [t.strip() for t in metadata.get("tags", "").split(",") if t.strip()],
                    "highlights": metadata.get("highlights", ""),
                    "confidence": round(confidence, 4) if confidence is not None else None,
                    "raw_document": document,
                }
            )

        return recommendations


# Module-level convenience functions
_store_instance: Optional[RegionalVectorStore] = None


def get_vector_store(
    db_path: str = "./chroma_db",
    collection_name: str = "regional_catalog",
) -> RegionalVectorStore:
    """Singleton-style getter for the regional vector store.

    Args:
        db_path: Path to the ChromaDB directory.
        collection_name: Name of the collection.

    Returns:
        RegionalVectorStore: Initialized vector store instance.
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = RegionalVectorStore(
            db_path=db_path,
            collection_name=collection_name,
        )
    return _store_instance


def retrieve_recommendations(
    query: str,
    region: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 3,
    db_path: str = "./chroma_db",
) -> list[dict[str, Any]]:
    """Retrieve recommendations matching query, region, and category.

    Args:
        query: User text or transcribed intent.
        region: Target region (e.g., 'telangana').
        category: Target category (food, fashion, travel, diaspora).
        top_k: Number of recommendations to retrieve.
        db_path: Path to ChromaDB directory.

    Returns:
        list[dict[str, Any]]: List of recommended item dicts.
    """
    store = get_vector_store(db_path=db_path)
    return store.retrieve(
        query=query,
        region=region,
        category=category,
        top_k=top_k,
    )


if __name__ == "__main__":
    print(f"Device: {resolve_torch_device()}")
    vector_store = RegionalVectorStore()
    count = vector_store.load_catalog(catalog_path="data/regional_catalog.json", force_reload=True)
    print(f"Loaded {count} items.")

    print("\n--- Test Query 1: 'spicy traditional breakfast' (category: food) ---")
    results_food = vector_store.retrieve(
        query="spicy traditional breakfast",
        region="telangana",
        category="food",
        top_k=2,
    )
    for res in results_food:
        print(f"[{res['category'].upper()}] {res['title']} (Score: {res['confidence']}) - {res['location']}")

    print("\n--- Test Query 2: 'heritage silk clothing' (category: fashion) ---")
    results_fashion = vector_store.retrieve(
        query="heritage silk clothing",
        region="telangana",
        category="fashion",
        top_k=1,
    )
    for res in results_fashion:
        print(f"[{res['category'].upper()}] {res['title']} (Score: {res['confidence']}) - {res['location']}")

    print("\n--- Test Query 3: 'temples and historical architecture' (category: travel) ---")
    results_travel = vector_store.retrieve(
        query="temples and historical architecture",
        region="telangana",
        category="travel",
        top_k=1,
    )
    for res in results_travel:
        print(f"[{res['category'].upper()}] {res['title']} (Score: {res['confidence']}) - {res['location']}")
