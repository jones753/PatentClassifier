"""Sentence embedding generation for patent classification."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.config import get_config


class EmbeddingGenerator:
    """Generate semantic embeddings using sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of sentence-transformer model
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.embeddings_config = self.config.embeddings

        # Get model name from config if not provided
        if model_name is None:
            model_name = self.embeddings_config.get('model_name', 'all-MiniLM-L6-v2')

        self.model_name = model_name
        self.batch_size = self.embeddings_config.get('batch_size', 32)

        # Load sentence transformer model
        print(f"\nLoading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.embedding_dim}")

    def generate(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text documents
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings (for cosine similarity)

        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        if show_progress:
            print(f"\nGenerating embeddings for {len(texts)} documents...")

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        if show_progress:
            print(f"  Generated embeddings shape: {embeddings.shape}")

        return embeddings

    def generate_single(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            normalize: Whether to normalize embedding

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embedding

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity (assuming normalized embeddings)
            return np.dot(embedding1, embedding2)

        elif metric == "euclidean":
            # Euclidean distance (convert to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)

        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'.")

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between a query and multiple embeddings.

        Args:
            query_embedding: Query embedding vector
            embeddings: Matrix of embeddings (n_samples, embedding_dim)
            metric: Similarity metric

        Returns:
            Array of similarity scores
        """
        if metric == "cosine":
            # Cosine similarity (assuming normalized embeddings)
            similarities = np.dot(embeddings, query_embedding)
            return similarities

        elif metric == "euclidean":
            # Euclidean distance
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            similarities = 1.0 / (1.0 + distances)
            return similarities

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Optional[str] = None
    ) -> None:
        """
        Save embeddings to disk.

        Args:
            embeddings: Embedding matrix
            filepath: Path to save embeddings
        """
        if filepath is None:
            filepath = self.config.models.get('embeddings_data', 'models/embeddings/embeddings.npy')

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.save(filepath, embeddings)
        print(f"\nSaved embeddings to {filepath}")
        print(f"  Shape: {embeddings.shape}")

    def load_embeddings(
        self,
        filepath: Optional[str] = None
    ) -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            filepath: Path to load embeddings from

        Returns:
            Embedding matrix
        """
        if filepath is None:
            filepath = self.config.models.get('embeddings_data', 'models/embeddings/embeddings.npy')

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")

        embeddings = np.load(filepath)
        print(f"\nLoaded embeddings from {filepath}")
        print(f"  Shape: {embeddings.shape}")

        return embeddings


def generate_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Convenience function to generate embeddings.

    Args:
        texts: List of text documents
        model_name: Sentence-transformer model name
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        Embedding matrix
    """
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings
