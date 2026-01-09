"""TF-IDF feature extraction for patent classification."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from src.utils.config import get_config


class TfidfFeatureExtractor:
    """Extract TF-IDF features from patent text."""

    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Optional[Tuple[int, int]] = None,
        min_df: Optional[Union[int, float]] = None,
        max_df: Optional[Union[int, float]] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize TF-IDF feature extractor.

        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.tfidf_config = self.config.tfidf

        # Use provided parameters or fall back to config
        self.max_features = max_features or self.tfidf_config.get('max_features', 10000)
        self.ngram_range = ngram_range or tuple(self.tfidf_config.get('ngram_range', [1, 2]))
        self.min_df = min_df if min_df is not None else self.tfidf_config.get('min_df', 5)
        self.max_df = max_df if max_df is not None else self.tfidf_config.get('max_df', 0.8)

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            sublinear_tf=True,  # Use logarithmic tf scaling
            strip_accents='unicode'
        )

        self._is_fitted = False

    def fit(self, texts: List[str]) -> 'TfidfFeatureExtractor':
        """
        Fit the TF-IDF vectorizer on texts.

        Args:
            texts: List of text documents

        Returns:
            Self
        """
        print(f"\nFitting TF-IDF vectorizer...")
        print(f"  max_features: {self.max_features}")
        print(f"  ngram_range: {self.ngram_range}")
        print(f"  min_df: {self.min_df}")
        print(f"  max_df: {self.max_df}")

        self.vectorizer.fit(texts)
        self._is_fitted = True

        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"  Vocabulary size: {vocab_size}")

        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF features.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF feature matrix (sparse)
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit() first.")

        features = self.vectorizer.transform(texts)
        return features

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit vectorizer and transform texts in one step.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF feature matrix (sparse)
        """
        print(f"\nFitting and transforming TF-IDF features...")
        print(f"  max_features: {self.max_features}")
        print(f"  ngram_range: {self.ngram_range}")
        print(f"  min_df: {self.min_df}")
        print(f"  max_df: {self.max_df}")

        features = self.vectorizer.fit_transform(texts)
        self._is_fitted = True

        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Feature matrix shape: {features.shape}")

        return features

    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary).

        Returns:
            List of feature names
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names.")

        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features_for_document(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a single document.

        Args:
            text: Input text
            top_n: Number of top features to return

        Returns:
            List of (feature, score) tuples
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted first.")

        # Transform the text
        tfidf_vector = self.vectorizer.transform([text])

        # Get feature names
        feature_names = self.get_feature_names()

        # Get scores for this document
        scores = tfidf_vector.toarray()[0]

        # Get top features
        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_features = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]

        return top_features

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save the vectorizer to disk.

        Args:
            filepath: Path to save the vectorizer
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted vectorizer.")

        if filepath is None:
            filepath = self.config.models.get('tfidf_vectorizer', 'models/tfidf/vectorizer.pkl')

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print(f"\nSaved TF-IDF vectorizer to {filepath}")

    def load(self, filepath: Optional[str] = None) -> 'TfidfFeatureExtractor':
        """
        Load a saved vectorizer from disk.

        Args:
            filepath: Path to load the vectorizer from

        Returns:
            Self
        """
        if filepath is None:
            filepath = self.config.models.get('tfidf_vectorizer', 'models/tfidf/vectorizer.pkl')

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")

        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self._is_fitted = True

        print(f"\nLoaded TF-IDF vectorizer from {filepath}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        return self

    @property
    def is_fitted(self) -> bool:
        """Check if vectorizer is fitted."""
        return self._is_fitted


def create_tfidf_features(
    train_texts: List[str],
    test_texts: Optional[List[str]] = None,
    config_path: str = "config.yaml"
) -> Union[csr_matrix, Tuple[csr_matrix, csr_matrix]]:
    """
    Convenience function to create TF-IDF features.

    Args:
        train_texts: Training texts
        test_texts: Test texts (optional)
        config_path: Path to configuration file

    Returns:
        TF-IDF features for train (and test if provided)
    """
    extractor = TfidfFeatureExtractor(config_path=config_path)

    X_train = extractor.fit_transform(train_texts)

    if test_texts is not None:
        X_test = extractor.transform(test_texts)
        return X_train, X_test

    return X_train
