"""Text preprocessing utilities for patent data."""

import re
from typing import Optional, Union

import pandas as pd

from src.utils.config import get_config


class TextPreprocessor:
    """Preprocess patent text for classification."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the preprocessor.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.preprocessing_config = self.config.preprocessing

        self.lowercase = self.preprocessing_config.get('lowercase', True)
        self.combine_fields = self.preprocessing_config.get('combine_fields', True)

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize a single text string.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if pd.isna(text) or text is None:
            return ""

        # Convert to string if needed
        text = str(text)

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Convert to lowercase if configured
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def combine_patent_text(
        self,
        abstract: str,
        main_claim: str,
        separator: str = " "
    ) -> str:
        """
        Combine patent abstract and main claim.

        Args:
            abstract: Patent abstract text
            main_claim: Patent main claim text
            separator: Separator between abstract and claim

        Returns:
            Combined text
        """
        # Preprocess both fields
        abstract = self.preprocess_text(abstract)
        main_claim = self.preprocess_text(main_claim)

        # Combine with separator
        combined = f"{abstract}{separator}{main_claim}"

        return combined.strip()

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess patent data in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name for the output text column
            inplace: Whether to modify DataFrame in place

        Returns:
            DataFrame with preprocessed text column
        """
        if not inplace:
            df = df.copy()

        if self.combine_fields:
            # Combine abstract and main_claim
            if 'abstract' in df.columns and 'main_claim' in df.columns:
                df[text_column] = df.apply(
                    lambda row: self.combine_patent_text(
                        row['abstract'],
                        row['main_claim']
                    ),
                    axis=1
                )
                print(f"Combined 'abstract' and 'main_claim' into '{text_column}'")
            else:
                raise ValueError(
                    "DataFrame must have 'abstract' and 'main_claim' columns "
                    "when combine_fields is True"
                )
        else:
            # Just preprocess existing text column
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in DataFrame")
            df[text_column] = df[text_column].apply(self.preprocess_text)

        # Remove empty texts
        initial_count = len(df)
        df = df[df[text_column].str.len() > 0]
        removed = initial_count - len(df)

        if removed > 0:
            print(f"Removed {removed} rows with empty text after preprocessing")

        return df


def preprocess_text(
    text: str,
    lowercase: bool = True
) -> str:
    """
    Simple standalone text preprocessing function.

    Args:
        text: Input text
        lowercase: Whether to convert to lowercase

    Returns:
        Preprocessed text
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)
    text = ' '.join(text.split())

    if lowercase:
        text = text.lower()

    return text.strip()


def combine_patent_fields(
    abstract: str,
    main_claim: str,
    lowercase: bool = True
) -> str:
    """
    Standalone function to combine patent abstract and claim.

    Args:
        abstract: Patent abstract
        main_claim: Patent main claim
        lowercase: Whether to convert to lowercase

    Returns:
        Combined and preprocessed text
    """
    abstract = preprocess_text(abstract, lowercase)
    main_claim = preprocess_text(main_claim, lowercase)

    return f"{abstract} {main_claim}".strip()
