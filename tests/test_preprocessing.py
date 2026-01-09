"""Unit tests for preprocessing functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest

from src.data.preprocessor import (
    TextPreprocessor,
    preprocess_text,
    combine_patent_fields,
)


class TestTextPreprocessing:
    """Test text preprocessing functions."""

    def test_preprocess_text_lowercase(self):
        """Test lowercase conversion."""
        text = "This Is A TEST"
        result = preprocess_text(text, lowercase=True)
        assert result == "this is a test"

    def test_preprocess_text_no_lowercase(self):
        """Test without lowercase conversion."""
        text = "This Is A TEST"
        result = preprocess_text(text, lowercase=False)
        assert result == "This Is A TEST"

    def test_preprocess_text_whitespace(self):
        """Test whitespace normalization."""
        text = "Too    much    space"
        result = preprocess_text(text, lowercase=True)
        assert result == "too much space"

    def test_preprocess_text_empty(self):
        """Test empty string handling."""
        result = preprocess_text("", lowercase=True)
        assert result == ""

    def test_preprocess_text_none(self):
        """Test None handling."""
        result = preprocess_text(None, lowercase=True)
        assert result == ""

    def test_combine_patent_fields(self):
        """Test combining abstract and main claim."""
        abstract = "This is an abstract"
        main_claim = "This is a claim"
        result = combine_patent_fields(abstract, main_claim, lowercase=True)
        assert "abstract" in result
        assert "claim" in result
        assert result == "this is an abstract this is a claim"

    def test_combine_patent_fields_with_whitespace(self):
        """Test combining with extra whitespace."""
        abstract = "  Abstract  with  spaces  "
        main_claim = "  Claim  with  spaces  "
        result = combine_patent_fields(abstract, main_claim, lowercase=True)
        assert "  " not in result  # No double spaces
        assert result.strip() == result  # No leading/trailing spaces


class TestTextPreprocessor:
    """Test TextPreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized."""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
        assert preprocessor.lowercase is True

    def test_preprocess_single_text(self):
        """Test preprocessing a single text."""
        preprocessor = TextPreprocessor()
        text = "Test TEXT with CAPS"
        result = preprocessor.preprocess_text(text)
        assert result == "test text with caps"

    def test_combine_patent_text_method(self):
        """Test combining patent text method."""
        preprocessor = TextPreprocessor()
        abstract = "Abstract text"
        main_claim = "Claim text"
        result = preprocessor.combine_patent_text(abstract, main_claim)
        assert "abstract" in result
        assert "claim" in result

    def test_preprocess_dataframe(self):
        """Test preprocessing a DataFrame."""
        preprocessor = TextPreprocessor()

        # Create sample DataFrame
        df = pd.DataFrame({
            'abstract': ['Abstract ONE', 'Abstract TWO'],
            'main_claim': ['Claim ONE', 'Claim TWO'],
            'cpc_class': ['A', 'B']
        })

        result_df = preprocessor.preprocess_dataframe(df, text_column='text')

        assert 'text' in result_df.columns
        assert len(result_df) == 2
        assert 'abstract one' in result_df['text'].iloc[0]
        assert 'claim one' in result_df['text'].iloc[0]

    def test_preprocess_dataframe_removes_empty(self):
        """Test that empty texts are removed."""
        preprocessor = TextPreprocessor()

        # Create DataFrame with empty text
        df = pd.DataFrame({
            'abstract': ['Good abstract', ''],
            'main_claim': ['Good claim', ''],
            'cpc_class': ['A', 'B']
        })

        result_df = preprocessor.preprocess_dataframe(df, text_column='text')

        # Should only have one row (the valid one)
        assert len(result_df) == 1
        assert result_df['cpc_class'].iloc[0] == 'A'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self):
        """Test with very long text."""
        long_text = "word " * 10000
        result = preprocess_text(long_text, lowercase=True)
        assert len(result) > 0
        assert result.startswith("word")

    def test_special_characters(self):
        """Test with special characters."""
        text = "Text with @#$% special chars!"
        result = preprocess_text(text, lowercase=True)
        assert result == "text with @#$% special chars!"

    def test_unicode_text(self):
        """Test with unicode characters."""
        text = "Café résumé naïve"
        result = preprocess_text(text, lowercase=True)
        assert "café" in result

    def test_newlines_and_tabs(self):
        """Test with newlines and tabs."""
        text = "Line one\nLine two\tTabbed"
        result = preprocess_text(text, lowercase=True)
        assert "\n" not in result
        assert "\t" not in result
        assert "line one line two tabbed" == result


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
