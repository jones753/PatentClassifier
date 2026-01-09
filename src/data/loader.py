"""Data loading utilities for patent classification."""

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import get_config


class PatentDataLoader:
    """Load and split patent data from CSV files."""

    REQUIRED_COLUMNS = ['abstract', 'main_claim', 'cpc_class']

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the data loader.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.data_config = self.config.data

    def load_raw_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw patent data from CSV.

        Args:
            csv_path: Path to CSV file (if None, uses config)

        Returns:
            DataFrame with patent data

        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If required columns are missing
        """
        if csv_path is None:
            csv_path = self.data_config.get('raw_path', 'data/raw/patents.csv')

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}\n"
                f"Please place your patent data CSV at {csv_path}"
            )

        # Load CSV
        df = pd.read_csv(csv_path)

        # Validate columns
        self._validate_columns(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        print(f"Loaded {len(df)} patents from {csv_path}")
        print(f"Number of unique CPC classes: {df['cpc_class'].nunique()}")

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Required columns: {self.REQUIRED_COLUMNS}\n"
                f"Found columns: {list(df.columns)}"
            )

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in required columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        initial_count = len(df)

        # Check for missing values
        missing_counts = df[self.REQUIRED_COLUMNS].isnull().sum()
        if missing_counts.any():
            print("\nMissing values detected:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count} missing ({count/len(df)*100:.2f}%)")

        # Drop rows with any missing required columns
        df = df.dropna(subset=self.REQUIRED_COLUMNS)

        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing values")

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_seed: Optional[int] = None,
        save_splits: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets with stratification.

        Args:
            df: Input DataFrame
            train_size: Proportion for training (if None, uses config)
            val_size: Proportion for validation (if None, uses config)
            random_seed: Random seed for reproducibility (if None, uses config)
            save_splits: Whether to save splits to disk

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if train_size is None:
            train_size = self.data_config.get('train_test_split', 0.8)
        if val_size is None:
            val_size = self.data_config.get('validation_split', 0.1)
        if random_seed is None:
            random_seed = self.data_config.get('random_seed', 42)

        # Calculate test size
        test_size = 1.0 - train_size
        val_ratio = val_size / train_size  # Ratio of val to train

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=df['cpc_class']
        )

        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=train_val_df['cpc_class']
        )

        print(f"\nData split (seed={random_seed}):")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

        # Verify stratification
        print("\nClass distribution:")
        print(f"  Train: {train_df['cpc_class'].value_counts().head()}")

        if save_splits:
            self._save_splits(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        Save train, val, test splits to disk.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
        """
        processed_path = Path(self.data_config.get('processed_path', 'data/processed'))
        processed_path.mkdir(parents=True, exist_ok=True)

        # Save DataFrames
        train_df.to_csv(processed_path / 'train.csv', index=False)
        val_df.to_csv(processed_path / 'val.csv', index=False)
        test_df.to_csv(processed_path / 'test.csv', index=False)

        # Save statistics
        stats = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'num_classes': train_df['cpc_class'].nunique(),
            'class_counts': train_df['cpc_class'].value_counts().to_dict()
        }

        with open(processed_path / 'split_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nSaved splits to {processed_path}/")

    def load_splits(
        self,
        processed_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved train/val/test splits.

        Args:
            processed_path: Path to processed data directory

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if processed_path is None:
            processed_path = self.data_config.get('processed_path', 'data/processed')

        processed_path = Path(processed_path)

        train_df = pd.read_csv(processed_path / 'train.csv')
        val_df = pd.read_csv(processed_path / 'val.csv')
        test_df = pd.read_csv(processed_path / 'test.csv')

        print(f"Loaded splits from {processed_path}/")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")

        return train_df, val_df, test_df


def load_data(
    csv_path: Optional[str] = None,
    use_saved_splits: bool = False,
    config_path: str = "config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and split patent data.

    Args:
        csv_path: Path to CSV file (if None, uses config)
        use_saved_splits: Whether to load previously saved splits
        config_path: Path to configuration file

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = PatentDataLoader(config_path)

    if use_saved_splits:
        return loader.load_splits()
    else:
        df = loader.load_raw_data(csv_path)
        return loader.split_data(df)
