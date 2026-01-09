# Patent Data Directory

This directory contains patent data for training and testing the classifier.

## Directory Structure

```
data/
├── raw/
│   └── patents.csv         # Your original patent data (place here)
├── processed/
│   ├── train.csv           # Training set (auto-generated)
│   ├── val.csv             # Validation set (auto-generated)
│   ├── test.csv            # Test set (auto-generated)
│   └── split_stats.json    # Statistics about the split (auto-generated)
└── README.md               # This file
```

## Required CSV Format

Your patent data CSV must be placed at `data/raw/patents.csv` and contain these columns:

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `abstract` | Patent abstract text | "A method for processing data using neural networks..." |
| `main_claim` | Patent main claim text | "A device comprising a processor configured to..." |
| `cpc_class` | CPC classification code | "H04L29/06" or "G06F21/62" |

### Example CSV

```csv
abstract,main_claim,cpc_class
"A method for secure data transmission over a network using encryption techniques...","A device comprising a processor configured to encrypt data before transmission...","H04L29/06"
"An apparatus for processing images using convolutional neural networks...","A system including an image sensor and a neural network processor...","G06K9/62"
"A system for managing user authentication with biometric data...","A method comprising capturing biometric data and comparing...","G06F21/32"
```

## Data Requirements

### Minimum Dataset Size

- **Minimum**: 500 patents with at least 10 unique CPC classes
- **Recommended**: 5,000+ patents for better model performance
- **Ideal**: Balanced class distribution

### Data Quality Checklist

- [ ] CSV file has correct column names: `abstract`, `main_claim`, `cpc_class`
- [ ] No missing values in required columns
- [ ] Abstract and main claim text are non-empty
- [ ] CPC classes are consistent (same class uses same label)
- [ ] Text encoding is UTF-8
- [ ] No duplicate patents (optional, but recommended)

## Data Sources

You can obtain patent data from:

1. **USPTO (United States Patent and Trademark Office)**
   - PatentsView: https://patentsview.org/
   - Bulk data downloads available

2. **EPO (European Patent Office)**
   - Open Patent Services: https://www.epo.org/searching-for-patents/data.html

3. **Google Patents Public Datasets**
   - BigQuery: https://console.cloud.google.com/marketplace/browse?q=google%20patents

4. **WIPO (World Intellectual Property Organization)**
   - Patentscope: https://patentscope.wipo.int/

## Preprocessing

The system automatically:

1. **Combines** abstract and main_claim into a single text field
2. **Cleans** text (lowercase, remove excess whitespace)
3. **Splits** data into train/val/test sets (80/10/10 by default)
4. **Stratifies** split to maintain class distribution
5. **Removes** rows with missing values

## Data Splits

After first training run, you'll find:

- `processed/train.csv`: 80% of data (by default)
- `processed/val.csv`: 10% of data
- `processed/test.csv`: 10% of data
- `processed/split_stats.json`: Statistics about the split

To reuse these splits (for reproducibility):

```bash
python scripts/train_tfidf.py --use-saved-splits
```

## Custom Data Path

If your CSV is in a different location:

```bash
python scripts/train_tfidf.py --data-path path/to/your/patents.csv
```

## Sample Data

To test the system without real patent data, you can create a small sample CSV:

```python
import pandas as pd

# Create sample data
data = {
    'abstract': [
        'A method for processing data using machine learning algorithms',
        'An apparatus for image recognition using neural networks',
        'A system for secure communication over networks',
        'A device for biometric authentication using fingerprints',
        'A method for data compression using statistical models'
    ] * 100,  # Repeat to get 500 samples
    'main_claim': [
        'A processor configured to execute machine learning algorithms',
        'An image sensor coupled to a neural network processor',
        'A transmitter configured to encrypt data before transmission',
        'A scanner configured to capture fingerprint data',
        'A compressor configured to reduce data size'
    ] * 100,
    'cpc_class': [
        'G06N3/08',
        'G06K9/62',
        'H04L29/06',
        'G06F21/32',
        'H03M7/30'
    ] * 100
}

df = pd.DataFrame(data)
df.to_csv('data/raw/patents.csv', index=False)
print(f"Created sample data: {len(df)} patents, {df['cpc_class'].nunique()} classes")
```

## Data Privacy

If using real patent data:

- Ensure you have appropriate rights to use the data
- Patents are generally public information
- Check licensing terms for specific data sources
- Remove any personally identifiable information if present

## Troubleshooting

### "CSV file not found"

Solution: Ensure file is at `data/raw/patents.csv` or specify `--data-path`

### "Missing required columns"

Solution: Check that CSV has columns named exactly: `abstract`, `main_claim`, `cpc_class`

### "Removed X rows with missing values"

Solution: Check your CSV for empty cells in required columns. Either fill them or accept removal.

### Imbalanced classes

If you see warnings about imbalanced classes:

- Consider collecting more data for underrepresented classes
- Use stratified splitting (already default)
- Try different evaluation metrics (F1 instead of accuracy)

## Statistics

After loading data, the system will show:

```
Loaded 5000 patents from data/raw/patents.csv
Number of unique CPC classes: 25

Data split (seed=42):
  Train: 4000 samples (80.0%)
  Val:   500 samples (10.0%)
  Test:  500 samples (10.0%)

Class distribution:
  Train: H04L29/06    425
         G06F21/62    387
         G06K9/62     356
         ...
```

This helps you understand your dataset before training.
