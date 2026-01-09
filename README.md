# Patent Classifier

A machine learning project for classifying patents into CPC (Cooperative Patent Classification) classes and finding similar patents using two approaches:

1. **TF-IDF + Traditional ML** (Logistic Regression, SVM, Random Forest)
2. **Semantic Embeddings** (Sentence-Transformers + K-NN)

## Features

- Patent classification into CPC classes
- Semantic similarity search for patents
- Two complementary ML approaches for comparison
- Clean, modular, and easy-to-understand codebase
- End-to-end training and inference pipelines

## Project Structure

```
PatentClassifier/
├── data/
│   ├── raw/                    # Place your CSV data here
│   └── processed/              # Auto-generated train/val/test splits
├── models/
│   ├── tfidf/                  # Trained TF-IDF models
│   └── embeddings/             # Trained embedding models
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── features/               # Feature extraction (TF-IDF, embeddings)
│   ├── models/                 # Classification models
│   └── utils/                  # Configuration and evaluation
├── scripts/
│   ├── train_tfidf.py          # Train TF-IDF model
│   ├── train_embedding.py      # Train embedding model
│   ├── predict.py              # Classify new patents
│   └── find_similar.py         # Find similar patents
└── config.yaml                 # Configuration file
```

## Installation

### 1. Clone or download this project

```bash
cd PatentClassifier
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

### CSV Format

Your patent data CSV should have these columns:

- `abstract`: Patent abstract text
- `main_claim`: Patent main claim text
- `cpc_class`: CPC classification label (e.g., "H04L29/06", "G06F21/62")

Example:

```csv
abstract,main_claim,cpc_class
"A method for secure data transmission...","A device comprising...","H04L29/06"
"An apparatus for image processing...","A system including...","G06K9/62"
```

### Place Your Data

Put your CSV file in `data/raw/patents.csv`, or specify a custom path when training.

## Quick Start

### 1. Train TF-IDF Model

```bash
python scripts/train_tfidf.py
```

This will:
- Load and split your data (80% train, 10% val, 10% test)
- Preprocess text (combine abstract + main claim)
- Extract TF-IDF features
- Train a Logistic Regression classifier
- Evaluate on validation set
- Save the model to `models/tfidf/`

### 2. Train Embedding Model

```bash
python scripts/train_embedding.py
```

This will:
- Load and split your data
- Generate semantic embeddings using sentence-transformers
- Train a K-NN classifier
- Evaluate on validation set
- Save the model to `models/embeddings/`

### 3. Classify New Patents

```bash
python scripts/predict.py \
    --abstract "A method for processing data using machine learning..." \
    --main-claim "A system comprising a processor configured to..." \
    --model both
```

Or use interactive mode:

```bash
python scripts/predict.py --interactive
```

### 4. Find Similar Patents

```bash
python scripts/find_similar.py \
    --abstract "A method for image recognition..." \
    --main-claim "An apparatus comprising..." \
    --top-k 5
```

Or use interactive mode:

```bash
python scripts/find_similar.py --interactive
```

## Configuration

Edit `config.yaml` to customize:

### Data Settings
```yaml
data:
  raw_path: "data/raw/patents.csv"
  train_test_split: 0.8
  validation_split: 0.1
  random_seed: 42
```

### TF-IDF Settings
```yaml
tfidf:
  max_features: 10000
  ngram_range: [1, 2]
  classifier: "logistic_regression"  # or "svm", "random_forest"
```

### Embedding Settings
```yaml
embeddings:
  model_name: "all-MiniLM-L6-v2"  # Fast and efficient
  batch_size: 32
  top_k: 5
```

## Advanced Usage

### Training Options

#### TF-IDF Training

```bash
# Use a different classifier
python scripts/train_tfidf.py --model-type svm

# Show top features per class
python scripts/train_tfidf.py --show-features

# Evaluate on test set
python scripts/train_tfidf.py --eval-test

# Use saved data splits
python scripts/train_tfidf.py --use-saved-splits
```

#### Embedding Training

```bash
# Use a different sentence-transformer model
python scripts/train_embedding.py --model-name all-mpnet-base-v2

# Adjust K for K-NN
python scripts/train_embedding.py --k 10

# Show confidence scores
python scripts/train_embedding.py --show-confidence

# Save embedding matrices
python scripts/train_embedding.py --save-embeddings
```

### Prediction Options

```bash
# Use only TF-IDF model
python scripts/predict.py --model tfidf --abstract "..." --main-claim "..."

# Use only embedding model
python scripts/predict.py --model embedding --abstract "..." --main-claim "..."

# Show top-3 predictions
python scripts/predict.py --top-k 3 --abstract "..." --main-claim "..."
```

## Model Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **TF-IDF + ML** | Fast training/inference<br>Interpretable features<br>Works well with limited data<br>Low computational requirements | Doesn't capture semantics<br>Bag-of-words limitation<br>No similarity search |
| **Embeddings + K-NN** | Captures semantic meaning<br>Enables similarity search<br>Better generalization<br>Transfer learning benefits | Slower inference<br>Requires more memory<br>Less interpretable |

## Evaluation Metrics

Both approaches provide:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class and macro-averaged
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis
- **Classification Report**: Per-class metrics

## Example Output

### Training

```
TF-IDF Patent Classifier Training Pipeline
==========================================================

[1/6] Loading data...
Loaded 5000 patents from data/raw/patents.csv
Number of unique CPC classes: 25

[2/6] Preprocessing text...
Combined 'abstract' and 'main_claim' into 'text'

[3/6] Extracting TF-IDF features...
  max_features: 10000
  ngram_range: (1, 2)
  Vocabulary size: 8543
  Feature matrix shape: (4000, 8543)

[4/6] Training classifier...
  Training samples: 4000
  Features: 8543
  Classes: 25

[5/6] Evaluating on validation set...
Accuracy:          0.8520
F1 (macro):        0.8301
```

### Prediction

```
TF-IDF Model Prediction
==========================================================

Predicted CPC Class: H04L29/06

Top 3 predictions:
  1. H04L29/06          (confidence: 0.9234)
  2. H04L29/08          (confidence: 0.0456)
  3. G06F21/62          (confidence: 0.0189)
```

## Testing

Run basic tests:

```bash
pytest tests/
```

## Requirements

- Python 3.8+
- scikit-learn >= 1.3.0
- PyTorch >= 2.0.0
- sentence-transformers >= 2.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0

See `requirements.txt` for full list.

## Tips for Best Results

### Data Quality
- Ensure clean, consistent CPC class labels
- Remove or fix patents with missing abstracts/claims
- Aim for balanced class distribution (or use stratified splitting)
- Minimum recommended: 500 patents, 10+ classes

### Hyperparameter Tuning
- For TF-IDF: Adjust `max_features`, `ngram_range`, classifier `C` parameter
- For embeddings: Try different K values, experiment with model names
- Use validation set for tuning, test set for final evaluation

### Model Selection
- Use TF-IDF for: Speed, interpretability, limited resources
- Use embeddings for: Semantic search, better generalization, similarity features

## Troubleshooting

### "CSV file not found"
- Ensure your CSV is at `data/raw/patents.csv`
- Or specify custom path: `--data-path path/to/your.csv`

### "Model not found"
- Train the model first using the training scripts
- Check that models are saved in `models/tfidf/` or `models/embeddings/`

### Low accuracy
- Check data quality and class balance
- Increase training data size
- Tune hyperparameters
- Try different model types

### Out of memory
- Reduce `batch_size` in config.yaml
- Use smaller embedding model (e.g., "all-MiniLM-L6-v2")
- Reduce `max_features` for TF-IDF

## License

This project is for educational purposes.

## Contributing

Feel free to extend this project with:
- Additional classification algorithms
- Cross-validation
- Ensemble methods
- Fine-tuning sentence-transformers
- Web interface
- API endpoints

## Acknowledgments

- scikit-learn for traditional ML algorithms
- sentence-transformers for semantic embeddings
- Hugging Face for pre-trained models
