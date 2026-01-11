# Patent Classifier - Technical Overview

A machine learning system that automatically classifies patents into CPC categories based on their abstracts and claims. (trained on a dataset of ~5,000 patents across 5 CPC classes)

## What This Project Does

This project implements two complementary approaches for patent classification:

1. **TF-IDF + Logistic Regression**: word frequency-based method for fast and interpretable classification
2. **Semantic Embeddings + K-NN**: for capturing semantic relationships between patents



## 1. First step: getting the data from Google BigQuery

Patent data was extracted from Google's public patent dataset using BigQuery:

```sql
-- Kategoria 1: Koneoppiminen
(SELECT publication_number, 'G06N20/00' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'G06N20/00')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- Kategoria 2: Droonit
(SELECT publication_number, 'B64C39/02' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'B64C39/02')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- Kategoria 3: Ravintolisät
(SELECT publication_number, 'A23L33/10' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'A23L33/10')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- Kategoria 4: Rakentaminen
(SELECT publication_number, 'E04B1/00' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'E04B1/00')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- Kategoria 5: Tuulivoima
(SELECT publication_number, 'F03D1/00' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'F03D1/00')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)
```

Plan was to select 1,000 patents from each of 5 distinct CPC classes (randomly picked), but some classes had fewer available patents meeting the criteria (Multiple patents didnt were missing the claims (thats why in the query we have ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL)).

### Dataset Overview

In the end, training dataset consisted of **~3,500 patents** across **5 diverse CPC classes**:

| CPC Class | Category | Patents | Domain |
|-----------|----------|---------|---------|
| G06N20/00 | Machine Learning | 1,000 | Computer Science |
| B64C39/02 | Drones/UAVs | 1,000 | Aerospace |
| A23L33/10 | Dietary Supplements | 1,000 | Food Science |
| E04B1/00 | Building Construction | 1,000 | Civil Engineering |
| F03D1/00 | Wind Turbines | 1,000 | Renewable Energy |


## 2. Step, Preprocessing the data

After querying the data from BigQuery, the following preprocessing steps were applied:

**1. Claim Extraction (`data/clean_claims.py`)**:
- Extract only the first substantive claim from each patent's claim text

**2. Feature Engineering (`src/data/preprocessor.py`)**:
- Text cleaning: Remove special characters, normalize whitespace
- Combine cleaned abstract and first claim into single text field
- Label encoding: Convert CPC class strings to numeric labels
- Stratified splitting: 80% train, 10% validation, 10% test (maintains class distribution)

Processed data is saved to `data/processed/` for reproducible experiments.

## 3. Step, Model Training

Training using 2 different approaches:

### TF-IDF Approach

```bash
python scripts/train_tfidf.py
```

**Training Pipeline**:
1. Extract TF-IDF features (10K max features, 1-2 grams)
2. Train Logistic Regression classifier with L2 regularization
3. Evaluate on validation set
4. Save model to `models/tfidf/`

**Performance**: ~85% accuracy, very fast inference

### Embedding Approach

```bash
python scripts/train_embedding.py
```

**Training Pipeline**:
1. Generate 384-dim embeddings using `all-MiniLM-L6-v2` (sentence-transformers)
2. Train K-Nearest Neighbors classifier (K=5)
3. Evaluate on validation set
4. Save model and embeddings to `models/embeddings/`

**Performance**: ~82% accuracy, enables semantic similarity search

## Key Technical Decisions

- **TF-IDF**: Chosen for speed and interpretability; ideal for production classification
- **Sentence-Transformers**: Pre-trained model provides strong semantic understanding without fine-tuning
- **K-NN**: Simple but effective for embedding-based classification and similarity search
- **Stratified Splitting**: Ensures balanced class representation across train/val/test sets

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── features/       # TF-IDF and embedding feature extraction
├── models/         # Classifier implementations
└── utils/          # Configuration and evaluation utilities

scripts/            # Training and inference pipelines
data/               # Raw and processed datasets
models/             # Saved model artifacts
```

## 4. Results

## 5. Future Work

