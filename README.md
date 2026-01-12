# Patent Classifier project

A machine learning project for **classifying patents into CPC classes** and **finding semantically similar patents** based on their abstracts and claims. Total dataset: 3,720 patents across 5 CPC classes. 70/10/20 train/val/test split.

This project implements two complementary approaches for patent classification:

1. **TF-IDF + Logistic Regression**: word frequency-based method for fast and interpretable classification
2. **Semantic Embeddings + K-NN**: for capturing semantic relationships between patents

**Core ML Libraries**:
- `scikit-learn` (1.3.0+) - TF-IDF vectorization, Logistic Regression, SVM, Random Forest, K-NN
- `sentence-transformers` (2.2.0+) - Pre-trained semantic embeddings (all-MiniLM-L6-v2)
- `numpy` (1.24.0+) - Numerical computations
- `scipy` (1.11.0+) - Sparse matrix operations

**Data Processing**:
- `pandas` (2.0.0+) - Data loading and manipulation
- `pyyaml` (6.0+) - Configuration management

## 1. First step: getting the data from Google BigQuery

Patent data was extracted from Google's public patent dataset using BigQuery:

```sql
-- category 1: machine learning
(SELECT publication_number, 'G06N20/00' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'G06N20/00')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- category 2: Aeroplanes, helicopters
(SELECT publication_number, 'B64C39/02' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'B64C39/02')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- category 3: Dietary Supplements
(SELECT publication_number, 'A23L33/10' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'A23L33/10')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- category 4: Building Construction
(SELECT publication_number, 'E04B1/00' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'E04B1/00')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)

UNION ALL

-- category 5: Wind Turbines
(SELECT publication_number, 'F03D1/00' AS cpc_class, 
  abstract_localized[SAFE_OFFSET(0)].text AS abstract, 
  claims_localized[SAFE_OFFSET(0)].text AS main_claim
 FROM `patents-public-data.patents.publications`
 WHERE EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code = 'F03D1/00')
 AND ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
 LIMIT 1000)
```

Plan was to select 1,000 patents from each of 5 distinct CPC classes (randomly picked), but some classes had fewer available patents meeting the criteria (some patents were missing the claims (thats why in the query we have ARRAY_LENGTH(claims_localized) > 0 AND claims_localized[SAFE_OFFSET(0)].text IS NOT NULL)).

### Dataset Overview

In the end, the dataset consisted of **3,720 patents** across **5 diverse CPC classes**:

| CPC Class | Category | Total Patents | Domain |
|-----------|----------|---------------|---------|
| G06N20/00 | Machine Learning | 1,000 | Computer Science |
| B64C39/02 | Aeroplanes, helicopters | 1,000 | Aerospace |
| A23L33/10 | Dietary Supplements | 1,000 | Food Science |
| F03D1/00 | Wind Turbines | 487 | Renewable Energy |
| E04B1/00 | Building Construction | 233 | Civil Engineering |

**Data Split**:
- **Training set**: 2,604 patents (70%)
- **Validation set**: 372 patents (10%)
- **Test set**: 744 patents (20%)

## 2. Step, Preprocessing the data

After querying the data from BigQuery, the following preprocessing steps were applied:

**1. Claim Extraction (`src/data/clean_claims.py`)**:
- Extract only the first substantive claim from each patent's claim text

**2. Feature Engineering (`src/data/preprocessor.py`)**:
- Text cleaning: Remove special characters, normalize whitespace
- Combine cleaned abstract and first claim into single text field
- Label encoding: Convert CPC class strings to numeric labels
- Stratified splitting: 70% train, 10% validation, 20% test (maintains class distribution)

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


### Embedding Approach

```bash
python scripts/train_embedding.py
```

**Training Pipeline**:
1. Generate 384-dim embeddings using `all-MiniLM-L6-v2` (sentence-transformers)
2. Train K-Nearest Neighbors classifier (K=5)
3. Evaluate on validation set
4. Save model and embeddings to `models/embeddings/`

## 4. Results
### TF-IDF + Logistic Regression Performance

Overall Metrics:

-  Accuracy: 95.16% - Correctly classified 354 out of 372 validation patents
-  F1-Score (macro): 94.29% - Balanced performance across all classes
- Recall (weighted): 95.16% - High coverage of actual positive cases
-  Precision (weighted): 95.29% - High confidence in predictions

**What does these metrics mean**:
- **Accuracy**: "How many did I get right overall?" → 95.16%
- **Precision**: "When I say it's class X, how often am I correct?" → 95.29%
- **Recall**: "How many of the actual class X patents did I find?" → 95.16%
- **F1-Score**: "Balance between precision and recall" → 94.29%

#### Per-Class Performance:

| CPC Class | Category | Precision | Recall | F1-Score | Validation Samples |
|-----------|----------|-----------|--------|----------|-------------------|
| A23L33/10 | Dietary Supplements | 100% | 100% | 100% | 100 |
| G06N20/00 | Machine Learning | 95% | 96% | 96% | 100 |
| F03D1/00 | Wind Turbines | 96% | 92% | 94% | 49 |
| B64C39/02 | Aeroplanes, helicopters | 90% | 94% | 92% | 100 |
| E04B1/00 | Building Construction | 100% | 83% | 90% | 23 |

**Observations**:
- **Dietary Supplements** (A23L33/10): Perfect 100% classification, probably because it has highly distinctive technical vocabulary compared to other classes.
- **Building Construction** (E04B1/00): Perfect precision (100%) but lower recall (83%) due to smallest sample size (only 23 validation patents)

### Embedding + K-NN Performance
#### Per-Class Performance (Embeddings):

| CPC Class | Category | Precision | Recall | F1-Score | Validation Samples |
|-----------|----------|-----------|--------|----------|-------------------|
| A23L33/10 | Dietary Supplements | 100% | 100% | 100% | 100 |
| G06N20/00 | Machine Learning | **98%** | **92%** | 95% | 100 |
| F03D1/00 | Wind Turbines | **98%** | **96%** | **97%** | 49 |
| B64C39/02 | Aeroplanes, helicopters | 90% | **97%** | **93%** | 100 |
| E04B1/00 | Building Construction | **95%** | **91%** | **93%** | 23 |


**Why Embeddings Perform Better**:
- **Semantic understanding**: Captures meaning beyond exact word matches (e.g., "construct" ≈ "build")
- **Better generalization**: Works better with smaller classes (Building Construction improved most)
- **Transfer learning**: Pre-trained on massive text corpora

## What I learned / Skills demonstrated

- **Machine Learning**: End-to-end ML pipeline for text classification
- **NLP & Text Processing**: TF-IDF and semantic embeddings for patent classification
- **Data Engineering**: SQL/BigQuery data extraction
- **scikit-learn**
- **Model Evaluation**: Comprehensive metrics analysis and model comparison
