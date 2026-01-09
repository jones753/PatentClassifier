"""Train embedding-based patent classifier."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_data
from src.data.preprocessor import TextPreprocessor
from src.features.embedding_features import EmbeddingGenerator
from src.models.embedding_classifier import EmbeddingClassifier
from src.utils.evaluation import ModelEvaluator, print_classification_report


def main(args):
    """Main training pipeline."""
    print("="*60)
    print("Embedding-Based Patent Classifier Training Pipeline")
    print("="*60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    try:
        train_df, val_df, test_df = load_data(
            csv_path=args.data_path,
            use_saved_splits=args.use_saved_splits,
            config_path=args.config
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure your patent data CSV is placed in the correct location.")
        print("Expected format: columns 'abstract', 'main_claim', 'cpc_class'")
        return

    # 2. Preprocess text
    print("\n[2/6] Preprocessing text...")
    preprocessor = TextPreprocessor(config_path=args.config)

    train_df = preprocessor.preprocess_dataframe(train_df, text_column='text')
    val_df = preprocessor.preprocess_dataframe(val_df, text_column='text')
    test_df = preprocessor.preprocess_dataframe(test_df, text_column='text')

    # Extract text and labels
    X_train_text = train_df['text'].tolist()
    y_train = train_df['cpc_class'].values

    X_val_text = val_df['text'].tolist()
    y_val = val_df['cpc_class'].values

    X_test_text = test_df['text'].tolist()
    y_test = test_df['cpc_class'].values

    # 3. Generate embeddings
    print("\n[3/6] Generating embeddings...")
    embedding_generator = EmbeddingGenerator(
        model_name=args.model_name,
        config_path=args.config
    )

    print("\nGenerating training embeddings...")
    X_train = embedding_generator.generate(X_train_text, show_progress=True)

    print("\nGenerating validation embeddings...")
    X_val = embedding_generator.generate(X_val_text, show_progress=True)

    print("\nGenerating test embeddings...")
    X_test = embedding_generator.generate(X_test_text, show_progress=True)

    print(f"\nEmbedding shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # 4. Train classifier
    print("\n[4/6] Training K-NN classifier...")
    classifier = EmbeddingClassifier(
        k=args.k,
        metric=args.metric,
        config_path=args.config
    )
    classifier.fit(X_train, y_train, texts=X_train_text)

    # 5. Evaluate on validation set
    print("\n[5/6] Evaluating on validation set...")
    evaluator = ModelEvaluator()

    y_val_pred = classifier.predict(X_val)
    val_metrics = evaluator.evaluate(y_val, y_val_pred, verbose=True)

    # Print detailed classification report
    unique_classes = sorted(train_df['cpc_class'].unique())
    print_classification_report(y_val, y_val_pred, target_names=unique_classes)

    # Show confidence scores
    if args.show_confidence:
        print("\n" + "="*60)
        print("Prediction Confidence Analysis")
        print("="*60)
        y_val_pred, scores = classifier.predict(X_val, return_scores=True)
        print(f"\nAverage confidence: {scores.mean():.4f}")
        print(f"Min confidence: {scores.min():.4f}")
        print(f"Max confidence: {scores.max():.4f}")

        # Show low confidence predictions
        low_conf_threshold = 0.5
        low_conf_mask = scores < low_conf_threshold
        if low_conf_mask.any():
            print(f"\nLow confidence predictions (< {low_conf_threshold}):")
            print(f"  Count: {low_conf_mask.sum()} ({low_conf_mask.sum()/len(scores)*100:.1f}%)")

    # 6. Save models
    print("\n[6/6] Saving models...")
    classifier.save()

    # Optionally save embeddings separately
    if args.save_embeddings:
        print("\nSaving embedding matrices...")
        embedding_generator.save_embeddings(X_train, filepath="models/embeddings/train_embeddings.npy")
        embedding_generator.save_embeddings(X_val, filepath="models/embeddings/val_embeddings.npy")
        embedding_generator.save_embeddings(X_test, filepath="models/embeddings/test_embeddings.npy")

    # Final test set evaluation (optional)
    if args.eval_test:
        print("\n" + "="*60)
        print("Final Test Set Evaluation")
        print("="*60)
        y_test_pred = classifier.predict(X_test)
        test_metrics = evaluator.evaluate(y_test, y_test_pred, verbose=True)
        print_classification_report(y_test, y_test_pred, target_names=unique_classes)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nValidation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation F1 (macro): {val_metrics['f1_macro']:.4f}")

    if args.eval_test:
        print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")

    print("\nModels saved and ready for inference!")
    print("Use scripts/predict.py to classify new patents.")
    print("Use scripts/find_similar.py to find similar patents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train embedding-based patent classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV file (if None, uses config)"
    )
    parser.add_argument(
        "--use-saved-splits",
        action="store_true",
        help="Use previously saved train/val/test splits"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Sentence-transformer model name (if None, uses config)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors for K-NN"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for K-NN"
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show prediction confidence analysis"
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save embedding matrices separately"
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate on test set after training"
    )

    args = parser.parse_args()
    main(args)
