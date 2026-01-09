"""Train TF-IDF based patent classifier."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_data
from src.data.preprocessor import TextPreprocessor
from src.features.tfidf_features import TfidfFeatureExtractor
from src.models.tfidf_classifier import TfidfClassifier
from src.utils.evaluation import ModelEvaluator, print_classification_report


def main(args):
    """Main training pipeline."""
    print("="*60)
    print("TF-IDF Patent Classifier Training Pipeline")
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

    # 3. Extract TF-IDF features
    print("\n[3/6] Extracting TF-IDF features...")
    feature_extractor = TfidfFeatureExtractor(config_path=args.config)

    X_train = feature_extractor.fit_transform(X_train_text)
    X_val = feature_extractor.transform(X_val_text)
    X_test = feature_extractor.transform(X_test_text)

    print(f"Feature matrix shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # 4. Train classifier
    print("\n[4/6] Training classifier...")
    classifier = TfidfClassifier(
        model_type=args.model_type,
        config_path=args.config
    )
    classifier.fit(X_train, y_train)

    # 5. Evaluate on validation set
    print("\n[5/6] Evaluating on validation set...")
    evaluator = ModelEvaluator()

    y_val_pred = classifier.predict(X_val)
    val_metrics = evaluator.evaluate(y_val, y_val_pred, verbose=True)

    # Print detailed classification report
    unique_classes = sorted(train_df['cpc_class'].unique())
    print_classification_report(y_val, y_val_pred, target_names=unique_classes)

    # Show feature importances (if available)
    if args.show_features:
        print("\n" + "="*60)
        print("Top Features per Class")
        print("="*60)
        feature_names = feature_extractor.get_feature_names()
        importances = classifier.get_feature_importances(
            feature_names=feature_names,
            top_n=10
        )

        for class_label, features in importances.items():
            print(f"\nClass: {class_label}")
            for feature, score in features[:10]:
                print(f"  {feature:30s} {score:+.4f}")

    # 6. Save models
    print("\n[6/6] Saving models...")
    feature_extractor.save()
    classifier.save()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TF-IDF based patent classifier"
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
        "--model-type",
        type=str,
        default=None,
        help="Model type: logistic_regression, svm, random_forest (if None, uses config)"
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Show top features per class"
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate on test set after training"
    )

    args = parser.parse_args()
    main(args)
