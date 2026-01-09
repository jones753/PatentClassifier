"""Predict CPC class for new patents using trained models."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import combine_patent_fields
from src.features.tfidf_features import TfidfFeatureExtractor
from src.models.tfidf_classifier import TfidfClassifier


def predict_tfidf(
    abstract: str,
    main_claim: str,
    config_path: str = "config.yaml",
    show_top_k: int = 3
):
    """
    Predict CPC class using TF-IDF model.

    Args:
        abstract: Patent abstract text
        main_claim: Patent main claim text
        config_path: Path to configuration file
        show_top_k: Number of top predictions to show

    Returns:
        Predicted class and probabilities (if available)
    """
    print("\n" + "="*60)
    print("TF-IDF Model Prediction")
    print("="*60)

    # Load models
    print("\nLoading TF-IDF models...")
    feature_extractor = TfidfFeatureExtractor(config_path=config_path)
    feature_extractor.load()

    classifier = TfidfClassifier(config_path=config_path)
    classifier.load()

    # Preprocess text
    text = combine_patent_fields(abstract, main_claim, lowercase=True)

    # Extract features
    features = feature_extractor.transform([text])

    # Predict
    prediction = classifier.predict(features)[0]

    print(f"\nPredicted CPC Class: {prediction}")

    # Try to show probabilities
    try:
        proba = classifier.predict_proba(features)[0]
        top_k_indices = proba.argsort()[-show_top_k:][::-1]

        print(f"\nTop {show_top_k} predictions:")
        for rank, idx in enumerate(top_k_indices, 1):
            class_label = classifier.classes_[idx]
            confidence = proba[idx]
            print(f"  {rank}. {class_label:20s} (confidence: {confidence:.4f})")

    except AttributeError:
        print("\n(Probability scores not available for this model type)")

    # Show top TF-IDF features for this text
    print("\nTop TF-IDF features in this patent:")
    top_features = feature_extractor.get_top_features_for_document(text, top_n=10)
    for feature, score in top_features[:10]:
        print(f"  {feature:30s} {score:.4f}")

    return prediction


def main(args):
    """Main prediction function."""
    # Check if models exist
    if args.model == "tfidf" or args.model == "both":
        model_path = Path("models/tfidf/classifier.pkl")
        if not model_path.exists():
            print(f"\nError: TF-IDF model not found at {model_path}")
            print("Please train the model first using scripts/train_tfidf.py")
            return

    # Handle input
    if args.abstract and args.main_claim:
        abstract = args.abstract
        main_claim = args.main_claim
    elif args.interactive:
        print("\n" + "="*60)
        print("Interactive Patent Classification")
        print("="*60)
        print("\nEnter patent information:")
        abstract = input("\nAbstract: ").strip()
        main_claim = input("Main Claim: ").strip()

        if not abstract or not main_claim:
            print("\nError: Both abstract and main claim are required.")
            return
    else:
        print("\nError: Please provide --abstract and --main-claim, or use --interactive mode")
        return

    # Display input
    print("\n" + "="*60)
    print("Patent Input")
    print("="*60)
    print(f"\nAbstract:\n{abstract[:200]}{'...' if len(abstract) > 200 else ''}")
    print(f"\nMain Claim:\n{main_claim[:200]}{'...' if len(main_claim) > 200 else ''}")

    # Predict
    if args.model == "tfidf":
        predict_tfidf(abstract, main_claim, args.config, args.top_k)

    elif args.model == "both":
        # TF-IDF prediction
        predict_tfidf(abstract, main_claim, args.config, args.top_k)

        # Try embedding prediction (if available)
        try:
            from src.features.embedding_features import EmbeddingGenerator
            from src.models.embedding_classifier import EmbeddingClassifier

            print("\n" + "="*60)
            print("Embedding Model Prediction")
            print("="*60)

            print("\nLoading embedding models...")
            embedding_generator = EmbeddingGenerator(config_path=args.config)
            embedding_classifier = EmbeddingClassifier(config_path=args.config)
            embedding_classifier.load()

            # Generate embedding
            text = combine_patent_fields(abstract, main_claim, lowercase=True)
            embedding = embedding_generator.generate([text])[0]

            # Predict
            prediction, similarity = embedding_classifier.predict_with_similarity(embedding)
            print(f"\nPredicted CPC Class: {prediction}")
            print(f"Similarity Score: {similarity:.4f}")

        except (ImportError, FileNotFoundError) as e:
            print("\nEmbedding model not available. Train it using scripts/train_embedding.py")

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict CPC class for new patents"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--abstract",
        type=str,
        default=None,
        help="Patent abstract text"
    )
    parser.add_argument(
        "--main-claim",
        type=str,
        default=None,
        help="Patent main claim text"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tfidf",
        choices=["tfidf", "embedding", "both"],
        help="Which model to use for prediction"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: prompt for input"
    )

    args = parser.parse_args()
    main(args)
