"""Find similar patents using semantic embeddings."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import combine_patent_fields
from src.features.embedding_features import EmbeddingGenerator
from src.models.embedding_classifier import EmbeddingClassifier


def find_similar_patents(
    abstract: str,
    main_claim: str,
    top_k: int = 5,
    config_path: str = "config.yaml"
):
    """
    Find similar patents to a query patent.

    Args:
        abstract: Patent abstract text
        main_claim: Patent main claim text
        top_k: Number of similar patents to return
        config_path: Path to configuration file
    """
    print("\n" + "="*60)
    print("Finding Similar Patents")
    print("="*60)

    # Load models
    print("\nLoading models...")
    embedding_generator = EmbeddingGenerator(config_path=config_path)
    classifier = EmbeddingClassifier(config_path=config_path)
    classifier.load()

    # Preprocess and combine text
    text = combine_patent_fields(abstract, main_claim, lowercase=True)

    # Generate embedding
    print("\nGenerating embedding for query patent...")
    query_embedding = embedding_generator.generate_single(text)

    # Find similar patents
    print(f"\nFinding top {top_k} most similar patents...")
    similar_patents = classifier.find_similar(
        query_embedding,
        top_k=top_k,
        return_texts=True
    )

    # Display results
    print("\n" + "="*60)
    print("Most Similar Patents")
    print("="*60)

    for idx, row in similar_patents.iterrows():
        print(f"\n{row['rank']}. CPC Class: {row['cpc_class']}")
        print(f"   Similarity: {row['similarity']:.4f}")
        print(f"   Distance:   {row['distance']:.4f}")

        if 'text' in row and row['text']:
            # Show snippet of text
            text_snippet = row['text'][:150]
            print(f"   Text: {text_snippet}{'...' if len(row['text']) > 150 else ''}")

    # Predict CPC class based on similarity
    prediction, similarity = classifier.predict_with_similarity(query_embedding)
    print("\n" + "="*60)
    print("Classification Result")
    print("="*60)
    print(f"Predicted CPC Class: {prediction}")
    print(f"Similarity Score: {similarity:.4f}")

    return similar_patents


def main(args):
    """Main function."""
    # Check if model exists
    model_path = Path("models/embeddings/index.pkl")
    if not model_path.exists():
        print(f"\nError: Embedding model not found at {model_path}")
        print("Please train the model first using scripts/train_embedding.py")
        return

    # Handle input
    if args.abstract and args.main_claim:
        abstract = args.abstract
        main_claim = args.main_claim
    elif args.interactive:
        print("\n" + "="*60)
        print("Interactive Similar Patent Search")
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

    # Display query patent
    print("\n" + "="*60)
    print("Query Patent")
    print("="*60)
    print(f"\nAbstract:\n{abstract[:200]}{'...' if len(abstract) > 200 else ''}")
    print(f"\nMain Claim:\n{main_claim[:200]}{'...' if len(main_claim) > 200 else ''}")

    # Find similar patents
    find_similar_patents(
        abstract=abstract,
        main_claim=main_claim,
        top_k=args.top_k,
        config_path=args.config
    )

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find similar patents using semantic embeddings"
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
        "--top-k",
        type=int,
        default=5,
        help="Number of similar patents to return"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: prompt for input"
    )

    args = parser.parse_args()
    main(args)
