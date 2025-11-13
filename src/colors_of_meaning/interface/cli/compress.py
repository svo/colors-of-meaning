import tyro
import pickle  # nosec B403
from dataclasses import dataclass
from typing import List

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.application.use_case.compress_document_use_case import (
    CompressDocumentUseCase,
)


@dataclass
class CompressArgs:
    config: str = "configs/base.yaml"
    encoded_documents: str = "artifacts/encoded/test_documents.pkl"
    method: str = "vq"


def main(args: CompressArgs) -> None:
    print(f"Loading encoded documents from {args.encoded_documents}...")
    with open(args.encoded_documents, "rb") as f:
        documents: List[ColoredDocument] = pickle.load(f)  # nosec B301 nosemgrep

    use_case = CompressDocumentUseCase()

    print(f"\nAnalyzing compression for {len(documents)} documents...")
    batch_results = use_case.execute_batch(documents)

    print("\n" + "=" * 60)
    print("COMPRESSION ANALYSIS")
    print("=" * 60)
    print(f"Total bits: {batch_results['total_bits']:,}")
    print(f"Total tokens: {batch_results['total_tokens']:,}")
    print(f"Average bits per token: {batch_results['average_bits_per_token']:.2f}")
    print("=" * 60)

    print("\nSample individual results:")
    for i, result in enumerate(batch_results["individual_results"][:5]):
        print(f"\nDocument {i}:")
        print(f"  Tokens: {result['num_tokens']}")
        print(f"  Total bits: {result['total_bits']}")
        print(f"  Bits per token: {result['bits_per_token']:.2f}")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}x")


if __name__ == "__main__":
    main(tyro.cli(CompressArgs))
