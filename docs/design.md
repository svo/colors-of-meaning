# Synesthetic Color Representation and Compression

## Overview

This project implements a novel text representation system that maps text to colors using CIE Lab color space and represents documents as color distributions. The approach is inspired by synesthesia, where one sensory modality (text) is represented through another (color).

## Architecture

The system follows Hexagonal Architecture (Ports and Adapters) with Domain-Driven Design principles:

### Domain Layer
- **LabColor**: Value object representing CIE Lab color space (L: 0-100, a: -128 to 127, b: -128 to 127)
- **ColorCodebook**: Quantized palette of 4,096 colors (16x16x16 grid in Lab space)
- **ColoredDocument**: Document representation as normalized color histogram with optional temporal statistics
- **ColorCodebookRepository**: Interface for persisting codebooks
- **ColorMapper**: Interface for mapping embeddings to Lab colors
- **DistanceCalculator**: Interface for computing document distances

### Application Layer
- **TrainColorMappingUseCase**: Trains the embedding-to-Lab projector
- **EncodeDocumentUseCase**: Converts text documents to color histograms
- **CompareDocumentsUseCase**: Computes distances and finds nearest neighbors
- **CompressDocumentUseCase**: Analyzes compression efficiency (bits per token)

### Infrastructure Layer
- **SentenceEmbeddingAdapter**: Wraps sentence-transformers for text embeddings
- **PyTorchColorMapper**: Neural network projector (384→128→64→3) mapping embeddings to Lab
- **WassersteinDistanceCalculator**: Computes Wasserstein-2 distance on histograms
- **JensenShannonDistanceCalculator**: Computes Jensen-Shannon divergence
- **FileColorCodebookRepository**: Persistence layer for codebooks

### Interface Layer
- **CLI Commands**: train, encode, compare, compress using Tyro for configuration

## Design Rationale

### Why CIE Lab Color Space?
- Perceptually uniform: Euclidean distance approximates human color perception
- Three dimensions (L, a, b) map naturally to semantic axes
- Established standard in color science

### Why 4,096 Colors (12-bit)?
- Balance between expressiveness and compression
- 16 bins per dimension provides fine-grained color distinctions
- 12 bits per token enables efficient encoding

### Why Neural Projector?
- Learnable mapping captures semantic-to-perceptual relationships
- Bottleneck architecture enforces information compression
- Can be trained end-to-end with task-specific objectives

### Why Histogram Representation?
- Orderless representation (bag-of-colors) similar to bag-of-words
- Naturally normalized probability distribution
- Compatible with standard distance metrics (Wasserstein, Jensen-Shannon)

## Evaluation Strategy

### Datasets
- AG News (topic classification)
- IMDB (sentiment classification)
- 20 Newsgroups (retrieval)

### Baselines
- TF-IDF + Logistic Regression
- Sentence Embeddings + Product Quantization (FAISS)

### Metrics
- Classification: Accuracy, Macro-F1
- Retrieval: Recall@k, MRR
- Compression: Bits per token
- Fairness: Compare at matched bit budgets

## Ablation Studies (Planned)

1. **Quantization**: Continuous Lab vs. quantized bins
2. **Representation**: Histogram only vs. histogram + temporal stats
3. **Mapping**: Learned projector vs. interpretable clustering
4. **Distance**: Wasserstein vs. Jensen-Shannon

## Current Limitations

- Training uses random targets (unsupervised) - could benefit from task-specific supervision
- Sentence splitting is simplistic (needs proper NLP tokenization)
- No online learning or incremental updates
- Limited to English text via sentence-transformers

## Future Work

- Interpretable mapping with semantic axes (hue=topic, lightness=sentiment, saturation=concreteness)
- Multi-modal extension (images, audio)
- Hierarchical color spaces for variable-resolution encoding
- Integration with vector databases for large-scale retrieval
