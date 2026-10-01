# Fine-Tuned Semantic Search Engine

A semantic search engine for Computer Science research papers.

The system uses a fine-tuned Sentence Transformer model to understand
the meaning of a search query and retrieve semantically relevant papers
from a collection of 30,000 CS papers.

## Features

- Semantic search instead of simple keyword matching
- Fine-tuned all-MiniLM-L6-v2 model
- Multiple Negatives Ranking Loss (MNRL)
- FAISS vector similarity search
- Gradio web interface
- 30,000 Computer Science research papers
- Evaluation using Recall@K, MRR@10 and NDCG@10

## How It Works

```text
User Query
    ↓
Fine-Tuned Sentence Transformer
    ↓
Query Embedding
    ↓
FAISS Vector Search
    ↓
Top-K Similar Papers
    ↓
Gradio Interface
```

The research paper abstracts are converted into numerical vectors
called embeddings.

When a user enters a query, the query is also converted into an
embedding.

FAISS compares the query embedding with the stored paper embeddings
and returns the most similar papers.

## Dataset

The project uses 30,000 Computer Science research papers.

Each paper contains:

- Title
- Abstract

For fine-tuning:

- anchor = paper title
- positive = paper abstract

Dataset split:

- Training: 27,000 papers
- Validation: 1,000 papers
- Test: 2,000 papers

## Model

Base model: `all-MiniLM-L6-v2`

The model was fine-tuned using:

`MultipleNegativesRankingLoss`

Training configuration:

- Epochs: 2
- Batch size: 64
- Learning rate: 2e-5
- Maximum sequence length: 256
- Mixed precision: FP16
- Batch sampler: No duplicates

### Why Multiple Negatives Ranking Loss?

For every query-document pair in a batch:

- Its corresponding document is treated as the positive.
- Other documents in the same batch act as negative examples.

## Evaluation

### Baseline

| Metric | Score |
|---|---:|
| Recall@1 | 0.9300 |
| Recall@5 | 0.9835 |
| Recall@10 | 0.9925 |
| MRR@10 | 0.9541 |
| NDCG@10 | 0.9636 |

### Fine-Tuned Model

| Metric | Score |
|---|---:|
| Recall@1 | 0.9690 |
| Recall@5 | 0.9945 |
| Recall@10 | 0.9985 |
| MRR@10 | 0.9803 |
| NDCG@10 | 0.9848 |

### Improvement

| Metric | Improvement |
|---|---:|
| Recall@1 | +0.0390 |
| Recall@5 | +0.0110 |
| Recall@10 | +0.0060 |
| MRR@10 | +0.0262 |
| NDCG@10 | +0.0212 |

Recall@1 increased from 93.00% to 96.90% on the 2,000-query test set.

## Project Structure

```text
semantic-search-finetune/
├── app.py
├── search.py
├── requirements.txt
├── index.faiss
├── corpus.json
├── README.md
└── cs-paper-model/
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── modules.json
    ├── sentence_bert_config.json
    ├── config_sentence_transformers.json
    ├── 1_Pooling/
    └── 2_Normalize/
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Search Engine

```bash
python app.py
```

The application provides a Gradio web interface.

Example query:

`deep learning methods for detecting cyber attacks`

The system returns the top 5 semantically similar papers with:

- Rank
- Similarity score
- Paper title
- Abstract

## Main Files

### app.py
Creates the Gradio web interface.

### search.py
Loads the fine-tuned model, FAISS index and paper corpus and performs search.

### index.faiss
Stores the vector representations of the paper abstracts.

### corpus.json
Stores the title and abstract corresponding to each vector.

### cs-paper-model/
Contains the final fine-tuned Sentence Transformer model.

### requirements.txt
Contains the Python dependencies required to run the project.

## Technologies Used

- Python
- Sentence Transformers
- Hugging Face
- FAISS
- NumPy
- Gradio
- Datasets

## Future Improvements

- Increase the document collection size
- Add metadata filtering
- Add paper authors and publication dates
- Use a larger embedding model
- Add a cross-encoder reranker
- Integrate the search engine into a RAG system