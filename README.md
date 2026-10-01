# Fine-Tuned Semantic Search Engine

A domain-specific semantic search engine for **Computer Science research papers**, built by fine-tuning a Sentence Transformer model and combining it with FAISS vector search.

The system converts research-paper abstracts and user queries into dense vector embeddings, then retrieves the most semantically relevant papers using vector similarity rather than simple keyword matching.

---

## Overview

Traditional keyword search mainly looks for exact or closely related words.

For example, a user searching:

> `deep learning methods for detecting cyber attacks`

may want papers about:

- network intrusion detection
- anomaly detection
- adversarial attacks
- cybersecurity using neural networks

even when those exact words do not appear together in the paper.

This project uses **semantic embeddings** to capture the meaning of the query and documents.

### Pipeline

```text
                    User Query
                        |
                        v
              Fine-Tuned Sentence
                 Transformer
                        |
                        v
                 Query Embedding
                        |
                        v
                  FAISS Index
                        |
                        v
              Similarity Search
                        |
                        v
                 Top-K Papers
                        |
                        v
                Gradio Interface

```

---

## Key Features

- Fine-tuned Sentence Transformer model
- Domain-specific training on Computer Science papers
- Multiple Negatives Ranking Loss (MNRL)
- Semantic rather than keyword-based search
- FAISS vector similarity search
- Normalized embeddings
- Top-K paper retrieval
- Gradio web interface
- Quantitative comparison against the original pretrained model
- Evaluation using:
  - Recall\@1
  - Recall\@5
  - Recall\@10
  - MRR\@10
  - NDCG\@10

---

## Dataset

The project uses a collection of **30,000 Computer Science research papers**.

Each paper contains:

- `title`
- `abstract`

For fine-tuning, the data was converted into query-document pairs:

```text
anchor   → paper title
positive → paper abstract

```

### Dataset Split

| Split      | Number of Papers |
| ---------- | ---------------- |
| Training   | 27,000           |
| Validation | 1,000            |
| Test       | 2,000            |
| **Total**  | **30,000**       |

The test set was kept separate from training and was used only for final evaluation.

---

## Model

### Base Model

```text
all-MiniLM-L6-v2

```

The pretrained model was used as the starting point and then fine-tuned on the Computer Science paper dataset.

### Fine-Tuning Objective

The model was trained using:

```text
MultipleNegativesRankingLoss

```

The goal is to make the embedding of a query more similar to its relevant document and less similar to unrelated documents.

### In-Batch Negatives

Suppose a batch contains:

```text
Query 1 → Document 1
Query 2 → Document 2
Query 3 → Document 3

```

For Query 1:

```text
Positive: Document 1
Negative: Document 2
Negative: Document 3

```

The same idea is applied to every query in the batch.

This allows many negative examples to be obtained from the same training batch without manually creating negative pairs.

---

## Training Configuration

| Parameter               | Value                        |
| ----------------------- | ---------------------------- |
| Base model              | `all-MiniLM-L6-v2`           |
| Loss                    | MultipleNegativesRankingLoss |
| Epochs                  | 2                            |
| Batch size              | 64                           |
| Learning rate           | `2e-5`                       |
| Maximum sequence length | 256                          |
| Precision               | FP16                         |
| Batch sampler           | No duplicates                |

Training was performed using a Google Colab GPU environment.

The final training run completed **844 steps over 2 epochs**.

---

# Evaluation

An important part of the project was evaluating the pretrained model **before fine-tuning** and then evaluating the fine-tuned model on the same untouched test set.

This allows the effect of fine-tuning to be measured quantitatively.

## Evaluation Setup

For each test example:

```text
Query = paper title
Relevant document = corresponding paper abstract

```

The evaluator then searches the complete test corpus and checks where the relevant document appears in the ranking.

---

## Baseline Results

The original `all-MiniLM-L6-v2` model achieved:

| Metric     | Baseline |
| ---------- | -------- |
| Recall\@1  | 0.9300   |
| Recall\@5  | 0.9835   |
| Recall\@10 | 0.9925   |
| MRR\@10    | 0.9541   |
| NDCG\@10   | 0.9636   |

---

## Fine-Tuned Results

After fine-tuning:

| Metric     | Fine-Tuned |
| ---------- | ---------- |
| Recall\@1  | **0.9690** |
| Recall\@5  | **0.9945** |
| Recall\@10 | **0.9985** |
| MRR\@10    | **0.9803** |
| NDCG\@10   | **0.9848** |

---

## Improvement

| Metric     | Baseline | Fine-Tuned | Change      |
| ---------- | -------- | ---------- | ----------- |
| Recall\@1  | 0.9300   | **0.9690** | **+0.0390** |
| Recall\@5  | 0.9835   | **0.9945** | **+0.0110** |
| Recall\@10 | 0.9925   | **0.9985** | **+0.0060** |
| MRR\@10    | 0.9541   | **0.9803** | **+0.0262** |
| NDCG\@10   | 0.9636   | **0.9848** | **+0.0212** |

### Key Result

Recall\@1 improved from:

```text
93.00% → 96.90%

```

on the 2,000-query test set.

That corresponds to:

```text
Baseline correct top-1 results:
0.93 × 2000 = 1860

Fine-tuned correct top-1 results:
0.969 × 2000 = 1938

```

Therefore, the fine-tuned model retrieved the relevant document as the first result for **78 additional test queries**.

---

# Why These Metrics Matter

### Recall\@K

Recall\@K measures whether the relevant document appears within the first K retrieved results.

For example:

```text
Recall@5 = 0.9945

```

means the relevant document was found within the top 5 results for 99.45% of the evaluated queries.

### MRR\@10

Mean Reciprocal Rank measures how high the relevant document appears in the top 10 results.

A relevant document at rank 1 contributes:

```text
1 / 1 = 1

```

while a relevant document at rank 5 contributes:

```text
1 / 5 = 0.2

```

Therefore, higher MRR generally means relevant results are appearing closer to the top.

### NDCG\@10

NDCG evaluates the ranking quality of retrieved results, giving more importance to relevant documents appearing near the top.

---

# Vector Search

After fine-tuning, embeddings were generated for the **30,000 paper abstracts**.

The embeddings were normalized and stored in a FAISS index.

```text
30,000 abstracts
       |
       v
Fine-Tuned Sentence Transformer
       |
       v
Dense embeddings
       |
       v
Normalization
       |
       v
FAISS IndexFlatIP

```

The FAISS index uses **inner-product similarity** on normalized embeddings.

With normalized vectors, this provides cosine-similarity-style ranking.

---

# Search Process

When a user enters a query:

### 1. Query

Example:

```text
deep learning methods for detecting cyber attacks

```

### 2. Query Embedding

The fine-tuned Sentence Transformer converts the query into a numerical vector.

### 3. FAISS Search

FAISS compares the query vector against the 30,000 stored document vectors.

### 4. Ranking

The documents are ranked according to similarity.

### 5. Results

The application returns the top 5 papers containing:

- Rank
- Similarity score
- Title
- Abstract

---

# Example

Example query:

```text
deep learning methods for detecting cyber attacks

```

The search engine retrieves papers related to concepts such as:

- software security
- adversarial attacks
- cyber threat intelligence
- anomaly detection
- network intrusion detection

The system does not require an exact keyword match because retrieval is based on the semantic representation of the text.

---

# Project Structure

```text
semantic-search-finetune/
│
├── app.py
├── search.py
├── requirements.txt
├── README.md
├── .gitattributes
│
├── index.faiss
├── corpus.json
│
└── cs-paper-model/
    ├── model.safetensors
    ├── sentence_bert_config.json
    ├── config_sentence_transformers.json
    ├── config.json
    ├── modules.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    │
    ├── 1_Pooling/
    │   └── config.json
    │
    └── 2_Normalize/
        └── ...

```

Large files are tracked using **Git LFS**.

---

# File Description

### `app.py`

Creates the Gradio web interface.

The interface accepts a natural-language search query and displays the top 5 retrieved papers.

### `search.py`

Contains the actual semantic search logic.

It:

1. Loads the fine-tuned model.
2. Loads the FAISS index.
3. Loads the paper corpus.
4. Converts the query into an embedding.
5. Searches FAISS.
6. Returns the ranked results.

### `index.faiss`

FAISS vector index containing the embeddings of the 30,000 paper abstracts.

### `corpus.json`

Contains the metadata corresponding to each indexed document:

```json
{
    "title": "...",
    "abstract": "..."
}

```

The position of each paper corresponds to its vector position in the FAISS index.

### `cs-paper-model/`

Contains the final fine-tuned Sentence Transformer model and its configuration/tokenizer files.

### `requirements.txt`

Contains the Python dependencies required by the project.

### `.gitattributes`

Configures Git LFS tracking for large files.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/anushritadey05/semantic-search-finetune.git

```

Move into the project:

```bash
cd semantic-search-finetune

```

Install dependencies:

```bash
pip install -r requirements.txt

```

---

# Running the Application

Start the Gradio application:

```bash
python app.py

```

The application will start a local Gradio interface.

Enter a Computer Science research query, for example:

```text
deep learning methods for detecting cyber attacks

```

The system will return the top 5 semantically similar papers.

---

# Technology Stack

| Technology            | Purpose                        |
| --------------------- | ------------------------------ |
| Python                | Main programming language      |
| Sentence Transformers | Text embedding and fine-tuning |
| Hugging Face Datasets | Dataset loading and processing |
| PyTorch               | Model training                 |
| FAISS                 | Vector similarity search       |
| NumPy                 | Numerical processing           |
| Gradio                | Web interface                  |
| Git                   | Version control                |
| Git LFS               | Large file storage             |

---

# Architecture

```text
                 ┌─────────────────────┐
                 │    Research Papers  │
                 │      30,000         │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ Fine-Tuned Sentence │
                 │     Transformer     │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ Document Embeddings │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │    FAISS Index      │
                 │     IndexFlatIP     │
                 └──────────┬──────────┘
                            │
                            │
User Query ────────────────►│
                            │
                            v
                 ┌─────────────────────┐
                 │   Similarity Search │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │    Top-K Papers     │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │   Gradio Interface  │
                 └─────────────────────┘

```

---

# Important Design Choices

## Why fine-tune the model?

The original model is a general-purpose sentence embedding model.

Fine-tuning allows the embedding space to become better suited to the specific retrieval task and Computer Science paper domain used in this project.

The improvement in Recall\@1, MRR\@10 and NDCG\@10 provides quantitative evidence of this change on the held-out test set.

## Why a bi-encoder?

The Sentence Transformer encodes queries and documents independently.

This means document embeddings can be computed once and stored in FAISS.

At search time, only the user's query needs to be encoded.

This makes the approach much more suitable for large document collections than encoding every query-document pair from scratch.

## Why FAISS?

FAISS is designed for efficient similarity search over dense vectors.

Instead of comparing the query against documents using raw text matching, the system searches through their vector representations.

---

# Limitations

The current implementation has several limitations:

- The dataset contains 30,000 papers rather than a production-scale collection.
- The retrieval evaluation uses one known relevant document per query.
- Some semantically similar documents may be treated as negatives during training.
- The current index uses `IndexFlatIP`, which performs exact similarity search and may require more resources as the corpus becomes very large.
- There is no cross-encoder reranking stage.
- The current interface is a lightweight Gradio application rather than a production web service.

---

# Future Improvements

Possible extensions include:

### Larger Corpus

Scale the document collection to hundreds of thousands or millions of papers.

### Better Indexing

For very large collections, investigate approximate nearest-neighbor FAISS indexes such as IVF or HNSW.

### Cross-Encoder Reranking

Use a cross-encoder after initial vector retrieval to improve the ordering of the top candidate documents.

```text
Query
  ↓
Bi-Encoder
  ↓
FAISS
  ↓
Top 50 candidates
  ↓
Cross-Encoder
  ↓
Top 5 results

```

### Metadata Filtering

Allow users to filter papers by:

- publication year
- author
- research area
- category

### RAG Integration

The semantic search engine can act as the retrieval component of a Retrieval-Augmented Generation system:

```text
User Question
      ↓
Semantic Search
      ↓
Relevant Papers
      ↓
LLM
      ↓
Generated Answer

```

---

# Interview Concepts Demonstrated

This project demonstrates practical understanding of:

- Sentence embeddings
- Semantic search
- Transformer-based embeddings
- Domain-specific fine-tuning
- Multiple Negatives Ranking Loss
- In-batch negative sampling
- Bi-encoder architecture
- Vector databases/search indexes
- FAISS
- Cosine similarity
- Recall\@K
- MRR
- NDCG
- Information Retrieval evaluation
- Model deployment
- Git and Git LFS
- Gradio

---

# Results Summary

The main experimental result is:

```text
                    Baseline       Fine-Tuned

Recall@1              93.00%          96.90%
Recall@5              98.35%          99.45%
Recall@10             99.25%          99.85%

MRR@10                0.9541          0.9803
NDCG@10               0.9636          0.9848

```

The fine-tuned model therefore showed improved retrieval performance across all reported evaluation metrics on the held-out test set.

---

# Author

**Anushrita Dey**

B.Tech — Computer Science & Engineering (AI & ML)

Institute of Engineering and Management, Kolkata

---

# License

This project is intended for educational, research, and portfolio purposes.
