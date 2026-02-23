# Embeddings from Scratch — Chapter 2 Implementation

Student Name: Santiago Amaya Zapata

This repository contains my implementation and experimentation of **Chapter 2** from:

> *Build a Large Language Model (From Scratch)*  
> by Sebastian Raschka

The goal of this notebook is to understand how raw text becomes numerical representations (embeddings), which are the foundation of Large Language Models (LLMs) and agentic AI systems.

---

# Project Overview

This project focuses on:

- Tokenization using GPT-2 tokenizer (`tiktoken`)
- Creating training samples using sliding windows
- Understanding context windows (`max_length` and `stride`)
- Implementing embedding layers in PyTorch
- Experimenting with overlapping sequences
- Explaining why embeddings encode semantic meaning
- Connecting embeddings to agentic systems and neural networks

The notebook runs fully end-to-end.

---

# Project Architecture

The project structure is intentionally simple:

.

├── embeddings.ipynb

├── the-verdict.txt

├── .gitignore

└── README.md


### Components

**embeddings.ipynb**
- Loads raw text
- Tokenizes using GPT-2 encoding
- Creates sliding windows
- Builds tensors
- Applies embedding layer
- Runs an experiment modifying `max_length` and `stride`
- Contains personal conceptual explanations

**the-verdict.txt**
- Text dataset used for tokenization and window generation

---

# Installation & Setup

## 1. Clone the repository

```bash
git clone https://github.com/SantiagoAmaya21/TDSE_LLM_Text_Preprocessing_Foundations_Embeddings.git
cd https://github.com/SantiagoAmaya21/TDSE_LLM_Text_Preprocessing_Foundations_Embeddings.git
```

## 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

## 3. Install dependencies

```bash
%pip install torch tiktoken
```

## 4. Run the notebook

Run all cells sequentially.

The notebook should execute without errors.

## Why This Matters for LLMs

Large Language Models do not understand text directly.
They operate entirely in vector space.

This notebook demonstrates:

How text becomes tokens

How tokens become integers

How integers become vectors

How sliding windows create next-token prediction samples

How embeddings are trainable neural network parameters

Embeddings encode meaning because:

Tokens that appear in similar contexts receive similar gradient updates.

Backpropagation moves semantically related tokens closer in vector space.

The embedding matrix is a learnable parameter of the neural network.

In other words:

Meaning emerges from optimization.

## Experiment: Changing max_length and stride

The notebook includes an experiment modifying:

max_length

stride

We measure how many training samples are generated under different configurations.

Observations

Smaller stride → more overlap → more samples

Larger stride → fewer samples

Larger max_length → fewer samples but richer context

Why overlap is useful

Overlapping windows:

Increase effective dataset size

Provide smoother learning signals

Improve gradient stability

Help models generalize better

This mechanism is essential in real LLM pretraining pipelines.

## Connection to Agentic Systems

Embeddings are the foundation of:

Retrieval-Augmented Generation (RAG)

Semantic search

Vector databases

Memory in AI agents

Similarity comparison via cosine distance

Without embeddings, modern AI systems cannot reason about semantic similarity.

Embeddings transform language into geometry — and geometry is computable.

## Grading Criteria Coverage

- Notebook runs cleanly with outputs
- Clear personal explanations in markdown cells
- Experiment with modified parameters
- Conceptual understanding demonstrated
- Connection to neural networks and agent systems

## References

Sebastian Raschka — Build a Large Language Model (From Scratch)

PyTorch Documentation

OpenAI GPT-2 Tokenization (tiktoken)