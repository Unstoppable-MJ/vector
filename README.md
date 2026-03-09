# Vector Database & AI Similarity Demo

A educational repository designed to help developers understand the core concepts behind vectors, similarity metrics, and vector databases like ChromaDB.

## 🚀 Overview

This project provides a step-by-step learning path to understand how AI "understands" data through numerical representations.

### 📚 Learning Path

1.  **[1_vectors_intro.py](./1_vectors_intro.py)**: 
    - Introduces the concept of vectors as lists of numbers.
    - Uses a simple fruit example (Sweetness vs. Sourness) to visualize coordinates.
2.  **[2_vector_similarity.py](./2_vector_similarity.py)**: 
    - Explains **Cosine Similarity**.
    - Shows how to mathematically compare two vectors to see how similar they are.
3.  **[3_ai_examples.py](./3_ai_examples.py)**: 
    - Moves from simple 2D vectors to AI "Embeddings".
    - Demonstrates how words like "Pizza" and "Burger" are closer in a high-dimensional space than "Pizza" and "Laptop".
4.  **[4_chromadb_demo.py](./4_chromadb_demo.py)**: 
    - Hands-on with **ChromaDB**, an open-source vector database.
    - Shows how to store documents, automatically generate embeddings, and perform semantic searches.
5.  **[app.py](./app.py)**: 
    - The main Flask application that provides a search interface.
6.  **[import_products.py](./import_products.py)**: 
    - A script to import products from `products.csv` into ChromaDB.

## 🛍️ AI Product Search Interface

This project now includes a full search interface for products.

### 📥 Importing Data
To import the products into the vector database:
```bash
python3 import_products.py
```

### 🌐 Running the Web App
To start the Flask search interface:
```bash
python3 app.py
```
Then visit `http://127.0.0.1:5001` in your browser.

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- `pip` (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:amolc/vectordbdemo.git
   cd vectordbdemo
   ```
2. Install dependencies:
   ```bash
   pip install chromadb
   ```

### Running the Demos
Run the scripts in order to build your understanding:
```bash
python3 1_vectors_intro.py
python3 2_vector_similarity.py
python3 3_ai_examples.py
python3 4_chromadb_demo.py
```

## 🧠 Key Concepts Covered
- **Vectors**: Numerical representations of features.
- **Embeddings**: Turning complex data (like text) into vectors using AI models.
- **Cosine Similarity**: Measuring the angle between vectors to find similarity.
- **Vector Databases**: Specialized databases optimized for storing and searching through billions of vectors.
