import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

def import_products():
    # 1. Load the CSV
    print("Loading products.csv...")
    df = pd.read_csv('products.csv')
    
    # Fill NaN values to avoid errors
    df = df.fillna('')
    
    # 2. Initialize ChromaDB
    # We will use a persistent client so the data stays on disk
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 3. Setup Embedding Function
    # The user requested 1536 dimension, which usually refers to OpenAI's text-embedding-3-small or ada-002.
    # For this demo, we'll use a local embedding function if possible, 
    # but I'll provide instructions for OpenAI if needed.
    # By default, Chroma uses Sentence Transformers which are good for local testing.
    # To get exactly 1536, one would typically use OpenAI.
    
    # Check if OPENAI_API_KEY is available, if not, use default Chroma embedding
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("Using OpenAI embeddings (1536 dimensions)...")
        emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
    else:
        print("OPENAI_API_KEY not found. Using default Chroma embeddings (384 dimensions)...")
        print("Note: To get exactly 1536 dimensions, set your OPENAI_API_KEY.")
        emb_fn = embedding_functions.DefaultEmbeddingFunction()

    # 4. Create or Get Collection
    collection = client.get_or_create_collection(
        name="product_collection",
        embedding_function=emb_fn
    )

    # 5. Prepare Data for Chroma
    # We'll use the 'Description' and 'Title' as the text to embed
    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Combine title and description for better search context
        combined_text = f"Title: {row['Title']}\nDescription: {row['Description']}"
        documents.append(combined_text)
        
        # Store other useful info in metadata
        metadatas.append({
            "title": str(row['Title']),
            "price": str(row['Price']),
            "url": str(row['URL']),
            "image_url": str(row['Image URL']),
            "category": str(row['Category'])
        })
        
        # Use ID from CSV or index
        ids.append(str(row['ID']) if row['ID'] else str(index))

    # 6. Add to Collection in batches (Chroma handles large batches well, but let's be safe)
    print(f"Adding {len(documents)} products to ChromaDB...")
    
    # Add in batches of 100 to avoid any potential limits
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Imported batch {i//batch_size + 1}")

    print("Import completed successfully!")

if __name__ == "__main__":
    import_products()
