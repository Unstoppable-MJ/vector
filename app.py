import os
import difflib
from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Initialize Gemini AI
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB Client
# Use the same database we imported the products into
client = chromadb.PersistentClient(path="./chroma_db")

# Setup Embedding Function
# Same as in the import script
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-3-small"
    )
else:
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

# Get the collection
collection = client.get_or_create_collection(
    name="product_collection",
    embedding_function=emb_fn
)

def get_categories(all_data):
    """Extract distinct categories from ChromaDB metadata."""
    if not all_data or not all_data['metadatas']:
        return []
    cats = set([m.get('category', '') for m in all_data['metadatas']])
    return sorted([c for c in cats if c and str(c).lower() != 'nan'])

@app.route('/')
def home():
    category_filter = request.args.get('category')
    all_data = collection.get()
    categories = get_categories(all_data)
    
    formatted_results = []
    if all_data and all_data['ids']:
        for i in range(len(all_data['ids'])):
            meta = all_data['metadatas'][i]
            if category_filter and meta.get('category') != category_filter:
                continue
            formatted_results.append({
                'document': all_data['documents'][i],
                'metadata': meta,
                'distance': 0.0,
                'id': all_data['ids'][i]
            })
    return render_template('index.html', results=formatted_results, categories=categories, selected_category=category_filter)

@app.route('/search', methods=['POST', 'GET'])
def search():
    query = request.form.get('query') or request.args.get('query')
    category_filter = request.form.get('category') or request.args.get('category')
    
    all_data = collection.get()
    categories = get_categories(all_data)
    
    if not query:
        formatted_results = []
        if all_data and all_data['ids']:
            for i in range(len(all_data['ids'])):
                meta = all_data['metadatas'][i]
                if category_filter and meta.get('category') != category_filter:
                    continue
                formatted_results.append({
                    'document': all_data['documents'][i],
                    'metadata': meta,
                    'distance': 0.0,
                    'id': all_data['ids'][i]
                })
        return render_template('index.html', results=formatted_results, categories=categories, selected_category=category_filter)
    
    query_lower = query.lower()
    
    # 1. Prefix / Substring Search
    prefix_matches = []
    if all_data and all_data['ids']:
        for i in range(len(all_data['ids'])):
            meta = all_data['metadatas'][i]
            if category_filter and meta.get('category') != category_filter:
                continue
            title = meta['title']
            if query_lower in title.lower():
                prefix_matches.append({
                    'document': all_data['documents'][i],
                    'metadata': meta,
                    'distance': 0.0,
                    'id': all_data['ids'][i]
                })
    
    # If we found prefix/substring matches, return them immediately
    if prefix_matches:
        return render_template('index.html', query=query, results=prefix_matches, categories=categories, selected_category=category_filter)

    # 2. "Did you mean" Suggestion
    all_titles = []
    if all_data and all_data['metadatas']:
        for m in all_data['metadatas']:
            if category_filter and m.get('category') != category_filter:
                continue
            all_titles.append(m['title'])
            
    lower_to_orig = {t.lower(): t for t in all_titles}
    close_matches = difflib.get_close_matches(query_lower, list(lower_to_orig.keys()), n=1, cutoff=0.7)
    
    if close_matches:
        suggestion = lower_to_orig[close_matches[0]]
        return render_template('index.html', query=query, results=[], suggestion=suggestion, categories=categories, selected_category=category_filter)
    
    # 3. Fallback to Semantic Search
    where_clause = {"category": category_filter} if category_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=6,
        where=where_clause
    )
    
    # Format the results for the template
    formatted_results = []
    if results and results['documents'] and len(results['documents']) > 0:
        for i in range(len(results['documents'][0])):
            dist = results['distances'][0][i]
            # Filter matches with distance > 1.0 (Match Score < 0.0)
            if dist <= 1.0:
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': dist,
                    'id': results['ids'][0][i]
                })
        
    return render_template('index.html', query=query, results=formatted_results, categories=categories, selected_category=category_filter)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"response": "Please provide a message."}), 400

    # 1. Query ChromaDB for context
    results = collection.query(
        query_texts=[user_message],
        n_results=5
    )
    
    context_products = []
    if results and results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            dist = results['distances'][0][i]
            if dist <= 1.5:  # Slightly more lenient threshold for chat context
                meta = results['metadatas'][0][i]
                title = meta.get('title', 'Unknown')
                price = meta.get('price', 'N/A')
                category = meta.get('category', 'N/A')
                url = meta.get('url', '#')
                image_url = meta.get('image_url', '')
                desc = results['documents'][0][i]
                
                context_products.append(
                    f"Product {i+1}:\n"
                    f"Title: {title}\n"
                    f"Price: {price}\n"
                    f"Category: {category}\n"
                    f"Description: {desc}\n"
                    f"Product Link: {url}\n"
                    f"Image URL: {image_url}\n"
                )

    if not context_products:
        return jsonify({"response": "I'm sorry, I couldn't find a product related to your request in our catalog. Would you like to search for hair care, immune support, or skin products?"})

    context_str = "\n".join(context_products)
    prompt = f"""You are a helpful and polite store assistant chatbot.
A user asked: "{user_message}"

Here are the most relevant products from our catalog based on their query:
{context_str}

Please provide a structured response based ONLY on the products listed above. 
Your response must format the products cleanly using Markdown.
For each product you recommend, include the following details in your response exactly in this format:

### [Product Title]
![[Product Title]]([Image URL])
**Price:** €[Price]
**Category:** [Category]
**Description:** [A brief 1-2 sentence description]
[View Product]([Product Link])

IMPORTANT RULES:
- Only recommend products that are explicitly listed above.
- If the user asks for a product type not listed, apologize and say you do not have it.
- Keep the response concise, helpful, and naturally conversational.
- Ask the user if they want more details about any product at the end.
"""

    try:
        response = client_genai.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({"response": "I'm sorry, my AI services are currently unavaliable. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
