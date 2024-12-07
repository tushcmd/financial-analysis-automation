import streamlit as st
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API keys from .env
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Groq LLM Client
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Configuration
INDEX_NAME = "stocks"
NAMESPACE = "stock-descriptions"

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Embedding function
def get_embeddings(text):
    model = load_embedding_model()
    return model.encode(text)

def semantic_stock_search(query, top_k=10):
    query_embedding = get_embeddings(query)
    index = pc.Index(INDEX_NAME)
    
    results = index.query(
        vector=query_embedding.tolist(), 
        top_k=top_k, 
        include_metadata=True,
        namespace=NAMESPACE
    )
    
    return results['matches']

def get_stock_details(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        'Ticker': ticker,
        'Name': info.get('longName', 'N/A'),
        'Price': info.get('currentPrice', 'N/A'),
        'Market Cap': info.get('marketCap', 'N/A'),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'Business Summary': info.get('longBusinessSummary', 'N/A')
    }

def generate_llm_insight(query, context):
    augmented_query = f"<CONTEXT>\n{context}\n</CONTEXT>\n\nQUESTION: {query}"
    
    try:
        response = llm_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert stock market analyst. Provide insightful and concise analysis."},
                {"role": "user", "content": augmented_query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insight: {str(e)}"

def main():
    st.title("🔍 Advanced Stock Research")
    
    search_method = st.selectbox(
        "Choose Search Method", 
        ["Semantic Search", "Metric Search"]
    )
    
    if search_method == "Semantic Search":
        query = st.text_input("Describe the type of company you're looking for")
        
        if st.button("Search") and query:
            with st.spinner("Searching stocks and generating insights..."):
                results = semantic_stock_search(query)
                
                st.subheader("Search Results")
                for result in results:
                    ticker = result['metadata']['Ticker']
                    details = get_stock_details(ticker)
                    context = result['metadata']['Business Summary']
                    
                    with st.expander(f"{details['Name']} ({ticker})"):
                        for key, value in details.items():
                            st.write(f"**{key}**: {value}")
                        
                        st.subheader("AI-Generated Insight")
                        insight = generate_llm_insight(query, context)
                        st.write(insight)
    
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_market_cap = st.number_input("Minimum Market Cap ($)", min_value=0)
        
        with col2:
            sector = st.selectbox(
                "Sector", 
                ["All", "Technology", "Healthcare", "Finance", "Energy", "Consumer Discretionary"]
            )
        
        with col3:
            min_volume = st.number_input("Minimum Trading Volume", min_value=0)
        
        if st.button("Apply Filters"):
            st.write("Metric search functionality to be implemented")

if __name__ == "__main__":
    main()