import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
import streamlit as st
import pandas as pd
import pyterrier as pt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tempfile import mkdtemp

# 1. Initialize PyTerrier
if not pt.started():
    pt.init()

# 2. Setup NLTK (Exactly matching your notebook)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stop_words.difference_update(['no', 'nor', 'not'])
stemmer = PorterStemmer()

# 3. Your Preprocessing functions
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

def apply_stemming(text):
    return ' '.join([stemmer.stem(w) for w in text.split()])

def preprocess(text):
    if pd.isna(text): return ''
    return apply_stemming(remove_stopwords(normalize(text)))

# 4. Load data and build the index
@st.cache_resource
def load_system():
    collection = pd.read_csv("collection.csv")
    collection["content"] = collection["title"].fillna("") + " " + collection["text"].fillna("")
    collection["processed_content"] = collection["content"].apply(preprocess)
    
    documents = collection[["docno", "processed_content"]].copy()
    documents.columns = ["docno", "text"]
    
    indexer = pt.terrier.IterDictIndexer(mkdtemp(), stemmer=None, stopwords=None)
    index = indexer.index(documents.to_dict("records"))
    
    return index, collection

# Show a loading spinner while the index builds
with st.spinner("Building the search index... (This takes a few seconds)"):
    index, collection = load_system()

# 5. The Interface UI
st.set_page_config(page_title="Learnova Search", page_icon="🎓", layout="wide")
st.title("🎓 Learnova: Smart Online Learning")
st.write("Enter a query below to find the best Coursera courses.")

# Search Box and Settings
query = st.text_input("Search Courses:", placeholder="e.g., machine learning for beginners")

col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox("Select Retrieval Model:", ["TF_IDF", "BM25"])
with col2:
    top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)

if st.button("Search", type="primary"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        # Preprocess user query
        processed_query = preprocess(query)
        query_df = pd.DataFrame({"qid": ["Q1"], "query": [processed_query]})
        
        # Retrieve
        retriever = pt.terrier.Retriever(
            index, wmodel=model_choice, num_results=top_k, controls={"applypipeline": "off"}
        )
        results = retriever.transform(query_df)
        
        if results.empty:
            st.info("No results found.")
        else:
            # Combine results with original course titles
            results = results.merge(collection[["docno", "title", "text"]], on="docno")
            
            st.success(f"Showing top {len(results)} results using {model_choice}")
            
            # Display the results
            for _, row in results.iterrows():
                st.markdown(f"### {row['title']}")
                st.caption(f"**Doc ID:** {row['docno']} | **Score:** {row['score']:.4f} | **Rank:** {row['rank']}")
                
                # Show a snippet of the description
                snippet = row['text'][:400] + "..." if len(row['text']) > 400 else row['text']
                st.write(snippet)
                st.divider()
