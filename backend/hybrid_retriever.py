"""
State-of-the-Art Hybrid Retriever for MRIIA Educational Content

Combines multiple retrieval strategies:
- Dense retrieval using pre-computed embeddings
- Sparse retrieval using BM25
- Semantic reranking with cross-encoders
- Query expansion and fusion techniques
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Core dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Optional advanced dependencies (install if available)
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available. Using basic embeddings.")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank-bm25 not available. Using TF-IDF for sparse retrieval.")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Warning: spaCy not available. Using basic text processing.")


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    content_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    retrieval_method: str


@dataclass
class HybridConfig:
    """Configuration for hybrid retriever"""
    # Embedding models
    dense_model_name: str = "all-MiniLM-L6-v2"
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval parameters
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 10
    
    # Fusion weights
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    
    # Advanced features
    use_query_expansion: bool = True
    use_reranking: bool = True
    use_semantic_chunking: bool = True
    
    # Performance
    faiss_index_type: str = "IndexFlatIP"  # or "IndexIVFFlat"
    cache_embeddings: bool = True


class TextProcessor:
    """Advanced text processing utilities"""
    
    def __init__(self):
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("uk_core_news_sm")  # Ukrainian model
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")  # Fallback to English
                except OSError:
                    print("Warning: No spaCy model available")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Advanced processing with spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            # Remove stop words and punctuation, lemmatize
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct and token.is_alpha]
            return ' '.join(tokens)
        
        return text.lower()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text"""
        if self.nlp:
            doc = self.nlp(text)
            # Extract named entities and noun phrases
            keywords = []
            for ent in doc.ents:
                keywords.append(ent.text.lower())
            for chunk in doc.noun_chunks:
                keywords.append(chunk.text.lower())
            return list(set(keywords))[:max_keywords]
        
        # Fallback: simple word extraction
        words = text.lower().split()
        return [w for w in words if len(w) > 3][:max_keywords]


class DenseRetriever:
    """Dense retrieval using embeddings and FAISS"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(config.dense_model_name)
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents"""
        self.documents = documents
        texts = [doc['content'] for doc in documents]
        
        if self.model:
            # Use sentence transformer
            embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            # Fallback: use pre-computed embeddings if available
            embeddings = []
            for doc in documents:
                if 'embedding' in doc and doc['embedding'] is not None:
                    embeddings.append(doc['embedding'])
                else:
                    # Create dummy embedding
                    embeddings.append(np.random.normal(0, 1, 384))
            embeddings = np.array(embeddings)
        
        self.embeddings = embeddings.astype('float32')
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        if self.config.faiss_index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(documents) // 10))
            self.index.train(self.embeddings)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(self.embeddings)
        print(f"✓ Built dense index with {len(documents)} documents")
    
    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Search using dense retrieval"""
        if not self.index or not self.model:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(RetrievalResult(
                    content_id=doc.get('id', str(idx)),
                    content=doc['content'],
                    score=float(score),
                    source=doc.get('source', 'unknown'),
                    metadata=doc.get('metadata', {}),
                    retrieval_method='dense'
                ))
        
        return results


class SparseRetriever:
    """Sparse retrieval using BM25 or TF-IDF"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.bm25 = None
        self.tfidf = None
        self.documents = []
        self.processed_docs = []
        self.text_processor = TextProcessor()
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build sparse index from documents"""
        self.documents = documents
        
        # Process documents
        self.processed_docs = []
        for doc in documents:
            processed = self.text_processor.preprocess_text(doc['content'])
            self.processed_docs.append(processed.split())
        
        if HAS_BM25:
            # Use BM25
            self.bm25 = BM25Okapi(self.processed_docs)
            print(f"✓ Built BM25 index with {len(documents)} documents")
        else:
            # Fallback to TF-IDF
            texts = [' '.join(tokens) for tokens in self.processed_docs]
            self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            self.tfidf_matrix = self.tfidf.fit_transform(texts)
            print(f"✓ Built TF-IDF index with {len(documents)} documents")
    
    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Search using sparse retrieval"""
        processed_query = self.text_processor.preprocess_text(query).split()
        
        if self.bm25:
            # BM25 search
            scores = self.bm25.get_scores(processed_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    doc = self.documents[idx]
                    results.append(RetrievalResult(
                        content_id=doc.get('id', str(idx)),
                        content=doc['content'],
                        score=float(scores[idx]),
                        source=doc.get('source', 'unknown'),
                        metadata=doc.get('metadata', {}),
                        retrieval_method='bm25'
                    ))
            return results
        
        elif self.tfidf:
            # TF-IDF search
            query_vec = self.tfidf.transform([' '.join(processed_query)])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = self.documents[idx]
                    results.append(RetrievalResult(
                        content_id=doc.get('id', str(idx)),
                        content=doc['content'],
                        score=float(scores[idx]),
                        source=doc.get('source', 'unknown'),
                        metadata=doc.get('metadata', {}),
                        retrieval_method='tfidf'
                    ))
            return results
        
        return []


class QueryExpander:
    """Query expansion using various techniques"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def expand_query(self, query: str, documents: List[Dict[str, Any]] = None) -> List[str]:
        """Expand query with related terms"""
        expanded_queries = [query]
        
        # Extract keywords from original query
        keywords = self.text_processor.extract_keywords(query)
        
        # Add keyword-based variations
        if keywords:
            keyword_query = ' '.join(keywords[:5])  # Top 5 keywords
            if keyword_query != query:
                expanded_queries.append(keyword_query)
        
        # Add question variations
        if '?' not in query:
            expanded_queries.append(f"Що таке {query}?")  # Ukrainian "What is"
            expanded_queries.append(f"Як {query}?")       # Ukrainian "How"
        
        return expanded_queries


class CrossEncoderReranker:
    """Reranking using cross-encoder models"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.model = None
        
        if HAS_SENTENCE_TRANSFORMERS and config.use_reranking:
            try:
                self.model = CrossEncoder(config.rerank_model_name)
            except Exception as e:
                print(f"Warning: Could not load reranker model: {e}")
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        if not self.model or len(results) <= 1:
            return results[:top_k]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result.content) for result in results]
        
        try:
            # Get reranking scores
            rerank_scores = self.model.predict(pairs)
            
            # Update scores and sort
            for i, score in enumerate(rerank_scores):
                results[i].score = float(score)
                results[i].retrieval_method += '+rerank'
            
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Warning: Reranking failed: {e}")
            return results[:top_k]


class HybridRetriever:
    """State-of-the-art hybrid retriever combining multiple strategies"""
    
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.dense_retriever = DenseRetriever(self.config)
        self.sparse_retriever = SparseRetriever(self.config)
        self.query_expander = QueryExpander()
        self.reranker = CrossEncoderReranker(self.config)
        self.documents = []
        self.is_built = False
    
    def build(self, data_path: str = "data") -> None:
        """Build retriever from MRIIA dataset"""
        print("🔨 Building hybrid retriever...")
        
        # Load and process all content
        documents = self._load_documents(data_path)
        
        if not documents:
            raise ValueError("No documents found to build retriever")
        
        self.documents = documents
        
        # Build indices
        print("Building dense index...")
        self.dense_retriever.build_index(documents)
        
        print("Building sparse index...")
        self.sparse_retriever.build_index(documents)
        
        self.is_built = True
        print(f"✅ Hybrid retriever built with {len(documents)} documents")
    
    def _load_documents(self, data_path: str) -> List[Dict[str, Any]]:
        """Load documents from MRIIA dataset"""
        documents = []
        data_dir = Path(data_path)
        
        # Load textbook content (TOC)
        for model_dir in ['gemini-embedding-001', 'text-embedding-qwen']:
            toc_file = data_dir / model_dir / 'toc_for_hackathon_with_subtopics.parquet'
            if toc_file.exists():
                print(f"Loading {toc_file}...")
                df = pd.read_parquet(toc_file)
                
                for _, row in df.iterrows():
                    # Use pre-computed embeddings if available
                    embedding = None
                    if 'topic_embedding' in row and row['topic_embedding'] is not None:
                        try:
                            embedding = np.array(row['topic_embedding'])
                        except:
                            pass
                    
                    doc = {
                        'id': f"toc_{model_dir}_{row.get('book_topic_id', len(documents))}",
                        'content': row['topic_text'],
                        'source': 'textbook_toc',
                        'embedding': embedding,
                        'metadata': {
                            'title': row.get('topic_title', ''),
                            'subject': row.get('global_discipline_name', ''),
                            'grade': row.get('grade', 0),
                            'book_name': row.get('book_name', ''),
                            'topic_type': row.get('topic_type', ''),
                            'subtopics': row.get('subtopics', []),
                            'embedding_model': model_dir
                        }
                    }
                    documents.append(doc)
                
                # Only use one embedding model to avoid duplicates
                break
        
        # Load textbook pages
        for model_dir in ['gemini-embedding-001', 'text-embedding-qwen']:
            pages_file = data_dir / model_dir / 'pages_for_hackathon.parquet'
            if pages_file.exists():
                print(f"Loading {pages_file}...")
                df = pd.read_parquet(pages_file)
                
                for _, row in df.iterrows():
                    # Use pre-computed embeddings if available
                    embedding = None
                    if 'page_text_embedding' in row and row['page_text_embedding'] is not None:
                        try:
                            embedding = np.array(row['page_text_embedding'])
                        except:
                            pass
                    
                    doc = {
                        'id': f"page_{model_dir}_{row.get('book_id', '')}_{row.get('book_page_number', len(documents))}",
                        'content': row['page_text'],
                        'source': 'textbook_page',
                        'embedding': embedding,
                        'metadata': {
                            'subject': row.get('global_discipline_name', ''),
                            'grade': row.get('grade', 0),
                            'book_name': row.get('book_name', ''),
                            'page_number': row.get('book_page_number', 0),
                            'section_title': row.get('section_title', ''),
                            'topic_title': row.get('topic_title', ''),
                            'page_metadata': row.get('page_metadata', {}),
                            'embedding_model': model_dir
                        }
                    }
                    documents.append(doc)
                
                # Only use one embedding model to avoid duplicates
                break
        
        # Load LMS questions
        questions_file = data_dir / 'lms_questions_dev.parquet'
        if questions_file.exists():
            print(f"Loading {questions_file}...")
            df = pd.read_parquet(questions_file)
            
            for _, row in df.iterrows():
                # Combine question and answers for content
                answers_text = ' '.join(row.get('answers', []))
                content = f"Питання: {row['question_text']} Варіанти відповідей: {answers_text}"
                
                doc = {
                    'id': f"question_{row['question_id']}",
                    'content': content,
                    'source': 'lms_question',
                    'embedding': None,  # No pre-computed embeddings for questions
                    'metadata': {
                        'question_text': row['question_text'],
                        'answers': row.get('answers', []),
                        'correct_answer_indices': row.get('correct_answer_indices', []),
                        'subject': row.get('global_discipline_name', ''),
                        'grade': row.get('grade', 0),
                        'test_type': row.get('test_type', ''),
                        'model': row.get('model', ''),
                        'source': row.get('source', '')
                    }
                }
                documents.append(doc)
        
        print(f"Loaded {len(documents)} documents total")
        return documents
    
    def search(self, 
               query: str, 
               top_k: Optional[int] = None,
               subject_filter: Optional[str] = None,
               grade_filter: Optional[int] = None,
               source_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Search using hybrid retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            subject_filter: Filter by subject (e.g., 'Українська мова')
            grade_filter: Filter by grade (8 or 9)
            source_filter: Filter by source ('textbook_toc', 'textbook_page', 'lms_question')
        """
        if not self.is_built:
            raise ValueError("Retriever not built. Call build() first.")
        
        top_k = top_k or self.config.final_top_k
        
        print(f"🔍 Searching for: '{query}'")
        
        # Query expansion
        queries = [query]
        if self.config.use_query_expansion:
            expanded = self.query_expander.expand_query(query, self.documents)
            queries.extend(expanded[:2])  # Limit to avoid noise
        
        all_results = []
        
        # Search with each query variant
        for q in queries:
            # Dense retrieval
            dense_results = self.dense_retriever.search(q, self.config.dense_top_k)
            
            # Sparse retrieval
            sparse_results = self.sparse_retriever.search(q, self.config.sparse_top_k)
            
            all_results.extend(dense_results)
            all_results.extend(sparse_results)
        
        # Remove duplicates and apply filters
        seen_ids = set()
        filtered_results = []
        
        for result in all_results:
            if result.content_id in seen_ids:
                continue
            seen_ids.add(result.content_id)
            
            # Apply filters
            if subject_filter and result.metadata.get('subject') != subject_filter:
                continue
            if grade_filter and result.metadata.get('grade') != grade_filter:
                continue
            if source_filter and result.source != source_filter:
                continue
            
            filtered_results.append(result)
        
        # Fusion scoring (combine dense and sparse scores)
        self._apply_fusion_scoring(filtered_results)
        
        # Sort by fused score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # Take top candidates for reranking
        candidates = filtered_results[:min(50, len(filtered_results))]
        
        # Reranking
        if self.config.use_reranking and len(candidates) > 1:
            print("🔄 Reranking results...")
            candidates = self.reranker.rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]
        
        print(f"✅ Found {len(candidates)} results")
        return candidates
    
    def _apply_fusion_scoring(self, results: List[RetrievalResult]) -> None:
        """Apply fusion scoring to combine dense and sparse scores"""
        # Normalize scores by method
        dense_results = [r for r in results if 'dense' in r.retrieval_method]
        sparse_results = [r for r in results if r.retrieval_method in ['bm25', 'tfidf']]
        
        # Normalize dense scores
        if dense_results:
            dense_scores = [r.score for r in dense_results]
            if max(dense_scores) > min(dense_scores):
                for result in dense_results:
                    result.score = (result.score - min(dense_scores)) / (max(dense_scores) - min(dense_scores))
        
        # Normalize sparse scores
        if sparse_results:
            sparse_scores = [r.score for r in sparse_results]
            if max(sparse_scores) > min(sparse_scores):
                for result in sparse_results:
                    result.score = (result.score - min(sparse_scores)) / (max(sparse_scores) - min(sparse_scores))
        
        # Apply fusion weights
        for result in results:
            if 'dense' in result.retrieval_method:
                result.score *= self.config.dense_weight
            else:
                result.score *= self.config.sparse_weight
    
    def save_index(self, path: str) -> None:
        """Save built indices to disk"""
        if not self.is_built:
            raise ValueError("No index to save. Build first.")
        
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        if self.dense_retriever.index:
            faiss.write_index(self.dense_retriever.index, str(save_path / "dense_index.faiss"))
        
        # Save documents and metadata
        with open(save_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"✅ Index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """Load indices from disk"""
        load_path = Path(path)
        
        # Load config
        with open(load_path / "config.json", 'r') as f:
            config_dict = json.load(f)
            self.config = HybridConfig(**config_dict)
        
        # Load documents
        with open(load_path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load FAISS index
        faiss_path = load_path / "dense_index.faiss"
        if faiss_path.exists():
            self.dense_retriever.index = faiss.read_index(str(faiss_path))
            self.dense_retriever.documents = self.documents
        
        # Rebuild sparse index (lightweight)
        self.sparse_retriever.build_index(self.documents)
        
        self.is_built = True
        print(f"✅ Index loaded from {path}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize retriever
    config = HybridConfig(
        dense_top_k=30,
        sparse_top_k=30,
        final_top_k=10,
        use_query_expansion=True,
        use_reranking=True
    )
    
    retriever = HybridRetriever(config)
    
    # Build from data
    try:
        retriever.build("data")
        
        # Test searches
        test_queries = [
            "Що таке дієслово?",
            "Алгебраїчні рівняння",
            "Історія України 19 століття",
            "Граматика української мови"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            results = retriever.search(query, top_k=5)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result.retrieval_method}] Score: {result.score:.3f}")
                print(f"   Source: {result.source}")
                print(f"   Subject: {result.metadata.get('subject', 'N/A')}")
                print(f"   Grade: {result.metadata.get('grade', 'N/A')}")
                print(f"   Content: {result.content[:200]}...")
        
        # Save index for future use
        retriever.save_index("retriever_index")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the data directory exists and contains the required parquet files.")