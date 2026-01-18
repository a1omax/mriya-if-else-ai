"""
Embeddings module using SentenceTransformer for semantic similarity
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")


@dataclass
class EmbeddedNode:
    """A knowledge graph node with its embedding"""
    node_id: str
    node_type: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class EmbeddingManager:
    """
    Manages embeddings for knowledge graph nodes using SentenceTransformer
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_path: Optional[str] = None
    ):
        """
        Initialize the embedding manager
        
        Args:
            model_name: SentenceTransformer model name (multilingual recommended for Ukrainian)
            cache_path: Optional path to cache embeddings
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_path = cache_path
        
        # Storage for embedded nodes
        self.embedded_nodes: Dict[str, EmbeddedNode] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.node_ids: List[str] = []
        
        # Load cache if exists
        if cache_path and os.path.exists(cache_path):
            self._load_cache()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def embed_knowledge_graph(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Embed all relevant nodes from knowledge graph
        
        Args:
            nodes: List of knowledge graph nodes
        """
        texts_to_embed = []
        nodes_to_process = []
        
        for node in nodes:
            node_id = node.get("id", "")
            node_type = node.get("node_type", "")
            
            # Create text representation based on node type
            if node_type == "topic":
                text = self._create_topic_text(node)
            elif node_type == "term":
                text = node.get("name", "") or node.get("title", "")
            elif node_type == "section":
                text = node.get("title", "")
            elif node_type == "concept":
                text = f"{node.get('name', '')} {node.get('description', '')}"
            else:
                text = node.get("title", "") or node.get("name", "")
            
            if text.strip():
                texts_to_embed.append(text)
                nodes_to_process.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "text": text,
                    "metadata": node
                })
        
        if not texts_to_embed:
            return
        
        # Batch embed all texts
        embeddings = self.embed_texts(texts_to_embed)
        
        # Store embedded nodes
        for i, node_info in enumerate(nodes_to_process):
            embedded_node = EmbeddedNode(
                node_id=node_info["node_id"],
                node_type=node_info["node_type"],
                text=node_info["text"],
                embedding=embeddings[i],
                metadata=node_info["metadata"]
            )
            self.embedded_nodes[node_info["node_id"]] = embedded_node
        
        # Build matrix for fast similarity search
        self._build_embeddings_matrix()
        
        # Save cache
        if self.cache_path:
            self._save_cache()
    
    def _create_topic_text(self, node: Dict[str, Any]) -> str:
        """Create rich text representation for topic node"""
        parts = []
        
        if node.get("title"):
            parts.append(node["title"])
        
        if node.get("summary"):
            # Truncate summary if too long
            summary = node["summary"][:500] if len(node.get("summary", "")) > 500 else node["summary"]
            parts.append(summary)
        
        return " ".join(parts)
    
    def _build_embeddings_matrix(self) -> None:
        """Build matrix of all embeddings for fast similarity search"""
        self.node_ids = list(self.embedded_nodes.keys())
        self.embeddings_matrix = np.vstack([
            self.embedded_nodes[nid].embedding for nid in self.node_ids
        ])
    
    def find_similar_nodes(
        self, 
        query_text: str, 
        top_k: int = 5,
        node_type_filter: Optional[str] = None,
        threshold: float = 0.0
    ) -> List[Tuple[EmbeddedNode, float]]:
        """
        Find nodes most similar to query text
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            node_type_filter: Optional filter by node type
            threshold: Minimum similarity threshold
            
        Returns:
            List of (EmbeddedNode, similarity_score) tuples
        """
        if self.embeddings_matrix is None or len(self.node_ids) == 0:
            return []
        
        # Embed query
        query_embedding = self.embed_text(query_text)
        
        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings_matrix)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            node_id = self.node_ids[idx]
            node = self.embedded_nodes[node_id]
            similarity = float(similarities[idx])
            
            # Apply filters
            if similarity < threshold:
                continue
            
            if node_type_filter and node.node_type != node_type_filter:
                continue
            
            results.append((node, similarity))
        
        return results
    
    def find_similar_to_terms(
        self,
        terms: List[str],
        top_k: int = 5,
        node_type_filter: Optional[str] = None
    ) -> Dict[str, List[Tuple[EmbeddedNode, float]]]:
        """
        Find similar nodes for multiple terms
        
        Args:
            terms: List of terms to search for
            top_k: Number of results per term
            node_type_filter: Optional filter by node type
            
        Returns:
            Dictionary mapping each term to its similar nodes
        """
        results = {}
        for term in terms:
            results[term] = self.find_similar_nodes(
                term, 
                top_k=top_k, 
                node_type_filter=node_type_filter
            )
        return results
    
    def _cosine_similarity(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all vectors in matrix"""
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        
        # Normalize matrix rows
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        matrix_normalized = matrix / matrix_norms
        
        # Compute dot products
        return np.dot(matrix_normalized, query_norm)
    
    def _save_cache(self) -> None:
        """Save embeddings to cache file"""
        if not self.cache_path:
            return
        
        cache_data = {
            "model_name": self.model_name,
            "nodes": {}
        }
        
        for node_id, embedded_node in self.embedded_nodes.items():
            cache_data["nodes"][node_id] = {
                "node_type": embedded_node.node_type,
                "text": embedded_node.text,
                "embedding": embedded_node.embedding.tolist(),
                "metadata": embedded_node.metadata
            }
        
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
    
    def _load_cache(self) -> None:
        """Load embeddings from cache file"""
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Verify model matches
            if cache_data.get("model_name") != self.model_name:
                print(f"Cache model mismatch. Re-embedding required.")
                return
            
            for node_id, data in cache_data.get("nodes", {}).items():
                embedded_node = EmbeddedNode(
                    node_id=node_id,
                    node_type=data["node_type"],
                    text=data["text"],
                    embedding=np.array(data["embedding"]),
                    metadata=data["metadata"]
                )
                self.embedded_nodes[node_id] = embedded_node
            
            self._build_embeddings_matrix()
            print(f"Loaded {len(self.embedded_nodes)} embeddings from cache")
            
        except Exception as e:
            print(f"Failed to load cache: {e}")


class SemanticTopicMatcher:
    """
    Enhanced topic matcher using semantic embeddings
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        fuzzy_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        """
        Args:
            embedding_manager: Initialized EmbeddingManager with embedded KG
            fuzzy_weight: Weight for fuzzy string matching
            semantic_weight: Weight for semantic similarity
        """
        self.embeddings = embedding_manager
        self.fuzzy_weight = fuzzy_weight
        self.semantic_weight = semantic_weight
    
    def find_matching_topics(
        self,
        suggested_topic: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find matching topics using semantic similarity
        
        Args:
            suggested_topic: Topic text to match
            top_k: Number of results
            threshold: Minimum combined score threshold
            
        Returns:
            List of matches with scores
        """
        # Get semantic matches
        semantic_matches = self.embeddings.find_similar_nodes(
            suggested_topic,
            top_k=top_k * 2,  # Get more for filtering
            node_type_filter="topic"
        )
        
        results = []
        for node, semantic_score in semantic_matches:
            # Calculate fuzzy score
            from difflib import SequenceMatcher
            fuzzy_score = SequenceMatcher(
                None, 
                suggested_topic.lower(), 
                node.text.lower()
            ).ratio()
            
            # Combined score
            combined_score = (
                self.semantic_weight * semantic_score +
                self.fuzzy_weight * fuzzy_score
            )
            
            if combined_score >= threshold:
                results.append({
                    "topic_id": node.node_id,
                    "topic_title": node.metadata.get("title", node.text),
                    "semantic_score": float(semantic_score),
                    "fuzzy_score": float(fuzzy_score),
                    "combined_score": float(combined_score),
                    "match_type": "semantic",
                    "summary": node.metadata.get("summary", "")[:200],
                    "metadata": node.metadata
                })
        
        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]
    
    def find_related_terms(
        self,
        weak_terms: List[str],
        top_k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related nodes for weak terms
        """
        results = {}
        
        for term in weak_terms:
            matches = self.embeddings.find_similar_nodes(
                term,
                top_k=top_k,
                threshold=0.4
            )
            
            results[term] = [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "text": node.text,
                    "similarity": float(score),
                    "metadata": node.metadata
                }
                for node, score in matches
            ]
        
        return results
    
    def find_related_concepts(
        self,
        weak_concepts: List[str],
        top_k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related nodes for weak concepts
        """
        results = {}
        
        for concept in weak_concepts:
            # Search across all node types for concepts
            matches = self.embeddings.find_similar_nodes(
                concept,
                top_k=top_k,
                threshold=0.3
            )
            
            results[concept] = [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "text": node.text,
                    "similarity": float(score),
                    "metadata": node.metadata
                }
                for node, score in matches
            ]
        
        return results