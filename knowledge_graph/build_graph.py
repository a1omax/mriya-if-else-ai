"""
Educational Knowledge Graph Builder

This script builds a knowledge graph from educational parquet files by:
1. Extracting topic-subtopic relations from the parquet structure
2. Using GLiNER to extract educational terms from text
3. Normalizing and linking terms
4. Building and exporting the knowledge graph

Usage:
    python knowledge_graph_builder.py --topics topics.parquet --pages pages.parquet --output graph_output
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedTerm:
    """Represents an extracted educational term."""
    text: str
    label: str
    normalized: str
    confidence: float
    source_topic_id: str
    source_text_snippet: str = ""


@dataclass
class TopicNode:
    """Represents a topic in the knowledge graph."""
    topic_id: str
    title: str
    section_title: str
    book_id: str
    book_name: str
    grade: int
    discipline: str
    topic_type: str = ""
    summary: str = ""
    terms: list = field(default_factory=list)


@dataclass
class SubtopicNode:
    """Represents a subtopic in the knowledge graph."""
    subtopic_id: str
    name: str
    parent_topic_id: str
    text: str = ""
    terms: list = field(default_factory=list)


class TermNormalizer:
    """Normalizes and links educational terms."""
    
    def __init__(self, language: str = "uk"):
        self.language = language
        self.term_cache = {}
        self.synonym_groups = defaultdict(set)
        
    def normalize(self, term: str) -> str:
        """
        Normalize a term by:
        - Converting to lowercase
        - Removing extra whitespace
        - Removing punctuation (except hyphens in compound words)
        - Basic stemming/lemmatization
        """
        if term in self.term_cache:
            return self.term_cache[term]
        
        # Basic normalization
        normalized = term.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing punctuation but keep internal hyphens
        normalized = re.sub(r'^[^\w]+|[^\w]+$', '', normalized)
        
        # Remove parenthetical content for cleaner terms
        normalized = re.sub(r'\s*\([^)]*\)\s*', ' ', normalized).strip()
        
        # Cache the result
        self.term_cache[term] = normalized
        
        return normalized
    
    def are_similar(self, term1: str, term2: str, threshold: float = 0.8) -> bool:
        """Check if two terms are similar using character-level similarity."""
        norm1 = self.normalize(term1)
        norm2 = self.normalize(term2)
        
        if norm1 == norm2:
            return True
        
        # Check if one is substring of another
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        # Simple Jaccard similarity on character n-grams
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(norm1)
        ngrams2 = get_ngrams(norm2)
        
        if not ngrams1 or not ngrams2:
            return False
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return (intersection / union) >= threshold if union > 0 else False
    
    def add_synonym(self, term1: str, term2: str):
        """Add two terms as synonyms."""
        norm1 = self.normalize(term1)
        norm2 = self.normalize(term2)
        
        # Find existing groups
        group1 = None
        group2 = None
        
        for canonical, synonyms in self.synonym_groups.items():
            if norm1 in synonyms or norm1 == canonical:
                group1 = canonical
            if norm2 in synonyms or norm2 == canonical:
                group2 = canonical
        
        if group1 and group2:
            # Merge groups
            if group1 != group2:
                self.synonym_groups[group1].update(self.synonym_groups[group2])
                self.synonym_groups[group1].add(group2)
                del self.synonym_groups[group2]
        elif group1:
            self.synonym_groups[group1].add(norm2)
        elif group2:
            self.synonym_groups[group2].add(norm1)
        else:
            # Create new group with shorter term as canonical
            canonical = norm1 if len(norm1) <= len(norm2) else norm2
            other = norm2 if canonical == norm1 else norm1
            self.synonym_groups[canonical].add(other)
    
    def get_canonical(self, term: str) -> str:
        """Get the canonical form of a term."""
        normalized = self.normalize(term)
        
        for canonical, synonyms in self.synonym_groups.items():
            if normalized == canonical or normalized in synonyms:
                return canonical
        
        return normalized


class EntityExtractor:
    """Extracts educational entities using GLiNER with batch inference."""
    
    def __init__(
        self,
        model_name: str = "knowledgator/gliner-x-large",
        threshold: float = 0.5,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        
        # Educational entity labels
        self.labels = [
            "term",
            "concept",
            "definition",
            "rule",
            "example",
            "formula",
            "theorem",
            "law",
            "principle",
            "method",
            "process",
            "category",
            "type",
            "property",
            "characteristic"
        ]
    
    def load_model(self):
        """Load the GLiNER model."""
        if self.model is None:
            logger.info(f"Loading GLiNER model: {self.model_name}")
            try:
                from gliner import GLiNER
                self.model = GLiNER.from_pretrained(self.model_name)
                if self.device == "cuda":
                    self.model = self.model.to("cuda")
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def extract_entities(
        self,
        text: str,
        labels: Optional[list] = None,
        threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Extract entities from a single text using GLiNER.
        
        Args:
            text: Input text to extract entities from
            labels: Optional list of labels to extract (defaults to self.labels)
            threshold: Optional confidence threshold (defaults to self.threshold)
        
        Returns:
            List of extracted entities with text, label, and score
        """
        if self.model is None:
            self.load_model()
        
        if labels is None:
            labels = self.labels
        
        if threshold is None:
            threshold = self.threshold
        
        # Handle long texts by chunking
        if len(text) > self.max_length:
            chunks = self._chunk_text(text, self.max_length)
            # Use batch inference for chunks
            all_results = self.model.inference(
                texts=chunks,
                labels=labels,
                threshold=threshold,
                batch_size=self.batch_size,
                flat_ner=True,
                multi_label=False
            )
            # Flatten and deduplicate
            entities = []
            for chunk_entities in all_results:
                entities.extend(chunk_entities)
            return self._deduplicate_entities(entities)
        
        # Single text - still use inference for consistency
        results = self.model.inference(
            texts=[text],
            labels=labels,
            threshold=threshold,
            batch_size=1,
            flat_ner=True,
            multi_label=False
        )
        return results[0] if results else []
    
    def extract_entities_batch(
        self,
        texts: list[str],
        labels: Optional[list] = None,
        threshold: Optional[float] = None
    ) -> list[list[dict]]:
        """
        Extract entities from multiple texts using batch inference.
        
        This is significantly faster than calling extract_entities() in a loop.
        
        Args:
            texts: List of input texts to extract entities from
            labels: Optional list of labels to extract (defaults to self.labels)
            threshold: Optional confidence threshold (defaults to self.threshold)
        
        Returns:
            List of lists, where each inner list contains extracted entities for
            the corresponding input text
        """
        if self.model is None:
            self.load_model()
        
        if labels is None:
            labels = self.labels
        
        if threshold is None:
            threshold = self.threshold
        
        if not texts:
            return []
        
        # Pre-process texts: chunk long texts and track mapping
        processed_texts = []
        text_to_chunks_map = []  # Maps original text index to chunk indices
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                text_to_chunks_map.append([])
                continue
                
            if len(text) > self.max_length:
                chunks = self._chunk_text(text, self.max_length)
                start_idx = len(processed_texts)
                processed_texts.extend(chunks)
                text_to_chunks_map.append(list(range(start_idx, start_idx + len(chunks))))
            else:
                text_to_chunks_map.append([len(processed_texts)])
                processed_texts.append(text)
        
        if not processed_texts:
            return [[] for _ in texts]
        
        # Run batch inference
        all_results = self.model.inference(
            texts=processed_texts,
            labels=labels,
            threshold=threshold,
            batch_size=self.batch_size,
            flat_ner=True,
            multi_label=False
        )
        
        # Reconstruct results for original texts
        final_results = []
        for chunk_indices in text_to_chunks_map:
            if not chunk_indices:
                final_results.append([])
                continue
            
            # Gather entities from all chunks for this text
            text_entities = []
            for idx in chunk_indices:
                if idx < len(all_results):
                    text_entities.extend(all_results[idx])
            
            # Deduplicate
            final_results.append(self._deduplicate_entities(text_entities))
        
        return final_results
    
    def _deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """Remove duplicate entities based on text and label."""
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        return unique_entities
    
    def _chunk_text(self, text: str, max_length: int) -> list[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text[:max_length]]


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from educational data."""
    
    def __init__(
        self,
        extractor: EntityExtractor,
        normalizer: TermNormalizer
    ):
        self.extractor = extractor
        self.normalizer = normalizer
        self.graph = nx.DiGraph()
        
        # Storage for nodes
        self.books = {}
        self.sections = {}
        self.topics = {}
        self.subtopics = {}
        self.terms = {}
        
    def load_topics_data(self, parquet_path: str) -> pd.DataFrame:
        """Load topics data from parquet file."""
        logger.info(f"Loading topics data from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} topic records")
        return df
    
    def load_pages_data(self, parquet_path: str) -> pd.DataFrame:
        """Load pages data from parquet file."""
        logger.info(f"Loading pages data from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} page records")
        return df
    
    def build_graph_from_topics(self, df: pd.DataFrame, extract_terms: bool = True):
        """
        Build the knowledge graph from topics dataframe.
        
        Creates nodes for:
        - Books
        - Sections
        - Topics
        - Subtopics
        - Terms (if extract_terms=True)
        
        Uses batch inference for efficient term extraction.
        """
        logger.info("Building knowledge graph from topics data...")
        
        # First pass: build structural graph
        topic_texts = {}  # topic_id -> text
        subtopic_texts = {}  # subtopic_id -> (text, parent_topic_id)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building structure"):
            # Create book node
            book_id = str(row.get('book_id', ''))
            if book_id and book_id not in self.books:
                self.books[book_id] = {
                    'id': book_id,
                    'name': row.get('book_name', ''),
                    'grade': row.get('grade', ''),
                    'discipline': row.get('global_discipline_name', '')
                }
                self.graph.add_node(
                    f"book:{book_id}",
                    node_type="book",
                    **self.books[book_id]
                )
            
            # Create section node
            section_id = str(row.get('book_section_id', ''))
            section_title = row.get('section_title', '')
            if section_id and section_id not in self.sections:
                self.sections[section_id] = {
                    'id': section_id,
                    'title': section_title,
                    'book_id': book_id,
                    'start_page': row.get('section_start_page', ''),
                    'end_page': row.get('section_end_page', '')
                }
                self.graph.add_node(
                    f"section:{section_id}",
                    node_type="section",
                    **self.sections[section_id]
                )
                # Link section to book
                if book_id:
                    self.graph.add_edge(
                        f"book:{book_id}",
                        f"section:{section_id}",
                        relation="CONTAINS_SECTION"
                    )
            
            # Create topic node
            topic_id = str(row.get('book_topic_id', ''))
            topic_title = row.get('topic_title', '')
            topic_text = row.get('topic_text', '')
            
            if topic_id and topic_id not in self.topics:
                self.topics[topic_id] = {
                    'id': topic_id,
                    'title': topic_title,
                    'type': row.get('topic_type', ''),
                    'summary': row.get('topic_summary', ''),
                    'section_id': section_id,
                    'book_id': book_id,
                    'start_page': row.get('topic_start_page', ''),
                    'end_page': row.get('topic_end_page', ''),
                    'terms': []
                }
                self.graph.add_node(
                    f"topic:{topic_id}",
                    node_type="topic",
                    **{k: v for k, v in self.topics[topic_id].items() if k != 'terms'}
                )
                # Link topic to section
                if section_id:
                    self.graph.add_edge(
                        f"section:{section_id}",
                        f"topic:{topic_id}",
                        relation="CONTAINS_TOPIC"
                    )
                
                # Store text for batch extraction
                if extract_terms and topic_text:
                    topic_texts[topic_id] = topic_text
            
            # Process subtopics
            subtopics_data = row.get('subtopics_with_text', [])
            if subtopics_data is None:
                subtopics_data = []
                
            subtopics_names = row.get('subtopics', [])
            
            if len(subtopics_data) and isinstance(subtopics_data, list):
                for i, subtopic in enumerate(subtopics_data):
                    if isinstance(subtopic, dict):
                        subtopic_name = subtopic.get('name', '')
                        subtopic_text = subtopic.get('text', '')
                    else:
                        subtopic_name = subtopics_names[i] if i < len(subtopics_names) else str(subtopic)
                        subtopic_text = ""
                    
                    subtopic_id = f"{topic_id}_sub_{i}"
                    
                    if subtopic_id not in self.subtopics:
                        self.subtopics[subtopic_id] = {
                            'id': subtopic_id,
                            'name': subtopic_name,
                            'text': subtopic_text,
                            'parent_topic_id': topic_id,
                            'terms': []
                        }
                        self.graph.add_node(
                            f"subtopic:{subtopic_id}",
                            node_type="subtopic",
                            **{k: v for k, v in self.subtopics[subtopic_id].items() if k != 'terms'}
                        )
                        # Link subtopic to topic
                        self.graph.add_edge(
                            f"topic:{topic_id}",
                            f"subtopic:{subtopic_id}",
                            relation="HAS_SUBTOPIC"
                        )
                        
                        # Store text for batch extraction
                        if extract_terms and subtopic_text:
                            subtopic_texts[subtopic_id] = (subtopic_text, topic_id)
        
        # Second pass: batch extract terms
        if extract_terms and (topic_texts or subtopic_texts):
            logger.info("Extracting terms using batch inference...")
            
            # Extract from topics
            if topic_texts:
                topic_ids = list(topic_texts.keys())
                texts = [topic_texts[tid] for tid in topic_ids]
                
                logger.info(f"Processing {len(texts)} topic texts...")
                all_entities = self.extractor.extract_entities_batch(texts)
                
                for topic_id, entities in zip(topic_ids, all_entities):
                    self._link_extracted_terms(
                        entities,
                        f"topic:{topic_id}",
                        topic_id
                    )
            
            # Extract from subtopics
            if subtopic_texts:
                subtopic_ids = list(subtopic_texts.keys())
                texts = [subtopic_texts[sid][0] for sid in subtopic_ids]
                
                logger.info(f"Processing {len(texts)} subtopic texts...")
                all_entities = self.extractor.extract_entities_batch(texts)
                
                for subtopic_id, entities in zip(subtopic_ids, all_entities):
                    parent_topic_id = subtopic_texts[subtopic_id][1]
                    self._link_extracted_terms(
                        entities,
                        f"subtopic:{subtopic_id}",
                        parent_topic_id
                    )
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def build_graph_from_pages(self, df: pd.DataFrame, extract_terms: bool = True):
        """
        Enrich the knowledge graph with page-level data.
        
        This adds additional terms extracted from page texts and links them
        to existing topics. Uses batch inference for efficiency.
        """
        logger.info("Enriching knowledge graph from pages data...")
        
        # First pass: create page nodes and collect texts
        page_texts = {}  # page_id -> (text, topic_id)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing pages"):
            topic_id = str(row.get('book_topic_id', ''))
            page_text = row.get('page_text', '')
            page_number = row.get('page_number', '')
            
            # Create page node
            page_id = f"{row.get('book_id', '')}_{page_number}"
            
            if page_id not in self.graph:
                self.graph.add_node(
                    f"page:{page_id}",
                    node_type="page",
                    page_number=page_number,
                    book_page_number=row.get('book_page_number', ''),
                    book_id=row.get('book_id', '')
                )
                
                # Link page to topic if topic exists
                if f"topic:{topic_id}" in self.graph:
                    self.graph.add_edge(
                        f"topic:{topic_id}",
                        f"page:{page_id}",
                        relation="CONTAINS_PAGE"
                    )
                
                # Store text for batch extraction
                if extract_terms and page_text:
                    page_texts[page_id] = (page_text, topic_id)
        
        # Second pass: batch extract terms
        if extract_terms and page_texts:
            logger.info(f"Extracting terms from {len(page_texts)} pages using batch inference...")
            
            page_ids = list(page_texts.keys())
            texts = [page_texts[pid][0] for pid in page_ids]
            
            all_entities = self.extractor.extract_entities_batch(texts)
            
            for page_id, entities in zip(page_ids, all_entities):
                topic_id = page_texts[page_id][1]
                self._link_extracted_terms(
                    entities,
                    f"page:{page_id}",
                    topic_id
                )
        
        logger.info(f"Enriched graph now has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _link_extracted_terms(
        self,
        entities: list[dict],
        source_node: str,
        topic_id: str
    ):
        """Link extracted entities as term nodes to the source node."""
        for entity in entities:
            term_text = entity['text']
            term_label = entity['label']
            term_score = entity.get('score', 0.5)
            
            # Normalize the term
            normalized = self.normalizer.normalize(term_text)
            canonical = self.normalizer.get_canonical(term_text)
            
            # Skip very short or very long terms
            if len(normalized) < 2 or len(normalized) > 100:
                continue
            
            # Create term node ID
            term_node_id = f"term:{canonical}"
            
            # Add term node if not exists
            if term_node_id not in self.terms:
                self.terms[canonical] = {
                    'id': canonical,
                    'original_forms': [term_text],
                    'label': term_label,
                    'topics': [topic_id],
                    'occurrences': 1
                }
                self.graph.add_node(
                    term_node_id,
                    node_type="term",
                    normalized_text=canonical,
                    label=term_label
                )
            else:
                # Update existing term
                if term_text not in self.terms[canonical]['original_forms']:
                    self.terms[canonical]['original_forms'].append(term_text)
                if topic_id not in self.terms[canonical]['topics']:
                    self.terms[canonical]['topics'].append(topic_id)
                self.terms[canonical]['occurrences'] += 1
            
            # Link term to source node
            if not self.graph.has_edge(source_node, term_node_id):
                self.graph.add_edge(
                    source_node,
                    term_node_id,
                    relation="MENTIONS_TERM",
                    confidence=term_score
                )
    
    def link_similar_terms(self, similarity_threshold: float = 0.8):
        """Create links between similar terms."""
        logger.info("Linking similar terms...")
        
        term_list = list(self.terms.keys())
        linked_count = 0
        
        for i, term1 in enumerate(tqdm(term_list, desc="Linking terms")):
            for term2 in term_list[i+1:]:
                if self.normalizer.are_similar(term1, term2, similarity_threshold):
                    # Add bidirectional similarity edge
                    self.graph.add_edge(
                        f"term:{term1}",
                        f"term:{term2}",
                        relation="SIMILAR_TO"
                    )
                    self.graph.add_edge(
                        f"term:{term2}",
                        f"term:{term1}",
                        relation="SIMILAR_TO"
                    )
                    self.normalizer.add_synonym(term1, term2)
                    linked_count += 1
        
        logger.info(f"Created {linked_count} term similarity links")
    
    def compute_term_importance(self):
        """Compute importance scores for terms based on graph metrics."""
        logger.info("Computing term importance scores...")
        
        # Calculate various centrality metrics for term nodes
        term_nodes = [n for n in self.graph.nodes() if n.startswith("term:")]
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # PageRank (on the full graph)
        try:
            pagerank = nx.pagerank(self.graph, max_iter=100)
        except:
            pagerank = {n: 0 for n in self.graph.nodes()}
        
        # Update term nodes with importance scores
        for term_node in term_nodes:
            self.graph.nodes[term_node]['degree_centrality'] = degree_cent.get(term_node, 0)
            self.graph.nodes[term_node]['pagerank'] = pagerank.get(term_node, 0)
            
            # Combined importance score
            term_key = term_node.replace("term:", "")
            occurrences = self.terms.get(term_key, {}).get('occurrences', 1)
            topic_count = len(self.terms.get(term_key, {}).get('topics', []))
            
            importance = (
                0.3 * degree_cent.get(term_node, 0) +
                0.3 * pagerank.get(term_node, 0) * 1000 +  # Scale up PageRank
                0.2 * min(occurrences / 10, 1) +
                0.2 * min(topic_count / 5, 1)
            )
            self.graph.nodes[term_node]['importance'] = importance
        
        logger.info("Term importance scores computed")
    
    def get_topic_terms(self, topic_id: str) -> list[dict]:
        """Get all terms associated with a topic."""
        topic_node = f"topic:{topic_id}"
        
        if topic_node not in self.graph:
            return []
        
        terms = []
        for neighbor in self.graph.neighbors(topic_node):
            if neighbor.startswith("term:"):
                term_data = dict(self.graph.nodes[neighbor])
                term_data['node_id'] = neighbor
                terms.append(term_data)
        
        # Sort by importance
        terms.sort(key=lambda x: x.get('importance', 0), reverse=True)
        return terms
    
    def get_related_topics(self, topic_id: str) -> list[dict]:
        """Find topics related through shared terms."""
        topic_node = f"topic:{topic_id}"
        
        if topic_node not in self.graph:
            return []
        
        # Find terms mentioned by this topic
        topic_terms = set()
        for neighbor in self.graph.neighbors(topic_node):
            if neighbor.startswith("term:"):
                topic_terms.add(neighbor)
        
        # Find other topics that share these terms
        related_topics = defaultdict(int)
        
        for term_node in topic_terms:
            # Find all nodes that mention this term (reverse lookup)
            for node in self.graph.predecessors(term_node):
                if node.startswith("topic:") and node != topic_node:
                    related_topics[node] += 1
        
        # Sort by number of shared terms
        result = []
        for topic_node, shared_count in sorted(
            related_topics.items(), key=lambda x: x[1], reverse=True
        ):
            topic_data = dict(self.graph.nodes[topic_node])
            topic_data['node_id'] = topic_node
            topic_data['shared_terms'] = shared_count
            result.append(topic_data)
        
        return result
    
    def export_to_json(self, output_path: str):
        """Export the knowledge graph to JSON format."""
        logger.info(f"Exporting graph to JSON: {output_path}")
        
        # Convert graph to serializable format
        data = {
            'nodes': [],
            'edges': [],
            'statistics': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'books': len(self.books),
                'sections': len(self.sections),
                'topics': len(self.topics),
                'subtopics': len(self.subtopics),
                'terms': len(self.terms)
            }
        }
        
        # Export nodes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {'id': node, **attrs}
            data['nodes'].append(node_data)
        
        # Export edges
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {'source': source, 'target': target, **attrs}
            data['edges'].append(edge_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Graph exported to {output_path}")
    
    def export_to_graphml(self, output_path: str):
        """Export the knowledge graph to GraphML format."""
        logger.info(f"Exporting graph to GraphML: {output_path}")
        
        # Convert non-string attributes to strings for GraphML compatibility
        G_copy = self.graph.copy()
        for node in G_copy.nodes():
            for key, value in list(G_copy.nodes[node].items()):
                if not isinstance(value, (str, int, float, bool)):
                    G_copy.nodes[node][key] = str(value)
        
        for u, v in G_copy.edges():
            for key, value in list(G_copy.edges[u, v].items()):
                if not isinstance(value, (str, int, float, bool)):
                    G_copy.edges[u, v][key] = str(value)
        
        nx.write_graphml(G_copy, output_path)
        logger.info(f"Graph exported to {output_path}")
    
    def export_to_gexf(self, output_path: str):
        """Export the knowledge graph to GEXF format (for Gephi)."""
        logger.info(f"Exporting graph to GEXF: {output_path}")
        
        # Convert non-string attributes
        G_copy = self.graph.copy()
        for node in G_copy.nodes():
            for key, value in list(G_copy.nodes[node].items()):
                if not isinstance(value, (str, int, float, bool)):
                    G_copy.nodes[node][key] = str(value)
        
        for u, v in G_copy.edges():
            for key, value in list(G_copy.edges[u, v].items()):
                if not isinstance(value, (str, int, float, bool)):
                    G_copy.edges[u, v][key] = str(value)
        
        nx.write_gexf(G_copy, output_path)
        logger.info(f"Graph exported to {output_path}")
    
    def export_terms_report(self, output_path: str):
        """Export a report of extracted terms."""
        logger.info(f"Exporting terms report: {output_path}")
        
        terms_data = []
        for term_id, term_info in self.terms.items():
            term_node = f"term:{term_id}"
            node_data = self.graph.nodes.get(term_node, {})
            
            terms_data.append({
                'term': term_id,
                'original_forms': term_info.get('original_forms', []),
                'label': term_info.get('label', ''),
                'occurrences': term_info.get('occurrences', 0),
                'topic_count': len(term_info.get('topics', [])),
                'topics': term_info.get('topics', []),
                'importance': node_data.get('importance', 0),
                'pagerank': node_data.get('pagerank', 0),
                'degree_centrality': node_data.get('degree_centrality', 0)
            })
        
        # Sort by importance
        terms_data.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(terms_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Terms report exported to {output_path}")
        
        return terms_data
    
    def print_statistics(self):
        """Print graph statistics."""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*60)
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        print(f"\nNode breakdown:")
        print(f"  - Books: {len(self.books)}")
        print(f"  - Sections: {len(self.sections)}")
        print(f"  - Topics: {len(self.topics)}")
        print(f"  - Subtopics: {len(self.subtopics)}")
        print(f"  - Terms: {len(self.terms)}")
        
        # Count edges by type
        edge_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_types[data.get('relation', 'unknown')] += 1
        
        print(f"\nEdge breakdown:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"  - {edge_type}: {count}")
        
        # Top terms
        if self.terms:
            print(f"\nTop 10 most important terms:")
            term_nodes = [(n, self.graph.nodes[n].get('importance', 0)) 
                         for n in self.graph.nodes() if n.startswith("term:")]
            term_nodes.sort(key=lambda x: x[1], reverse=True)
            for term_node, importance in term_nodes[:10]:
                term_text = term_node.replace("term:", "")
                print(f"  - {term_text}: {importance:.4f}")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from educational parquet files"
    )
    parser.add_argument(
        "--topics",
        type=str,
        required=True,
        help="Path to topics parquet file"
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Path to pages parquet file (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="knowledge_graph",
        help="Output prefix for exported files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="knowledgator/gliner-x-large",
        help="GLiNER model name"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Entity extraction confidence threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (higher = faster but more memory)"
    )
    parser.add_argument(
        "--skip-term-extraction",
        action="store_true",
        help="Skip term extraction (only build structural graph)"
    )
    parser.add_argument(
        "--link-similar-terms",
        action="store_true",
        help="Create links between similar terms"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Threshold for term similarity linking"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    extractor = EntityExtractor(
        model_name=args.model,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size
    )
    normalizer = TermNormalizer(language="uk")
    
    # Build the graph
    builder = KnowledgeGraphBuilder(extractor, normalizer)
    
    # Load and process topics data
    topics_df = builder.load_topics_data(args.topics)
    builder.build_graph_from_topics(
        topics_df,
        extract_terms=not args.skip_term_extraction
    )
    
    # Load and process pages data if provided
    if args.pages:
        pages_df = builder.load_pages_data(args.pages)
        builder.build_graph_from_pages(
            pages_df,
            extract_terms=not args.skip_term_extraction
        )
    
    # Link similar terms if requested
    if args.link_similar_terms and not args.skip_term_extraction:
        builder.link_similar_terms(args.similarity_threshold)
    
    # Compute term importance
    if not args.skip_term_extraction:
        builder.compute_term_importance()
    
    # Print statistics
    builder.print_statistics()
    
    # Export in multiple formats
    output_dir = Path(args.output).parent
    output_prefix = Path(args.output).stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    builder.export_to_json(f"{args.output}.json")
    builder.export_to_graphml(f"{args.output}.graphml")
    builder.export_to_gexf(f"{args.output}.gexf")
    
    if not args.skip_term_extraction:
        builder.export_terms_report(f"{args.output}_terms.json")
    
    logger.info("Knowledge graph building complete!")


if __name__ == "__main__":
    main()