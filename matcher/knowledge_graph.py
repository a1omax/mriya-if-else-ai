"""
Knowledge Graph utilities for searching and linking nodes
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from collections import defaultdict

from models import (
    KnowledgeGraph, KnowledgeNode, TopicNode, SectionNode, BookNode,
    Edge, TopicMatch, NodeType, RelationType
)


class KnowledgeGraphManager:
    """Manager for knowledge graph operations"""
    
    def __init__(self, knowledge_graph: Optional[KnowledgeGraph] = None):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Edge] = []
        self.node_index: Dict[NodeType, List[str]] = defaultdict(list)
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # node_id -> [(neighbor_id, relation)]
        self.reverse_adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        if knowledge_graph:
            self.load_graph(knowledge_graph)
    
    def load_graph(self, knowledge_graph: KnowledgeGraph):
        """Load knowledge graph into memory with indices"""
        # Load nodes
        for node_data in knowledge_graph.nodes:
            node_id = node_data.get("id", "")
            self.nodes[node_id] = node_data
            
            # Index by type
            node_type_str = node_data.get("node_type", "")
            try:
                node_type = NodeType(node_type_str)
                self.node_index[node_type].append(node_id)
            except ValueError:
                pass
        
        # Load edges and build adjacency
        for edge in knowledge_graph.edges:
            self.edges.append(edge)
            self.adjacency[edge.source].append((edge.target, edge.relation))
            self.reverse_adjacency[edge.target].append((edge.source, edge.relation))
    
    def load_from_json(self, json_path: str):
        """Load knowledge graph from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get("nodes", [])
        edges_data = data.get("edges", [])
        
        edges = []
        for e in edges_data:
            try:
                edges.append(Edge(
                    source=e["source"],
                    target=e["target"],
                    relation=e["relation"],
                    confidence=e.get("confidence")
                ))
            except Exception:
                continue
        
        kg = KnowledgeGraph(nodes=nodes, edges=edges)
        self.load_graph(kg)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type"""
        return [self.nodes[nid] for nid in self.node_index.get(node_type, [])]
    
    def get_all_topics(self) -> List[Dict[str, Any]]:
        """Get all topic nodes"""
        return self.get_nodes_by_type(NodeType.TOPIC)
    
    def get_neighbors(self, node_id: str, relation_filter: Optional[RelationType] = None) -> List[Dict[str, Any]]:
        """Get neighboring nodes"""
        neighbors = []
        for neighbor_id, relation in self.adjacency.get(node_id, []):
            if relation_filter is None or relation == relation_filter:
                node = self.get_node(neighbor_id)
                if node:
                    neighbors.append(node)
        return neighbors
    
    def get_related_terms(self, topic_id: str) -> List[str]:
        """Get terms mentioned in a topic"""
        terms = []
        for neighbor_id, relation in self.adjacency.get(f"topic:{topic_id}", []):
            if relation == RelationType.MENTIONS_TERM:
                # Extract term from ID (format: "term:term_name")
                if neighbor_id.startswith("term:"):
                    terms.append(neighbor_id[5:])
                else:
                    node = self.get_node(neighbor_id)
                    if node:
                        terms.append(node.get("name", node.get("title", neighbor_id)))
        return terms
    
    def find_parent_section(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Find the section containing a topic"""
        topic = self.get_node(topic_id)
        if topic and "section_id" in topic:
            return self.get_node(topic["section_id"])
        return None
    
    def find_parent_book(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Find the book containing a node"""
        node = self.get_node(node_id)
        if node and "book_id" in node:
            return self.get_node(node["book_id"])
        return None


class TopicMatcher:
    """Matches suggested topics to knowledge graph topics"""
    
    def __init__(self, kg_manager: KnowledgeGraphManager):
        self.kg = kg_manager
        self.topics = self.kg.get_all_topics()
        self._build_search_index()
    
    def _build_search_index(self):
        """Build search index for topics"""
        self.title_index: Dict[str, str] = {}  # lowercase title -> topic_id
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)  # keyword -> [topic_ids]
        
        for topic in self.topics:
            topic_id = topic.get("id", "")
            title = str(topic.get("title", ""))
            
            # Index by full title
            self.title_index[title.lower()] = topic_id
            
            # Index by keywords
            keywords = self._extract_keywords(title)
            for kw in keywords:
                self.keyword_index[kw.lower()].append(topic_id)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common words and punctuation
        stop_words = {'і', 'та', 'або', 'в', 'на', 'з', 'до', 'для', 'як', 'що', 'це'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate string similarity using SequenceMatcher"""
        return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()
    
    def find_exact_match(self, suggested_topic: str) -> Optional[TopicMatch]:
        """Find exact title match"""
        topic_id = self.title_index.get(suggested_topic.lower())
        if topic_id:
            topic = self.kg.get_node(topic_id)
            if topic:
                return TopicMatch(
                    topic_id=topic_id,
                    topic_title=topic.get("title", ""),
                    similarity_score=1.0,
                    match_type="exact"
                )
        return None
    
    def find_fuzzy_matches(self, suggested_topic: str, threshold: float = 0.6) -> List[TopicMatch]:
        """Find fuzzy matches based on title similarity"""
        matches = []
        
        for topic in self.topics:
            topic_id = topic.get("id", "")
            title = topic.get("title", "")
            
            similarity = self._calculate_similarity(suggested_topic, title)
            
            if similarity >= threshold:
                matches.append(TopicMatch(
                    topic_id=topic_id,
                    topic_title=title,
                    similarity_score=similarity,
                    match_type="fuzzy"
                ))
        
        # Sort by similarity
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:5]  # Top 5 matches
    
    def find_keyword_matches(self, suggested_topic: str, min_keywords: int = 2) -> List[TopicMatch]:
        """Find matches based on keyword overlap"""
        keywords = self._extract_keywords(suggested_topic)
        topic_scores: Dict[str, int] = defaultdict(int)
        
        for kw in keywords:
            for topic_id in self.keyword_index.get(kw, []):
                topic_scores[topic_id] += 1
        
        matches = []
        for topic_id, score in topic_scores.items():
            if score >= min_keywords:
                topic = self.kg.get_node(topic_id)
                if topic:
                    # Normalize score
                    max_possible = len(keywords)
                    similarity = score / max_possible if max_possible > 0 else 0
                    
                    matches.append(TopicMatch(
                        topic_id=topic_id,
                        topic_title=topic.get("title", ""),
                        similarity_score=similarity,
                        match_type="keyword"
                    ))
        
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:5]
    
    def find_best_matches(self, suggested_topic: str) -> List[TopicMatch]:
        """Find best matches using all strategies"""
        all_matches: Dict[str, TopicMatch] = {}
        
        # Try exact match first
        exact = self.find_exact_match(suggested_topic)
        if exact:
            all_matches[exact.topic_id] = exact
        
        # Add fuzzy matches
        for match in self.find_fuzzy_matches(suggested_topic):
            if match.topic_id not in all_matches:
                all_matches[match.topic_id] = match
            elif match.similarity_score > all_matches[match.topic_id].similarity_score:
                all_matches[match.topic_id] = match
        
        # Add keyword matches
        for match in self.find_keyword_matches(suggested_topic):
            if match.topic_id not in all_matches:
                all_matches[match.topic_id] = match
        
        # Sort by score and return
        matches = list(all_matches.values())
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches
    
    def enrich_match_with_relations(self, match: TopicMatch) -> TopicMatch:
        """Add related nodes to a match"""
        # Get related terms
        related_terms = self.kg.get_related_terms(match.topic_id)
        
        # Get parent section and book
        section = self.kg.find_parent_section(match.topic_id)
        book = self.kg.find_parent_book(match.topic_id)
        
        related_nodes = []
        if section:
            related_nodes.append(f"section:{section.get('title', section.get('id'))}")
        if book:
            related_nodes.append(f"book:{book.get('name', book.get('id'))}")
        related_nodes.extend([f"term:{t}" for t in related_terms])
        
        match.related_nodes = related_nodes
        return match


def find_similar_nodes(
    kg_manager: KnowledgeGraphManager,
    target_terms: List[str],
    node_type: Optional[NodeType] = None
) -> List[Dict[str, Any]]:
    """
    Find nodes in knowledge graph similar to given terms
    """
    similar_nodes = []
    all_nodes = kg_manager.nodes.values() if node_type is None else kg_manager.get_nodes_by_type(node_type)
    
    for node in all_nodes:
        node_text = node.get("title", "") or node.get("name", "") or ""
        
        for term in target_terms:
            # Check for substring match
            if term.lower() in node_text.lower():
                similar_nodes.append({
                    "node": node,
                    "matched_term": term,
                    "match_type": "substring"
                })
                break
            
            # Check for fuzzy match
            similarity = SequenceMatcher(None, term.lower(), node_text.lower()).ratio()
            if similarity > 0.7:
                similar_nodes.append({
                    "node": node,
                    "matched_term": term,
                    "match_type": "fuzzy",
                    "similarity": similarity
                })
                break
    
    return similar_nodes


def create_topic_links(
    kg_manager: KnowledgeGraphManager,
    suggested_topics: List[str],
    weak_terms: List[str],
    weak_concepts: List[str]
) -> Dict[str, Any]:
    """
    Create links between suggested topics and knowledge graph nodes
    """
    matcher = TopicMatcher(kg_manager)
    
    results = {
        "topic_matches": [],
        "term_links": [],
        "concept_links": [],
        "recommended_learning_path": []
    }
    
    # Match suggested topics
    for topic in suggested_topics:
        matches = matcher.find_best_matches(topic)
        for match in matches:
            match = matcher.enrich_match_with_relations(match)
            results["topic_matches"].append({
                "suggested": topic,
                "matched": match.dict()
            })
    
    # Find similar nodes for weak terms
    term_nodes = find_similar_nodes(kg_manager, weak_terms)
    results["term_links"] = term_nodes
    
    # Find similar nodes for weak concepts
    concept_nodes = find_similar_nodes(kg_manager, weak_concepts)
    results["concept_links"] = concept_nodes
    
    # Build learning path from matched topics
    matched_topic_ids = [m["matched"]["topic_id"] for m in results["topic_matches"] if m["matched"]]
    for topic_id in matched_topic_ids:
        topic = kg_manager.get_node(topic_id)
        if topic:
            section = kg_manager.find_parent_section(topic_id)
            book = kg_manager.find_parent_book(topic_id)
            
            path_item = {
                "topic": topic.get("title", ""),
                "topic_id": topic_id,
                "section": section.get("title", "") if section else None,
                "book": book.get("name", "") if book else None,
                "pages": f"{topic.get('start_page', '')}-{topic.get('end_page', '')}" if topic.get('start_page') else None
            }
            results["recommended_learning_path"].append(path_item)
    
    return results