"""
Data models for the Student Analysis Agent
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    BOOK = "book"
    SECTION = "section"
    TOPIC = "topic"
    TERM = "term"
    PAGE = "page"
    CONCEPT = "concept"
    SKILL = "skill"


class RelationType(str, Enum):
    MENTIONS_TERM = "MENTIONS_TERM"
    CONTAINS_PAGE = "CONTAINS_PAGE"
    CONTAINS_TOPIC = "CONTAINS_TOPIC"
    CONTAINS_SECTION = "CONTAINS_SECTION"
    RELATED_TO = "RELATED_TO"
    PREREQUISITE_FOR = "PREREQUISITE_FOR"
    SIMILAR_TO = "SIMILAR_TO"


# Knowledge Graph Models
class KnowledgeNode(BaseModel):
    """Base model for knowledge graph nodes"""
    id: str
    node_type: NodeType
    name: Optional[str] = None
    title: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        return self.title or self.name or self.id


class BookNode(KnowledgeNode):
    """Book node in knowledge graph"""
    node_type: NodeType = NodeType.BOOK
    grade: Optional[int] = None
    discipline: Optional[str] = None


class SectionNode(KnowledgeNode):
    """Section node in knowledge graph"""
    node_type: NodeType = NodeType.SECTION
    book_id: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class TopicNode(KnowledgeNode):
    """Topic node in knowledge graph"""
    node_type: NodeType = NodeType.TOPIC
    type: Optional[str] = None  # theoretical, practical, etc.
    summary: Optional[str] = None
    section_id: Optional[str] = None
    book_id: Optional[str] = None
    start_page: Optional[float] = None
    end_page: Optional[float] = None


class Edge(BaseModel):
    """Edge in knowledge graph"""
    source: str
    target: str
    relation: RelationType
    confidence: Optional[float] = None


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph structure"""
    nodes: List[Dict[str, Any]]
    edges: List[Edge]


# Student Analysis Models
class StudentResponse(BaseModel):
    """Input: Student's response to analyze"""
    question: str = Field(description="The original question asked")
    correct_answer: Optional[str] = Field(default=None, description="The correct answer if available")
    student_answer: str = Field(description="Student's answer")
    student_explanation: Optional[str] = Field(default=None, description="Student's explanation for their answer")


class AnalysisResult(BaseModel):
    """Output: Structured analysis of student's response"""
    is_correct: bool = Field(description="Whether the answer is correct")
    score: int = Field(ge=0, le=100, description="Score from 0-100")
    summary: str = Field(description="Brief summary of the analysis")
    mistakes: List[str] = Field(default_factory=list, description="List of mistakes made")
    weak_terms: List[str] = Field(default_factory=list, description="Terms the student struggles with")
    weak_concepts: List[str] = Field(default_factory=list, description="Concepts needing reinforcement")
    weak_skills: List[str] = Field(default_factory=list, description="Skills needing practice")
    correct_aspects: List[str] = Field(default_factory=list, description="What the student got right")
    suggested_topics: List[str] = Field(default_factory=list, description="Topics to study")
    explanation: str = Field(description="Detailed explanation of the correct approach")
    encouragement: str = Field(description="Encouraging message for the student")


class TopicMatch(BaseModel):
    """A matched topic from knowledge graph"""
    topic_id: str
    topic_title: str
    similarity_score: float
    match_type: str  # exact, fuzzy, semantic
    related_nodes: List[str] = Field(default_factory=list)


class TopicRecommendation(BaseModel):
    """Final topic recommendation with linked resources"""
    suggested_topic: str
    matched_topics: List[TopicMatch]
    related_terms: List[str] = Field(default_factory=list)
    related_sections: List[str] = Field(default_factory=list)
    learning_path: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """State passed through the LangGraph agent"""
    # Input
    student_response: StudentResponse
    knowledge_graph: Optional[KnowledgeGraph] = None
    
    # Intermediate states
    analysis_result: Optional[AnalysisResult] = None
    topic_matches: List[TopicMatch] = Field(default_factory=list)
    
    # Output
    recommendations: List[TopicRecommendation] = Field(default_factory=list)
    
    # Metadata
    processing_steps: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True