"""
Pydantic models for request/response schemas and context management
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Customer Context (shared across agents)
# =============================================================================

@dataclass
class CustomerContext:
    """Stores customer information collected during the conversation."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    products_discussed: List[str] = field(default_factory=list)
    conversation_summary: str = ""
    selected_product: Optional[str] = None
    info_collection_complete: bool = False


# =============================================================================
# Conversation History
# =============================================================================

class ConversationMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(description="Message role: user or assistant")
    content: str = Field(description="Message content")
    agent: Optional[str] = Field(default=None, description="Agent name if assistant")
    timestamp: str = Field(description="ISO timestamp")


# =============================================================================
# Pydantic Response Schemas
# =============================================================================

class ProductSearchResult(BaseModel):
    """Schema for product search results from Pinecone."""
    product_id: str = Field(description="Unique identifier for the product")
    name: str = Field(description="Product name")
    series: Optional[str] = Field(default=None, description="Product series/line")
    category: str = Field(description="Product category (e.g., Entry Door)")
    tier: Optional[str] = Field(default=None, description="Product tier (Premium, Standard, etc.)")
    description: str = Field(description="Product description")
    key_features: List[str] = Field(default_factory=list, description="Key product features")
    door_style_codes: List[str] = Field(default_factory=list, description="Available door style codes")
    skin_options: List[str] = Field(default_factory=list, description="Available skin/wood options")
    compatible_frames: List[str] = Field(default_factory=list, description="Compatible frame types")
    glass_packages: List[str] = Field(default_factory=list, description="Available glass package names")
    decorative_glass_codes: List[str] = Field(default_factory=list, description="Compatible decorative glass codes")
    energy_star: bool = Field(default=False, description="Energy Star certified")
    u_factor: Optional[str] = Field(default=None, description="U-factor rating for energy efficiency")
    product_url: Optional[str] = Field(default=None, description="Product page URL")
    relevance_score: float = Field(description="Search relevance score")
    raw_finishes: Optional[str] = Field(default=None, description="Raw JSON of compatible finishes")
    raw_hardware: Optional[str] = Field(default=None, description="Raw JSON of compatible hardware")
    raw_warranty: Optional[str] = Field(default=None, description="Raw JSON of warranty info")
    raw_restrictions: Optional[str] = Field(default=None, description="Raw JSON of restrictions")


class ProductSearchResponse(BaseModel):
    """Schema for the complete search response."""
    query: str = Field(description="The search query used")
    results: List[ProductSearchResult] = Field(description="List of matching products")
    total_results: int = Field(description="Total number of results found")


class CustomerInfoStatus(BaseModel):
    """Schema for customer info collection status."""
    name_collected: bool = False
    email_collected: bool = False
    phone_collected: bool = False
    all_collected: bool = False
    missing_fields: List[str] = Field(default_factory=list)


class LeadEmailResponse(BaseModel):
    """Schema for lead email sending response."""
    success: bool = Field(description="Whether email was sent successfully")
    message: str = Field(description="Status message")
    recipient: Optional[str] = Field(default=None, description="Email recipient")


class LeadSubmissionResponse(BaseModel):
    """Schema for lead submission response (DB save + email via queue)."""
    success: bool = Field(description="Whether lead was submitted to queue successfully")
    message: str = Field(description="Status message for the customer")
    tasks_queued: int = Field(default=0, description="Number of tasks queued for processing")


# =============================================================================
# WebSocket Message Types
# =============================================================================

class MessageType(str, Enum):
    """Types of WebSocket messages."""
    # Client -> Server
    AUDIO_INPUT = "audio_input"
    TEXT_INPUT = "text_input"
    START_SESSION = "start_session"
    END_SESSION = "end_session"

    # Server -> Client
    AUDIO_OUTPUT = "audio_output"
    TRANSCRIPT = "transcript"
    AGENT_MESSAGE = "agent_message"
    USER_TRANSCRIPT = "user_transcript"
    PARTIAL_TRANSCRIPT = "partial_transcript"  # Real-time typing display
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    HANDOFF = "handoff"
    ERROR = "error"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    CONTEXT_UPDATE = "context_update"
    AGENT_SPEAKING = "agent_speaking"  # Agent is generating audio
    AGENT_DONE = "agent_done"  # Agent finished speaking


class WebSocketMessage(BaseModel):
    """Base WebSocket message schema."""
    type: MessageType
    data: dict = Field(default_factory=dict)
    timestamp: Optional[str] = None


# =============================================================================
# Guardrail Schemas
# =============================================================================

class DomainValidationResult(BaseModel):
    """Schema for domain validation guardrail."""
    is_valid: bool = Field(description="Whether the query is within domain")
    reason: str = Field(description="Explanation for the validation result")
    suggested_response: Optional[str] = Field(
        default=None,
        description="Suggested response if query is out of domain"
    )
