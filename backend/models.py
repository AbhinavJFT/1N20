"""
Pydantic models for request/response schemas and context management
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
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

class ProductImage(BaseModel):
    """Schema for a product image with description."""
    url: str = Field(description="Image URL/path")
    description: str = Field(description="What this image shows - helps agent decide relevance")
    category: str = Field(default="general", description="Image category (primary, styles, finishes, details, etc.)")


class RelevantImage(BaseModel):
    """Schema for an image to be rendered in the frontend."""
    url: str = Field(description="Image URL/path from search results")
    description: str = Field(description="Brief description of what the image shows")


class SalesAgentResponse(BaseModel):
    """Structured output schema for Sales Agent responses.

    This ensures the agent returns both text and images in a predictable format
    that the frontend can easily parse and render.
    """
    response: str = Field(description="The text response to show to the customer")
    images: List[RelevantImage] = Field(
        default_factory=list,
        description="List of relevant product images to display. Select 1-3 images that are most relevant to what the customer asked about."
    )


class ProductSearchResult(BaseModel):
    """Schema for product search results from Pinecone metadata.

    NOTE: Description field is intentionally excluded - it's only used for
    embedding/search, not for response context. All product info is in
    structured metadata fields below.
    """
    # Product identification
    product_id: str = Field(description="Unique identifier for the product")
    name: str = Field(description="Product display name")
    series: Optional[str] = Field(default=None, description="Product series/line (Embarq, Signet, etc.)")
    category: str = Field(description="Product category (Entry Door, Storm Door, Accessories, etc.)")
    subcategory: Optional[str] = Field(default=None, description="Product subcategory (Pet Access, Decorative Iron, etc.)")
    tier: Optional[str] = Field(default=None, description="Product tier (Premium, Standard, Security, etc.)")

    # Key features - primary selling points
    key_features: List[str] = Field(default_factory=list, description="List of key product features")

    # Energy specifications
    energy_star: bool = Field(default=False, description="Energy Star certified")
    u_factor: Optional[str] = Field(default=None, description="U-factor rating (lower = better efficiency)")
    energy_certification: Optional[str] = Field(default=None, description="Energy certification details")

    # Customization options
    skin_options: List[str] = Field(default_factory=list, description="Available skin/wood options (Mahogany, Cherry, Oak, etc.)")
    door_styles: List[str] = Field(default_factory=list, description="Available door style codes")
    glass_packages: List[str] = Field(default_factory=list, description="Available glass packages (ComforTech QLK, QLA, etc.)")
    decorative_glass: List[str] = Field(default_factory=list, description="Compatible decorative glass codes")
    compatible_frames: List[str] = Field(default_factory=list, description="Compatible frame types (FrameSaver, FusionFrame, etc.)")
    sidelites: List[str] = Field(default_factory=list, description="Available sidelite options")

    # For accessories
    brands: List[str] = Field(default_factory=list, description="Associated brands (for accessories)")

    # Search metadata
    search_tags: List[str] = Field(default_factory=list, description="Search tags for this product")

    # Detailed JSON data (parsed from _raw_ fields)
    finishes: Optional[dict] = Field(default=None, description="Detailed finish options (stain, paint, glazed)")
    hardware: Optional[Any] = Field(default=None, description="Compatible hardware info")
    warranty: Optional[dict] = Field(default=None, description="Warranty coverage details")
    restrictions: Optional[Any] = Field(default=None, description="Product restrictions and limitations")
    installation: Optional[dict] = Field(default=None, description="Installation requirements (for accessories)")

    # Images - consolidated list with descriptions for agent to select relevant ones
    images: List[ProductImage] = Field(default_factory=list, description="All product images with descriptions - agent selects relevant ones based on user query")

    # Links and references
    product_url: Optional[str] = Field(default=None, description="Product page URL")
    source_pages: List[str] = Field(default_factory=list, description="Catalog source page references")

    # Relevance score
    relevance_score: float = Field(default=0.0, description="Search relevance score (0-1)")

    # Additional metadata not captured by standard fields (chunk-specific info)
    additional_info: Optional[dict] = Field(default=None, description="Extra metadata fields unique to this product/chunk")

    # Fields that were expected but not available in metadata
    unavailable_fields: List[str] = Field(default_factory=list, description="List of fields not available for this product - contact team for details")


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
