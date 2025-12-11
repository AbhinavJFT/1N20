"""
MongoDB database connection and Lead model for the Voice Sales Agent
"""

from datetime import datetime
from typing import Optional, List
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from bson import ObjectId

from config import config


# =============================================================================
# MongoDB Connection
# =============================================================================

class MongoDB:
    """MongoDB connection manager using Motor (async driver)."""

    client: Optional[AsyncIOMotorClient] = None
    database = None

    @classmethod
    async def connect(cls):
        """Connect to MongoDB."""
        if cls.client is None:
            cls.client = AsyncIOMotorClient(config.MONGODB_URI)
            cls.database = cls.client[config.MONGODB_DATABASE]
            print(f"Connected to MongoDB: {config.MONGODB_DATABASE}")

    @classmethod
    async def disconnect(cls):
        """Disconnect from MongoDB."""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.database = None
            print("Disconnected from MongoDB")

    @classmethod
    def get_database(cls):
        """Get the database instance."""
        if cls.database is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return cls.database

    @classmethod
    def get_collection(cls, collection_name: str):
        """Get a collection from the database."""
        return cls.get_database()[collection_name]


# =============================================================================
# Lead Model
# =============================================================================

class LeadDocument(BaseModel):
    """Lead document model for MongoDB storage."""

    # Customer Information
    name: str = Field(description="Customer's full name")
    email: str = Field(description="Customer's email address")
    phone: str = Field(description="Customer's phone number")

    # Product Information
    selected_product: Optional[str] = Field(default=None, description="Final product selection")
    products_discussed: List[str] = Field(default_factory=list, description="List of products discussed")

    # Conversation Details
    conversation_summary: str = Field(default="", description="Summary of the conversation")

    # Metadata
    session_id: Optional[str] = Field(default=None, description="Voice session ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the lead was created")
    email_sent: bool = Field(default=False, description="Whether lead email was sent")
    email_sent_at: Optional[datetime] = Field(default=None, description="When email was sent")
    status: str = Field(default="new", description="Lead status: new, contacted, converted, closed")

    class Config:
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# Lead Repository (Database Operations)
# =============================================================================

class LeadRepository:
    """Repository for Lead database operations."""

    COLLECTION_NAME = "leads"

    @classmethod
    async def create_lead(cls, lead: LeadDocument) -> str:
        """
        Create a new lead in the database.

        Returns:
            str: The inserted document ID
        """
        collection = MongoDB.get_collection(cls.COLLECTION_NAME)
        lead_dict = lead.model_dump()
        result = await collection.insert_one(lead_dict)
        return str(result.inserted_id)

    @classmethod
    async def get_lead_by_id(cls, lead_id: str) -> Optional[dict]:
        """Get a lead by its ID."""
        collection = MongoDB.get_collection(cls.COLLECTION_NAME)
        return await collection.find_one({"_id": ObjectId(lead_id)})

    @classmethod
    async def get_lead_by_email(cls, email: str) -> Optional[dict]:
        """Get a lead by email address."""
        collection = MongoDB.get_collection(cls.COLLECTION_NAME)
        return await collection.find_one({"email": email})

    @classmethod
    async def get_lead_by_session(cls, session_id: str) -> Optional[dict]:
        """Get a lead by session ID."""
        collection = MongoDB.get_collection(cls.COLLECTION_NAME)
        return await collection.find_one({"session_id": session_id})

    @classmethod
    async def update_email_sent(cls, lead_id: str) -> bool:
        """Mark a lead as having email sent."""
        collection = MongoDB.get_collection(cls.COLLECTION_NAME)
        result = await collection.update_one(
            {"_id": ObjectId(lead_id)},
            {
                "$set": {
                    "email_sent": True,
                    "email_sent_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    @classmethod
    async def update_lead_status(cls, lead_id: str, status: str) -> bool:
        """Update the status of a lead."""
        collection = MongoDB.get_collection(cls.COLLECTION_NAME)
        result = await collection.update_one(
            {"_id": ObjectId(lead_id)},
            {"$set": {"status": status}}
        )
        return result.modified_count > 0

    @classmethod
    async def get_all_leads(cls, limit: int = 100, skip: int = 0) -> List[dict]:
        """Get all leads with pagination."""
        collection = MongoDB.get_collection(cls.COLLECTION_NAME)
        cursor = collection.find().sort("created_at", -1).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)
