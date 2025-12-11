"""
Tools for the Voice Sales Agent
- Customer info collection tools
- Pinecone RAG search tool
- Lead submission tool (DB + Email via queue)
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

from openai import OpenAI
from pinecone import Pinecone
from agents import function_tool, RunContextWrapper

from config import config
from models import (
    CustomerContext,
    ProductSearchResponse,
    ProductSearchResult,
    CustomerInfoStatus,
    LeadEmailResponse,
)
from task_queue import submit_lead_to_queue


# =============================================================================
# Initialize Clients
# =============================================================================

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)
pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)


# =============================================================================
# Helper Functions
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a text query using OpenAI."""
    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# =============================================================================
# Customer Information Tools (for Data Collection Agent)
# =============================================================================

@function_tool
def save_customer_name(
    context: RunContextWrapper[CustomerContext],
    name: str,
) -> str:
    """
    Save the customer's name.
    Call this when the customer provides their name.

    Args:
        name: Customer's full name
    """
    context.context.name = name
    return f"Saved customer name: {name}"


@function_tool
def save_customer_email(
    context: RunContextWrapper[CustomerContext],
    email: str,
) -> str:
    """
    Save the customer's email address.
    Call this when the customer provides their email.

    Args:
        email: Customer's email address
    """
    context.context.email = email
    return f"Saved customer email: {email}"


@function_tool
def save_customer_phone(
    context: RunContextWrapper[CustomerContext],
    phone: str,
) -> str:
    """
    Save the customer's phone number.
    Call this when the customer provides their phone number.

    Args:
        phone: Customer's phone number
    """
    context.context.phone = phone
    return f"Saved customer phone: {phone}"


@function_tool
def check_customer_info_complete(
    context: RunContextWrapper[CustomerContext],
) -> CustomerInfoStatus:
    """
    Check if all required customer information has been collected.
    Call this to verify before handing off to the Sales Agent.
    Returns the status of each field and whether all are collected.
    """
    ctx = context.context

    missing = []
    if not ctx.name:
        missing.append("name")
    if not ctx.email:
        missing.append("email")
    if not ctx.phone:
        missing.append("phone")

    all_collected = len(missing) == 0

    if all_collected:
        ctx.info_collection_complete = True

    return CustomerInfoStatus(
        name_collected=ctx.name is not None,
        email_collected=ctx.email is not None,
        phone_collected=ctx.phone is not None,
        all_collected=all_collected,
        missing_fields=missing
    )


# =============================================================================
# Pinecone RAG Tool (for Sales Agent)
# =============================================================================

@function_tool
def search_products(query: str) -> ProductSearchResponse:
    """
    Search the ProVia product catalog for doors and windows using semantic search.
    THIS TOOL MUST BE CALLED for every customer query about products.

    HOW TO FORMULATE EFFECTIVE QUERIES:
    The vector database contains product chunks with: description, key_features, search_tags, series name.
    To get the best results, include relevant terms from these categories:

    1. PRODUCT TYPES & SERIES:
       - "embarq fiberglass entry door" (premium, highest efficiency)
       - "signet fiberglass entry door" (high-end traditional)
       - "heritage fiberglass entry door" (classic styles)
       - "endure fiberglass entry door" (durable, value)
       - "doors without glass" (solid panel, privacy)
       - "steel entry door" (security focused)

    2. FEATURES & BENEFITS (use these terms):
       - Energy: "energy efficient", "quad glass", "r-10", "low u-factor", "energy star"
       - Construction: "fiberglass", "steel", "2.5 inch thick", "dovetailed construction"
       - Glass: "decorative glass", "full lite", "half lite", "privacy glass", "no glass"
       - Style: "modern", "traditional", "craftsman", "contemporary", "rustic"
       - Finish: "stain finish", "paint finish", "oak", "mahogany", "cherry", "knotty alder"

    3. SPECIFIC NEEDS:
       - "premium door highest efficiency" → finds Embarq
       - "traditional wood look fiberglass" → finds Signet/Heritage
       - "solid panel maximum privacy" → finds doors without glass
       - "decorative glass options entry door" → finds doors with glass compatibility

    Args:
        query: Search query combining product type + features + customer needs.
               BE SPECIFIC and use multiple descriptive terms.
               Examples:
               - "premium fiberglass entry door quad glass energy efficient"
               - "traditional entry door decorative glass mahogany stain"
               - "modern contemporary door full lite glass"
               - "solid panel door no glass maximum privacy"
               - "craftsman style fiberglass door with sidelites"

    Returns:
        ProductSearchResponse with matching products including:
        - Product name, series, tier, category
        - Key features and description
        - Available door styles, skin options, glass packages
        - Compatible frames, hardware, finishes
        - Energy ratings (Energy Star, U-factor)
        - Warranty and restriction info
    """
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query)

        # Query Pinecone (index: doorindex, namespace: doors)
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="doors"
        )

        # Format results with comprehensive metadata
        product_results = []
        for match in results.matches:
            metadata = match.metadata if hasattr(match, 'metadata') and match.metadata else {}

            # Product name from product_id, formatted nicely
            product_id = metadata.get('product_id', match.id)
            product_name = product_id.replace('_', ' ').title()

            # Extract list fields (handle both list and string formats)
            def get_list(key: str) -> list:
                val = metadata.get(key, [])
                if isinstance(val, list):
                    # Filter out complex JSON strings from skin_options etc.
                    return [item for item in val if isinstance(item, str) and not item.startswith('{')]
                return []

            # Build the product result with all available metadata
            product = ProductSearchResult(
                product_id=match.id,
                name=product_name,
                series=metadata.get('series'),
                category=metadata.get('category', 'Entry Door'),
                tier=metadata.get('tier'),
                description=metadata.get('description', 'No description available'),
                key_features=get_list('key_features'),
                door_style_codes=get_list('door_style_codes'),
                skin_options=get_list('skin_options'),
                compatible_frames=get_list('compatible_frames'),
                glass_packages=get_list('glass_package_names'),
                decorative_glass_codes=get_list('compatible_decorative_glass'),
                energy_star=metadata.get('energy_star', False),
                u_factor=metadata.get('u_factor'),
                product_url=metadata.get('product_url'),
                relevance_score=round(match.score, 3) if hasattr(match, 'score') else 0.0,
                # Include raw JSON for detailed queries about finishes, hardware, warranty
                raw_finishes=metadata.get('_raw_compatible_finishes'),
                raw_hardware=str(metadata.get('compatible_hardware', [])),
                raw_warranty=metadata.get('_raw_warranty'),
                raw_restrictions=metadata.get('_raw_restrictions'),
            )
            product_results.append(product)

        return ProductSearchResponse(
            query=query,
            results=product_results,
            total_results=len(product_results)
        )

    except Exception as e:
        # Return empty response on error
        print(f"[SEARCH ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return ProductSearchResponse(
            query=query,
            results=[],
            total_results=0
        )


# =============================================================================
# Product Interest Tools (for Sales Agent)
# =============================================================================

@function_tool
def save_product_interest(
    context: RunContextWrapper[CustomerContext],
    product_name: str,
    product_details: str,
) -> str:
    """
    Save a product the customer is interested in.
    Call this whenever the customer shows interest in a specific product.

    Args:
        product_name: Name of the product (e.g., "Embarq Entry Door")
        product_details: Details like color, size, style, features
    """
    product_info = f"{product_name}: {product_details}"
    context.context.products_discussed.append(product_info)
    return f"Recorded interest in: {product_info}"


@function_tool
def finalize_selection(
    context: RunContextWrapper[CustomerContext],
    selected_product: str,
    conversation_summary: str,
) -> str:
    """
    Finalize the customer's product selection.
    Call this when the customer has decided on a specific product.

    Args:
        selected_product: The final product choice with all details
        conversation_summary: Brief summary of the conversation and customer needs
    """
    context.context.selected_product = selected_product
    context.context.conversation_summary = conversation_summary
    return f"Selection finalized: {selected_product}"


# =============================================================================
# Email Tool (for Sales Agent)
# =============================================================================

@function_tool
def send_lead_email(
    context: RunContextWrapper[CustomerContext],
) -> LeadEmailResponse:
    """
    Send a lead notification email to the sales team.
    Call this ONLY after:
    1. Customer info is complete (name, email, phone)
    2. Customer has finalized their product selection
    """
    ctx = context.context

    # Validate required info
    if not all([ctx.name, ctx.email, ctx.phone]):
        missing = []
        if not ctx.name:
            missing.append("name")
        if not ctx.email:
            missing.append("email")
        if not ctx.phone:
            missing.append("phone")
        return LeadEmailResponse(
            success=False,
            message=f"Missing customer information: {', '.join(missing)}",
            recipient=None
        )

    if not ctx.selected_product:
        return LeadEmailResponse(
            success=False,
            message="No product selection finalized",
            recipient=None
        )

    # Create email content
    subject = f"New Lead: {ctx.name} - {ctx.selected_product}"

    body = f"""
NEW LEAD FROM VOICE SALES AGENT
{'='*60}

CUSTOMER INFORMATION
{'-'*40}
Name:  {ctx.name}
Email: {ctx.email}
Phone: {ctx.phone}

PRODUCT INTEREST
{'-'*40}
Selected Product: {ctx.selected_product}

Products Discussed:
{chr(10).join(f"  - {p}" for p in ctx.products_discussed) if ctx.products_discussed else "  - None recorded"}

CONVERSATION SUMMARY
{'-'*40}
{ctx.conversation_summary}

{'='*60}
This lead was automatically generated by the {config.COMPANY_NAME} Voice Sales Agent.
Please follow up with the customer at your earliest convenience.
    """

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = config.SMTP_USERNAME
        msg["To"] = config.CLIENT_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            server.send_message(msg)

        return LeadEmailResponse(
            success=True,
            message="Lead email sent successfully!",
            recipient=config.CLIENT_EMAIL
        )

    except Exception as e:
        return LeadEmailResponse(
            success=False,
            message=f"Failed to send email: {str(e)}",
            recipient=None
        )


# =============================================================================
# Submit Lead Tool (NEW - Uses Queue for DB + Email)
# =============================================================================

@function_tool
async def submit_lead(
    context: RunContextWrapper[CustomerContext],
) -> str:
    """
    Submit the lead to save to database and send email notification to our sales team.

    This tool:
    1. Saves all customer details to the database (name, email, phone, product info, conversation summary)
    2. Sends an email notification to the sales team

    Call this ONLY after:
    1. Customer info is complete (name, email, phone)
    2. Customer has finalized their product selection (call finalize_selection first)

    IMPORTANT: After this tool returns successfully, you MUST immediately tell the customer:
    "Great news! I've sent your details to our sales team. They will contact you shortly
    to help you complete your purchase of [product name]. Is there anything else
    I can help you with today?"
    """
    ctx = context.context

    # Validate required info
    if not all([ctx.name, ctx.email, ctx.phone]):
        missing = []
        if not ctx.name:
            missing.append("name")
        if not ctx.email:
            missing.append("email")
        if not ctx.phone:
            missing.append("phone")
        return f"ERROR: Cannot submit lead. Missing customer information: {', '.join(missing)}. Please collect this information first."

    if not ctx.selected_product:
        return "ERROR: Cannot submit lead. No product selection finalized. Please call finalize_selection first to record the customer's product choice."

    # Submit to background queue
    result = await submit_lead_to_queue(
        name=ctx.name,
        email=ctx.email,
        phone=ctx.phone,
        selected_product=ctx.selected_product,
        products_discussed=ctx.products_discussed,
        conversation_summary=ctx.conversation_summary,
        session_id=None,  # Can be passed from context if available
    )

    if result["success"]:
        return f"""SUCCESS: Lead submitted successfully!

Customer Details Sent:
- Name: {ctx.name}
- Email: {ctx.email}
- Phone: {ctx.phone}
- Selected Product: {ctx.selected_product}

The sales team has been notified and will contact the customer shortly.

NOW YOU MUST: Tell the customer that their details have been sent to your team and someone will contact them soon to assist with their purchase. Ask if they have any other questions."""
    else:
        return f"ERROR: Failed to submit lead. {result.get('message', 'Unknown error')}"
