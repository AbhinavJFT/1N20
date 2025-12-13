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
    ProductImage,
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
    Search the ProVia product catalog using semantic search.

    IMPORTANT: The Sales Agent should reframe customer questions into optimized
    queries before calling this tool. Use terms that match the embedded content.

    Args:
        query: An OPTIMIZED search query (reframed by the agent).
               Should include relevant terms like:
               - Product types: "embarq", "signet", "french door", "storm door", "pet door"
               - Features: "energy efficient", "quad glass", "r-10", "fiberglass", "steel"
               - Styles: "modern", "traditional", "craftsman", "rustic"
               - Materials: "mahogany", "cherry", "oak", "knotty alder"

    Returns:
        ProductSearchResponse with matching products containing ONLY metadata:
        - Product identification (name, series, tier, category)
        - Key features and specifications
        - Customization options (skins, styles, glass, frames)
        - Detailed info (finishes, hardware, warranty, restrictions)
        - Image URLs and product links

        NOTE: Description field is NOT included - all info is in structured metadata.
    """
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query)

        # Query Pinecone (index: 1n20, namespace: 1n20)
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="1n20"
        )

        # Format results using ONLY metadata (not description)
        product_results = []

        # Known metadata keys that we explicitly handle
        KNOWN_METADATA_KEYS = {
            # Identity
            'product_id', 'name', 'series', 'category', 'top_category', 'subcategory', 'tier', 'chunk_id',
            # Features
            'key_features', 'energy_star', 'u_factor', 'energy_certification',
            # Customization
            'skin_options', 'door_style_codes', 'styles_available', 'glass_package_names',
            'compatible_decorative_glass', 'compatible_frames', 'sidelites',
            # Accessories
            'brands',
            # Search
            'search_tags',
            # Raw JSON fields
            '_raw_compatible_finishes', '_raw_hardware', '_raw_warranty', '_raw_restrictions',
            '_raw_installation', '_raw_image_urls', '_raw_door_styles', '_raw_glass_packages',
            '_raw_astragal_details', '_raw_styles_configurations', '_raw_available_series',
            '_raw_eight_foot_availability',
            # Other known fields
            'compatible_hardware', 'restrictions', 'restrictions_and_notes',
            'image_primary', 'product_url', 'source_pages',
            # Description is known but intentionally not used
            'description',
        }

        # Important fields that should be flagged if missing (contextually important)
        IMPORTANT_FIELDS_MAP = {
            'Entry Door': ['key_features', 'skin_options', 'glass_package_names', 'compatible_frames', 'energy_star'],
            'Storm Doors': ['key_features', 'energy_star'],
            'Accessories': ['brands', 'restrictions'],
            'Finishes': ['key_features'],
            'Frame Options': ['key_features'],
            'Glass Options': ['key_features'],
            'Hardware': ['key_features'],
        }

        for match in results.matches:
            metadata = match.metadata if hasattr(match, 'metadata') and match.metadata else {}

            # Helper to extract list fields cleanly
            def get_list(key: str) -> list:
                val = metadata.get(key, [])
                if isinstance(val, list):
                    # Filter out complex JSON strings, keep clean values
                    return [item for item in val if isinstance(item, str) and not item.startswith('{')]
                return []

            # Helper to get raw JSON field as dict
            def get_raw_json(key: str) -> Optional[dict]:
                val = metadata.get(key)
                if val and isinstance(val, str):
                    try:
                        import json
                        return json.loads(val)
                    except:
                        return None
                return val if isinstance(val, dict) else None

            # Extract product name - use 'name' field if available, else format product_id
            product_name = metadata.get('name')
            if not product_name:
                product_id = metadata.get('product_id', match.id)
                product_name = product_id.replace('_', ' ').title()

            # Get category for context-aware field checking
            category = metadata.get('category', metadata.get('top_category', 'Product'))

            # Extract all standard fields
            key_features = get_list('key_features')
            skin_options = get_list('skin_options')
            door_styles = get_list('door_style_codes') or get_list('styles_available')
            glass_packages = get_list('glass_package_names')
            decorative_glass = get_list('compatible_decorative_glass')
            compatible_frames = get_list('compatible_frames')
            sidelites = get_list('sidelites')
            brands = get_list('brands')
            search_tags = get_list('search_tags')
            finishes = get_raw_json('_raw_compatible_finishes')
            hardware = get_raw_json('_raw_hardware') or get_list('compatible_hardware')
            warranty = get_raw_json('_raw_warranty')
            restrictions = get_raw_json('_raw_restrictions') or get_list('restrictions') or get_list('restrictions_and_notes')
            installation = get_raw_json('_raw_installation')

            # === BUILD CONSOLIDATED IMAGES LIST ===
            images: List[ProductImage] = []

            # 1. Parse _raw_image_urls (main source - JSON with descriptive keys)
            raw_image_urls = get_raw_json('_raw_image_urls')
            if raw_image_urls and isinstance(raw_image_urls, dict):
                for key, urls in raw_image_urls.items():
                    # Skip non-image entries that might be in the JSON
                    if not isinstance(urls, list):
                        continue
                    # Extract category and description from the key
                    # Keys are like "primary - main image of the door" or just "primary"
                    if ' - ' in key:
                        parts = key.split(' - ', 1)
                        img_category = parts[0].strip()
                        img_description = parts[1].strip()
                    else:
                        img_category = key.strip()
                        img_description = key.replace('_', ' ').strip()

                    for url in urls:
                        if isinstance(url, str) and (url.endswith('.jpg') or url.endswith('.png') or url.endswith('.jpeg')):
                            images.append(ProductImage(
                                url=url,
                                description=img_description,
                                category=img_category
                            ))

            # 2. Parse flattened image fields (image_primary, image_detail_images, etc.)
            for key, value in metadata.items():
                if key.startswith('image_') and isinstance(value, list):
                    img_category = key.replace('image_', '').replace('_', ' ')
                    for url in value:
                        if isinstance(url, str) and (url.endswith('.jpg') or url.endswith('.png') or url.endswith('.jpeg')):
                            # Check if this URL is already added from _raw_image_urls
                            if not any(img.url == url for img in images):
                                images.append(ProductImage(
                                    url=url,
                                    description=f"{img_category} image",
                                    category=img_category
                                ))

            # 3. Parse standalone descriptive keys with images (e.g., in Finishes)
            for key, value in metadata.items():
                if key.startswith('_raw_') or key.startswith('image_'):
                    continue
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and (item.endswith('.jpg') or item.endswith('.png') or item.endswith('.jpeg')):
                            # Check if this URL is already added
                            if not any(img.url == item for img in images):
                                # Use the key as description
                                if ' - ' in key:
                                    parts = key.split(' - ', 1)
                                    img_description = parts[1].strip()
                                    img_category = parts[0].strip()
                                else:
                                    img_description = key.replace('_', ' ')
                                    img_category = "other"
                                images.append(ProductImage(
                                    url=item,
                                    description=img_description,
                                    category=img_category
                                ))

            # === CASE 1: Capture extra metadata fields not in our model ===
            extra_metadata = {}
            for key, value in metadata.items():
                if key not in KNOWN_METADATA_KEYS and value:
                    # Skip empty values
                    if isinstance(value, (list, dict)) and not value:
                        continue
                    if isinstance(value, str) and not value.strip():
                        continue
                    extra_metadata[key] = value

            # === CASE 2: Track important fields that are missing ===
            unavailable = []
            important_fields = IMPORTANT_FIELDS_MAP.get(category, [])

            field_checks = {
                'key_features': key_features,
                'skin_options': skin_options,
                'glass_package_names': glass_packages,
                'compatible_frames': compatible_frames,
                'energy_star': metadata.get('energy_star'),
                'brands': brands,
                'restrictions': restrictions,
            }

            for field_name in important_fields:
                if field_name in field_checks:
                    value = field_checks[field_name]
                    # Check if field is empty/missing
                    if value is None or value == [] or value == {}:
                        unavailable.append(field_name)

            # Build comprehensive product result from metadata ONLY
            product = ProductSearchResult(
                product_id=match.id,
                name=product_name,
                series=metadata.get('series'),
                category=category,
                subcategory=metadata.get('subcategory'),
                tier=metadata.get('tier'),

                # Key features - primary selling points
                key_features=key_features,

                # Energy specifications
                energy_star=metadata.get('energy_star', False),
                u_factor=metadata.get('u_factor'),
                energy_certification=metadata.get('energy_certification'),

                # Customization options
                skin_options=skin_options,
                door_styles=door_styles,
                glass_packages=glass_packages,
                decorative_glass=decorative_glass,
                compatible_frames=compatible_frames,
                sidelites=sidelites,

                # For accessories
                brands=brands,

                # Search metadata
                search_tags=search_tags,

                # Detailed JSON data
                finishes=finishes,
                hardware=hardware,
                warranty=warranty,
                restrictions=restrictions,
                installation=installation,

                # Images with descriptions - agent selects relevant ones
                images=images,

                # Links and references
                product_url=metadata.get('product_url'),
                source_pages=get_list('source_pages'),

                # Relevance score
                relevance_score=round(match.score, 3) if hasattr(match, 'score') else 0.0,

                # Extra metadata not in our standard model
                additional_info=extra_metadata if extra_metadata else None,

                # Fields that were expected but not available
                unavailable_fields=unavailable,
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
