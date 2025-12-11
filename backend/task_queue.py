"""
Background Task Queue for Lead Processing

This module provides an async queue system that processes tasks sequentially:
1. Save lead to MongoDB
2. Send email notification

Tasks are processed one by one in the background while the agent
can immediately respond to the customer.
"""

import asyncio
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

from config import config
from database import MongoDB, LeadDocument, LeadRepository


# =============================================================================
# Task Types and Models
# =============================================================================

class TaskType(str, Enum):
    """Types of background tasks."""
    SAVE_LEAD_TO_DB = "save_lead_to_db"
    SEND_LEAD_EMAIL = "send_lead_email"
    PROCESS_LEAD = "process_lead"  # Combined: save to DB + send email


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a background task."""
    task_type: TaskType
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# =============================================================================
# Task Queue Manager
# =============================================================================

class TaskQueueManager:
    """
    Manages background tasks in an async queue.
    Tasks are processed sequentially (one by one) in the background.
    """

    def __init__(self):
        self._queue: asyncio.Queue[Task] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._processed_count: int = 0
        self._failed_count: int = 0

    async def start(self):
        """Start the background worker."""
        if self._is_running:
            return

        self._is_running = True
        self._worker_task = asyncio.create_task(self._worker())
        print("[TaskQueue] Background worker started")

    async def stop(self):
        """Stop the background worker."""
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        print("[TaskQueue] Background worker stopped")

    async def enqueue(self, task_type: TaskType, payload: Dict[str, Any]) -> Task:
        """
        Add a task to the queue.
        Returns immediately without waiting for the task to complete.
        """
        task = Task(task_type=task_type, payload=payload)
        await self._queue.put(task)
        print(f"[TaskQueue] Task enqueued: {task_type.value}")
        return task

    async def _worker(self):
        """Background worker that processes tasks one by one."""
        print("[TaskQueue] Worker starting...")

        while self._is_running:
            try:
                # Wait for a task (with timeout to check if we should stop)
                try:
                    task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process the task
                task.status = TaskStatus.PROCESSING
                print(f"[TaskQueue] Processing task: {task.task_type.value}")

                try:
                    result = await self._process_task(task)
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.utcnow()
                    self._processed_count += 1
                    print(f"[TaskQueue] Task completed: {task.task_type.value}")

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow()
                    self._failed_count += 1
                    print(f"[TaskQueue] Task failed: {task.task_type.value} - {e}")

                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[TaskQueue] Worker error: {e}")

        print("[TaskQueue] Worker stopped")

    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process a single task based on its type."""
        if task.task_type == TaskType.SAVE_LEAD_TO_DB:
            return await self._save_lead_to_db(task.payload)
        elif task.task_type == TaskType.SEND_LEAD_EMAIL:
            return await self._send_lead_email(task.payload)
        elif task.task_type == TaskType.PROCESS_LEAD:
            return await self._process_lead(task.payload)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    async def _process_lead(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a lead: save to DB first, then send email.
        This ensures the lead_id is available for the email.
        """
        # Step 1: Save to database
        print("[TaskQueue] Step 1: Saving lead to database...")
        db_result = await self._save_lead_to_db(payload)

        if not db_result.get("success"):
            return {
                "success": False,
                "message": f"Failed to save lead to database: {db_result.get('message')}",
                "db_saved": False,
                "email_sent": False,
            }

        lead_id = db_result.get("lead_id")
        print(f"[TaskQueue] Lead saved with ID: {lead_id}")

        # Step 2: Send email (include lead_id in payload)
        print("[TaskQueue] Step 2: Sending lead email...")
        email_payload = {**payload, "lead_id": lead_id}
        email_result = await self._send_lead_email(email_payload)

        return {
            "success": True,
            "message": "Lead processed successfully: saved to database and email sent",
            "lead_id": lead_id,
            "db_saved": True,
            "email_sent": email_result.get("success", False),
            "email_recipient": email_result.get("recipient"),
        }

    async def _save_lead_to_db(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Save lead to MongoDB."""
        lead = LeadDocument(
            name=payload["name"],
            email=payload["email"],
            phone=payload["phone"],
            selected_product=payload.get("selected_product"),
            products_discussed=payload.get("products_discussed", []),
            conversation_summary=payload.get("conversation_summary", ""),
            session_id=payload.get("session_id"),
        )

        lead_id = await LeadRepository.create_lead(lead)
        print(f"[TaskQueue] Lead saved to DB with ID: {lead_id}")

        return {
            "success": True,
            "lead_id": lead_id,
            "message": "Lead saved to database successfully"
        }

    async def _send_lead_email(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send lead notification email."""
        # Create email content
        subject = f"New Lead: {payload['name']} - {payload.get('selected_product', 'No product selected')}"

        body = f"""
NEW LEAD FROM VOICE SALES AGENT
{'='*60}

CUSTOMER INFORMATION
{'-'*40}
Name:  {payload['name']}
Email: {payload['email']}
Phone: {payload['phone']}

PRODUCT INTEREST
{'-'*40}
Selected Product: {payload.get('selected_product', 'Not specified')}

Products Discussed:
{chr(10).join(f"  - {p}" for p in payload.get('products_discussed', [])) or "  - None recorded"}

CONVERSATION SUMMARY
{'-'*40}
{payload.get('conversation_summary', 'No summary available')}

{'='*60}
This lead was automatically generated by the {config.COMPANY_NAME} Voice Sales Agent.
Please follow up with the customer at your earliest convenience.

Lead ID: {payload.get('lead_id', 'N/A')}
        """

        # Send email (runs in thread pool to not block async)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_email_sync, subject, body)

        # Update lead in database to mark email as sent
        lead_id = payload.get("lead_id")
        if lead_id:
            await LeadRepository.update_email_sent(lead_id)

        print(f"[TaskQueue] Lead email sent to: {config.CLIENT_EMAIL}")

        return {
            "success": True,
            "recipient": config.CLIENT_EMAIL,
            "message": "Lead email sent successfully"
        }

    def _send_email_sync(self, subject: str, body: str):
        """Synchronous email sending (called in executor)."""
        msg = MIMEMultipart()
        msg["From"] = config.SMTP_USERNAME
        msg["To"] = config.CLIENT_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            server.send_message(msg)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "is_running": self._is_running,
            "queue_size": self._queue.qsize(),
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
        }


# =============================================================================
# Global Task Queue Instance
# =============================================================================

task_queue = TaskQueueManager()


# =============================================================================
# Helper Function for Submitting Lead
# =============================================================================

async def submit_lead_to_queue(
    name: str,
    email: str,
    phone: str,
    selected_product: Optional[str] = None,
    products_discussed: Optional[list] = None,
    conversation_summary: str = "",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit a lead to the background queue for processing.

    This function:
    1. Enqueues a combined task that saves to DB first, then sends email

    The task is processed in the background.
    This function returns immediately without waiting.

    Returns:
        Dict with submission status
    """
    payload = {
        "name": name,
        "email": email,
        "phone": phone,
        "selected_product": selected_product,
        "products_discussed": products_discussed or [],
        "conversation_summary": conversation_summary,
        "session_id": session_id,
    }

    # Enqueue combined task (DB save + email send)
    await task_queue.enqueue(TaskType.PROCESS_LEAD, payload)

    return {
        "success": True,
        "message": "Lead submitted for processing. Your details have been sent to our team.",
        "tasks_queued": 1,
    }
