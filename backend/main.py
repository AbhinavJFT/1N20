"""
FastAPI Backend for Voice Sales Agent MVP
- WebSocket endpoint for real-time voice communication
- Session management with context persistence
- RealtimeAgent integration with proper audio streaming
"""

import asyncio
import json
import base64
import uuid
import struct
from datetime import datetime
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from agents.realtime import RealtimeRunner, RealtimeSession

from config import config
from models import CustomerContext, MessageType, WebSocketMessage, ConversationMessage
from agent_definitions import get_starting_agent
from database import MongoDB
from task_queue import task_queue

import os
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY


# =============================================================================
# Session Management
# =============================================================================

class SessionManager:
    """Manages active voice sessions with context and conversation history."""

    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def create_session(self, session_id: str) -> CustomerContext:
        """Create a new session with fresh context."""
        context = CustomerContext()
        self.sessions[session_id] = {
            "context": context,
            "created_at": datetime.now().isoformat(),
            "realtime_session": None,
            "conversation_history": [],  # Store conversation for context
            "current_agent": "GreetingAgent",
            "partial_transcript": "",  # For real-time typing display
        }
        return context

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get existing session data."""
        return self.sessions.get(session_id)

    def get_context(self, session_id: str) -> Optional[CustomerContext]:
        """Get context for a session."""
        session = self.sessions.get(session_id)
        return session["context"] if session else None

    def update_realtime_session(self, session_id: str, realtime_session: RealtimeSession):
        """Store the realtime session reference."""
        if session_id in self.sessions:
            self.sessions[session_id]["realtime_session"] = realtime_session

    def add_message(self, session_id: str, role: str, content: str, agent: str = None):
        """Add a message to conversation history."""
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_history"].append(
                ConversationMessage(
                    role=role,
                    content=content,
                    agent=agent,
                    timestamp=datetime.now().isoformat()
                )
            )

    def get_history(self, session_id: str) -> List[ConversationMessage]:
        """Get conversation history for a session."""
        session = self.sessions.get(session_id)
        return session["conversation_history"] if session else []

    def update_current_agent(self, session_id: str, agent_name: str):
        """Update the current active agent."""
        if session_id in self.sessions:
            self.sessions[session_id]["current_agent"] = agent_name

    def update_partial_transcript(self, session_id: str, text: str):
        """Update partial transcript for real-time display."""
        if session_id in self.sessions:
            self.sessions[session_id]["partial_transcript"] = text

    def end_session(self, session_id: str):
        """Clean up a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_all_sessions(self) -> Dict[str, dict]:
        """Get all active sessions (for debugging)."""
        return {
            sid: {
                "created_at": data["created_at"],
                "current_agent": data["current_agent"],
                "message_count": len(data["conversation_history"]),
                "context": {
                    "name": data["context"].name,
                    "email": data["context"].email,
                    "phone": data["context"].phone,
                    "info_complete": data["context"].info_collection_complete,
                }
            }
            for sid, data in self.sessions.items()
        }


# Global session manager
session_manager = SessionManager()


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print(f"Starting {config.COMPANY_NAME} Voice Sales Agent...")
    print(f"Server running on http://{config.HOST}:{config.PORT}")

    # Initialize MongoDB connection
    print("Connecting to MongoDB...")
    await MongoDB.connect()

    # Start background task queue worker
    print("Starting background task queue...")
    await task_queue.start()

    yield

    # Cleanup on shutdown
    print("Shutting down...")
    await task_queue.stop()
    await MongoDB.disconnect()


app = FastAPI(
    title=f"{config.COMPANY_NAME} Voice Sales Agent",
    description="Real-time voice sales agent for doors and windows",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check and info endpoint."""
    return {
        "status": "running",
        "service": f"{config.COMPANY_NAME} Voice Sales Agent",
        "version": "1.0.0",
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions (for debugging)."""
    return session_manager.get_all_sessions()


@app.post("/api/sessions")
async def create_session():
    """Create a new session and return session ID."""
    session_id = str(uuid.uuid4())
    session_manager.create_session(session_id)
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "context": {
            "name": session["context"].name,
            "email": session["context"].email,
            "phone": session["context"].phone,
            "products_discussed": session["context"].products_discussed,
            "selected_product": session["context"].selected_product,
            "info_complete": session["context"].info_collection_complete,
        }
    }


@app.get("/api/queue/stats")
async def get_queue_stats():
    """Get background task queue statistics."""
    return task_queue.stats


@app.get("/api/leads")
async def list_leads(limit: int = 100, skip: int = 0):
    """List all leads from the database."""
    from database import LeadRepository
    leads = await LeadRepository.get_all_leads(limit=limit, skip=skip)
    # Convert ObjectId to string for JSON serialization
    for lead in leads:
        lead["_id"] = str(lead["_id"])
    return {"leads": leads, "count": len(leads)}


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time voice communication.

    Message Protocol:
    - Client sends: {"type": "audio_input", "data": {"audio": "<base64>"}}
    - Client sends: {"type": "text_input", "data": {"text": "..."}}
    - Server sends: {"type": "audio_output", "data": {"audio": "<base64>"}}
    - Server sends: {"type": "transcript", "data": {"text": "...", "role": "assistant"}}
    """
    await websocket.accept()

    # Get or create session
    session = session_manager.get_session(session_id)
    if not session:
        context = session_manager.create_session(session_id)
    else:
        context = session["context"]

    # Send session started message
    await send_ws_message(websocket, MessageType.SESSION_STARTED, {
        "session_id": session_id,
        "message": f"Welcome to {config.COMPANY_NAME}! Connecting you to our assistant..."
    })

    try:
        # Create RealtimeRunner with the starting agent and configuration
        # Note: Voice is set per-session. Using greeting agent voice initially.
        # For different voices per agent, the session would need to be recreated on handoff.
        runner = RealtimeRunner(
            starting_agent=get_starting_agent(),
            config={
                "model_settings": {
                    "voice": config.GREETING_AGENT_VOICE,  # Use greeting agent voice
                    "modalities": ["audio"],  # Audio-only mode (text not supported with audio)
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "gpt-4o-transcribe",
                        "language": "en",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.95,  # Higher threshold to reduce background noise
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 800,  # Shorter silence to detect end of speech faster
                    },
                }
            }
        )

        # Start realtime session
        realtime_session = await runner.run(context=context)
        async with realtime_session:
            session_manager.update_realtime_session(session_id, realtime_session)

            # Create tasks for bidirectional communication
            receive_task = asyncio.create_task(
                handle_client_messages(websocket, realtime_session, session_id)
            )
            send_task = asyncio.create_task(
                handle_agent_events(websocket, realtime_session, session_id, context)
            )

            # Wait for either task to complete (client disconnect or error)
            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error for {session_id}: {e}")
        await send_ws_message(websocket, MessageType.ERROR, {"error": str(e)})
    finally:
        await send_ws_message(websocket, MessageType.SESSION_ENDED, {
            "session_id": session_id,
            "context": {
                "name": context.name,
                "email": context.email,
                "phone": context.phone,
            }
        })


async def handle_client_messages(
    websocket: WebSocket,
    realtime_session: RealtimeSession,
    session_id: str
):
    """Handle incoming messages from the client."""
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")
            msg_data = message.get("data", {})

            if msg_type == MessageType.AUDIO_INPUT.value:
                # Decode base64 audio and send to realtime session
                audio_base64 = msg_data.get("audio", "")
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    await realtime_session.send_audio(audio_bytes)

            elif msg_type == MessageType.TEXT_INPUT.value:
                # Send text message to realtime session
                text = msg_data.get("text", "")
                if text:
                    await realtime_session.send_message(text)
                    # Don't echo back - frontend already displays it

            elif msg_type == MessageType.END_SESSION.value:
                break

    except WebSocketDisconnect:
        pass  # Normal disconnect, don't re-raise
    except asyncio.CancelledError:
        pass  # Task was cancelled
    except Exception as e:
        print(f"Error handling client message: {e}")


async def handle_agent_events(
    websocket: WebSocket,
    realtime_session: RealtimeSession,
    session_id: str,
    context: CustomerContext
):
    """Handle events from the realtime agent with proper streaming."""
    try:
        current_agent_name = "GreetingAgent"
        partial_agent_transcript = ""
        partial_user_transcript = ""
        last_context_update = datetime.now()

        async for event in realtime_session:
            event_type = getattr(event, 'type', None)

            # Handle raw_model_event for transcripts and audio
            if event_type == 'raw_model_event':
                raw_data = getattr(event, 'data', None)
                if raw_data:
                    raw_type = getattr(raw_data, 'type', '')

                    # Debug: log important raw event types only
                    if raw_type and raw_type not in ('raw_server_event', 'audio', 'transcript_delta'):
                        print(f"  [RAW] {raw_type}")

                    # Check for conversation item events that might have user input
                    if raw_type == 'item_updated':
                        item = getattr(raw_data, 'item', None)
                        if item:
                            item_type = getattr(item, 'type', '')
                            role = getattr(item, 'role', '')
                            print(f"  [ITEM_UPDATED] type={item_type}, role={role}")
                            # Check if this is a user message with content
                            if role == 'user':
                                content = getattr(item, 'content', None)
                                if content:
                                    print(f"  [USER CONTENT] {content}")
                                # Also check for transcript
                                transcript = getattr(item, 'transcript', None)
                                if transcript:
                                    print(f"  [USER TRANSCRIPT from item] {transcript}")

                    # Handle transcript delta (agent speaking - real-time)
                    if raw_type == 'transcript_delta':
                        delta = getattr(raw_data, 'delta', '')
                        if delta:
                            partial_agent_transcript += delta
                            await send_ws_message(websocket, MessageType.PARTIAL_TRANSCRIPT, {
                                "text": partial_agent_transcript,
                                "role": "assistant",
                                "agent": current_agent_name
                            })

                    # Handle audio delta (the actual audio bytes)
                    elif raw_type == 'audio':
                        audio_delta = getattr(raw_data, 'delta', None)
                        if audio_delta:
                            await send_ws_message(websocket, MessageType.AUDIO_OUTPUT, {
                                "audio": audio_delta
                            })

                    # Handle audio done - finalize transcript
                    elif raw_type == 'audio_done':
                        if partial_agent_transcript:
                            await send_ws_message(websocket, MessageType.TRANSCRIPT, {
                                "text": partial_agent_transcript,
                                "role": "assistant",
                                "agent": current_agent_name
                            })
                            session_manager.add_message(session_id, "assistant", partial_agent_transcript, current_agent_name)
                            partial_agent_transcript = ""

                    # Handle turn ended - reset state
                    elif raw_type == 'turn_ended':
                        partial_agent_transcript = ""

                    # Handle input audio transcription (user speech)
                    elif 'input_audio_transcription' in raw_type:
                        print(f"  [TRANSCRIPTION] raw_type: {raw_type}")
                        print(f"  [TRANSCRIPTION] raw_data attrs: {[a for a in dir(raw_data) if not a.startswith('_')]}")
                        transcript = getattr(raw_data, 'transcript', '')
                        print(f"  [TRANSCRIPTION] transcript: '{transcript}'")

                        # Only send completed transcriptions (not deltas)
                        # The 'completed' event has the final accurate transcription
                        if 'completed' in raw_type and transcript:
                            await send_ws_message(websocket, MessageType.USER_TRANSCRIPT, {
                                "text": transcript,
                                "role": "user"
                            })
                            session_manager.add_message(session_id, "user", transcript)
                        elif 'delta' not in raw_type and transcript:
                            # Fallback for other transcription events
                            await send_ws_message(websocket, MessageType.USER_TRANSCRIPT, {
                                "text": transcript,
                                "role": "user"
                            })
                            session_manager.add_message(session_id, "user", transcript)

                continue

            # Skip other noisy events
            if event_type in ('history_updated', 'history_added'):
                continue

            # Debug logging for non-noisy events
            print(f"[{session_id[:8]}] Event: {event_type}")

            if event_type == "audio":
                # RealtimeAudio event - audio property contains RealtimeModelAudioEvent
                audio_event = getattr(event, 'audio', None)
                if audio_event:
                    # Try multiple ways to get audio data
                    audio_data = None

                    # Try delta (base64 string)
                    audio_delta = getattr(audio_event, 'delta', None)
                    if audio_delta and isinstance(audio_delta, str):
                        audio_data = audio_delta

                    # Try data (bytes)
                    if not audio_data:
                        raw_data = getattr(audio_event, 'data', None)
                        if raw_data and isinstance(raw_data, bytes):
                            audio_data = base64.b64encode(raw_data).decode('utf-8')

                    # Try audio attribute directly
                    if not audio_data:
                        raw_audio = getattr(audio_event, 'audio', None)
                        if raw_audio and isinstance(raw_audio, bytes):
                            audio_data = base64.b64encode(raw_audio).decode('utf-8')

                    if audio_data:
                        await send_ws_message(websocket, MessageType.AUDIO_OUTPUT, {
                            "audio": audio_data
                        })
                    else:
                        # Debug: print what attributes we have
                        print(f"  [DEBUG] audio_event type: {type(audio_event)}")
                        print(f"  [DEBUG] audio_event attrs: {[a for a in dir(audio_event) if not a.startswith('_')]}")

            elif event_type == "audio_end":
                # Agent finished speaking
                await send_ws_message(websocket, MessageType.AGENT_DONE, {
                    "agent": current_agent_name
                })
                partial_agent_transcript = ""

            elif event_type == "audio_interrupted":
                # User interrupted the agent
                await send_ws_message(websocket, MessageType.AGENT_DONE, {
                    "agent": current_agent_name,
                    "interrupted": True
                })
                partial_agent_transcript = ""

            elif event_type == "agent_start":
                # Agent started processing
                agent = getattr(event, 'agent', None)
                if agent:
                    current_agent_name = getattr(agent, 'name', current_agent_name)
                    session_manager.update_current_agent(session_id, current_agent_name)
                await send_ws_message(websocket, MessageType.AGENT_SPEAKING, {
                    "agent": current_agent_name
                })

            elif event_type == "agent_end":
                # Agent finished
                pass

            elif event_type == "tool_start":
                # Tool execution started
                tool = getattr(event, 'tool', None)
                tool_name = getattr(tool, 'name', 'unknown') if tool else 'unknown'
                await send_ws_message(websocket, MessageType.TOOL_CALL, {
                    "tool": tool_name,
                    "status": "started"
                })

            elif event_type == "tool_end":
                # Tool execution completed
                tool = getattr(event, 'tool', None)
                tool_name = getattr(tool, 'name', 'unknown') if tool else 'unknown'
                tool_output = getattr(event, 'output', None)
                await send_ws_message(websocket, MessageType.TOOL_RESULT, {
                    "tool": tool_name,
                    "status": "completed",
                    "result": str(tool_output) if tool_output else None
                })

            elif event_type == "handoff":
                # Agent handoff occurred
                old_agent = current_agent_name
                to_agent = getattr(event, 'to_agent', None)
                new_agent_name = getattr(to_agent, 'name', 'Unknown') if to_agent else 'Unknown'
                current_agent_name = new_agent_name
                session_manager.update_current_agent(session_id, current_agent_name)

                await send_ws_message(websocket, MessageType.HANDOFF, {
                    "from_agent": old_agent,
                    "to_agent": new_agent_name,
                    "message": f"Transferring you to {new_agent_name}..."
                })

            elif event_type == "guardrail_tripped":
                # Guardrail triggered
                message = getattr(event, 'message', "I can only help with questions about doors and windows.")
                await send_ws_message(websocket, MessageType.ERROR, {
                    "type": "guardrail",
                    "message": message
                })

            elif event_type == "error":
                # Error occurred
                error = getattr(event, 'error', None)
                error_msg = str(error) if error else "Unknown error"
                await send_ws_message(websocket, MessageType.ERROR, {
                    "error": error_msg
                })

            # Send context update periodically (throttled)
            now = datetime.now()
            if (now - last_context_update).total_seconds() > 1.0:
                await send_ws_message(websocket, MessageType.CONTEXT_UPDATE, {
                    "name": context.name,
                    "email": context.email,
                    "phone": context.phone,
                    "info_complete": context.info_collection_complete,
                    "products_discussed": context.products_discussed,
                    "selected_product": context.selected_product,
                    "current_agent": current_agent_name,
                })
                last_context_update = now

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Error handling agent event: {e}")
        import traceback
        traceback.print_exc()


async def send_ws_message(websocket: WebSocket, msg_type: MessageType, data: dict):
    """Send a formatted WebSocket message."""
    try:
        message = WebSocketMessage(
            type=msg_type,
            data=data,
            timestamp=datetime.now().isoformat()
        )
        await websocket.send_text(message.model_dump_json())
    except WebSocketDisconnect:
        pass  # Client disconnected
    except RuntimeError as e:
        if "websocket.close" in str(e) or "already completed" in str(e):
            pass  # WebSocket already closed
        else:
            print(f"Error sending WebSocket message: {e}")
    except Exception as e:
        print(f"Error sending WebSocket message: {e}")


# =============================================================================
# Serve Frontend
# =============================================================================

# Get the directory containing this file
import pathlib
BACKEND_DIR = pathlib.Path(__file__).parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"


@app.get("/app")
async def serve_frontend():
    """Serve the frontend HTML."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
async def serve_styles():
    """Serve the CSS file."""
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    """Serve the JavaScript file."""
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  {config.COMPANY_NAME} Voice Sales Agent")
    print(f"{'='*60}")
    print(f"\nBackend API: http://localhost:{config.PORT}")
    print(f"Frontend:    http://localhost:{config.PORT}/app")
    print(f"\nPress Ctrl+C to stop\n")

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
    )
