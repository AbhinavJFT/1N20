"""
FastAPI Backend for Voice Sales Agent MVP
- WebSocket endpoint for real-time voice communication
- Session management with context persistence
- VoicePipeline integration (STT → LLM → TTS) for cost-effective voice

Architecture: VoicePipeline (3-step process)
1. STT: gpt-4o-mini-transcribe converts user speech to text
2. LLM: gpt-4o-mini processes the text and generates response
3. TTS: gpt-4o-mini-tts converts response to speech
"""

import asyncio
import json
import base64
import uuid
from datetime import datetime
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from agents import Agent, Runner
from agents.voice import (
    VoicePipeline,
    VoiceWorkflowBase,
    VoiceWorkflowHelper,
    VoicePipelineConfig,
    TTSModelSettings,
    StreamedAudioInput,
)
from typing import AsyncIterator, Any

from config import config
from models import CustomerContext, MessageType, WebSocketMessage, ConversationMessage
from agent_definitions import get_starting_agent, get_sales_agent, AGENT_VOICES
from database import MongoDB
from task_queue import task_queue

import os
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# Audio configuration (must match frontend)
CONFIG_AUDIO_SAMPLE_RATE = 24000  # 24kHz sample rate
CONFIG_AUDIO_CHANNELS = 1  # Mono


# =============================================================================
# Custom Voice Workflow with Context Support
# =============================================================================

class ContextAwareVoiceWorkflow(VoiceWorkflowBase):
    """
    Custom voice workflow that passes CustomerContext to the agent.

    Unlike SingleAgentVoiceWorkflow, this workflow passes the context parameter
    to Runner.run_streamed(), enabling tools to access CustomerContext.
    """

    def __init__(
        self,
        agent: Agent[Any],
        context: CustomerContext,
        input_history: list = None,
        on_transcription: callable = None,  # Callback when transcription is ready
        on_tool_call: callable = None,  # Callback when tool is called
        on_tool_result: callable = None,  # Callback when tool returns
    ):
        self._current_agent = agent
        self._context = context
        self._input_history: list = input_history or []
        self._last_agent = agent  # Track the last agent for handoff detection
        self._last_transcription = ""  # Store the last user transcription
        self._on_transcription = on_transcription
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result

    async def run(self, transcription: str) -> AsyncIterator[str]:
        """Run the workflow with context passed to the agent."""
        # Store the transcription
        self._last_transcription = transcription

        # Call transcription callback immediately so UI can show user message
        if self._on_transcription:
            await self._on_transcription(transcription)

        # Add the transcription to the input history
        self._input_history.append({
            "role": "user",
            "content": transcription,
        })

        # Run the agent with context - THIS IS THE KEY DIFFERENCE
        result = Runner.run_streamed(
            self._current_agent,
            self._input_history,
            context=self._context,  # Pass CustomerContext here!
        )

        # Track tool calls for callbacks
        tool_names_by_id: Dict[str, str] = {}

        # Process events and stream text
        async for event in result.stream_events():
            event_type = type(event).__name__

            # Handle tool call items
            if event_type == "RunItemStreamEvent":
                item = event.item
                item_type = getattr(item, 'type', '')

                if item_type == "tool_call_item" and self._on_tool_call:
                    raw_item = getattr(item, 'raw_item', None)
                    if raw_item:
                        tool_name = getattr(raw_item, 'name', 'tool')
                        call_id = getattr(raw_item, 'call_id', '')
                        tool_names_by_id[call_id] = tool_name
                        await self._on_tool_call(tool_name, call_id)

                elif item_type == "tool_call_output_item" and self._on_tool_result:
                    raw_item = getattr(item, 'raw_item', None)
                    if raw_item:
                        call_id = getattr(raw_item, 'call_id', '')
                        output = getattr(raw_item, 'output', None)
                        tool_name = tool_names_by_id.get(call_id, 'tool')
                        await self._on_tool_result(tool_name, call_id, str(output) if output else '')

                elif item_type == "message_output_item":
                    # This contains text output
                    raw_item = getattr(item, 'raw_item', None)
                    if raw_item:
                        content = getattr(raw_item, 'content', [])
                        for part in content:
                            if hasattr(part, 'text'):
                                yield part.text

            # Handle raw response streaming for partial text
            elif event_type == "RawResponsesStreamEvent":
                data = getattr(event, 'data', None)
                if data:
                    # Extract text delta if available
                    delta = getattr(data, 'delta', None)
                    if delta:
                        text = getattr(delta, 'text', None)
                        if text:
                            yield text

        # Update the input history and track the last agent
        self._input_history = result.to_input_list()
        self._last_agent = result.last_agent

        # Update current agent if handoff occurred
        if result.last_agent and result.last_agent != self._current_agent:
            self._current_agent = result.last_agent

    @property
    def last_agent(self) -> Agent:
        """Get the last agent that ran (for handoff detection)."""
        return self._last_agent

    @property
    def current_agent(self) -> Agent:
        """Get the current active agent."""
        return self._current_agent

    @property
    def input_history(self) -> list:
        """Get the conversation history."""
        return self._input_history

    @property
    def last_transcription(self) -> str:
        """Get the last user transcription."""
        return self._last_transcription


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
            "conversation_history": [],  # Store conversation for context
            "current_agent": "GreetingAgent",
            "current_voice": AGENT_VOICES.get("GreetingAgent", config.VOICE),
            "voice_pipeline": None,  # VoicePipeline instance
            "streamed_input": None,  # StreamedAudioInput for voice mode
            "is_voice_mode": False,  # Track if currently in voice mode
            "voice_input_history": [],  # Conversation history for VoicePipeline workflow
        }
        return context

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get existing session data."""
        return self.sessions.get(session_id)

    def get_context(self, session_id: str) -> Optional[CustomerContext]:
        """Get context for a session."""
        session = self.sessions.get(session_id)
        return session["context"] if session else None

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

    def get_history_as_messages(self, session_id: str) -> List[dict]:
        """Get conversation history formatted for agent input."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return [
            {"role": msg.role, "content": msg.content}
            for msg in session["conversation_history"]
        ]

    def update_current_agent(self, session_id: str, agent_name: str):
        """Update the current active agent and voice."""
        if session_id in self.sessions:
            self.sessions[session_id]["current_agent"] = agent_name
            self.sessions[session_id]["current_voice"] = AGENT_VOICES.get(agent_name, config.VOICE)

    def end_session(self, session_id: str):
        """Clean up a session."""
        if session_id in self.sessions:
            # Close any open streamed input
            session = self.sessions[session_id]
            if session.get("streamed_input"):
                try:
                    session["streamed_input"].close()
                except:
                    pass
            del self.sessions[session_id]

    def get_all_sessions(self) -> Dict[str, dict]:
        """Get all active sessions (for debugging)."""
        return {
            sid: {
                "created_at": data["created_at"],
                "current_agent": data["current_agent"],
                "message_count": len(data["conversation_history"]),
                "is_voice_mode": data.get("is_voice_mode", False),
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
# VoicePipeline Factory
# =============================================================================

def create_voice_pipeline(
    agent: Agent,
    voice: str,
    context: CustomerContext,
    input_history: list = None,
    on_transcription: callable = None,
    on_tool_call: callable = None,
    on_tool_result: callable = None,
) -> tuple[VoicePipeline, ContextAwareVoiceWorkflow]:
    """
    Create a VoicePipeline with context-aware workflow.

    Returns both the pipeline and workflow so we can access workflow properties
    (like last_agent for handoff detection) after the pipeline runs.

    Callbacks:
        on_transcription: Called immediately when user speech is transcribed
        on_tool_call: Called when a tool is invoked
        on_tool_result: Called when a tool returns
    """
    # Create our custom workflow that passes context to the agent
    workflow = ContextAwareVoiceWorkflow(
        agent=agent,
        context=context,
        input_history=input_history,
        on_transcription=on_transcription,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    )

    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=config.STT_MODEL,
        tts_model=config.TTS_MODEL,
        config=VoicePipelineConfig(
            tts_settings=TTSModelSettings(
                voice=voice,
                instructions="Speak clearly and professionally. Be friendly and helpful."
            ),
            tracing_disabled=True,  # Disable tracing for production
        )
    )

    return pipeline, workflow


def get_agent_by_name(name: str) -> Agent:
    """Get agent instance by name."""
    if name == "SalesAgent":
        return get_sales_agent()
    return get_starting_agent()


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for voice and text communication using VoicePipeline.

    Message Protocol:
    - Client sends: {"type": "audio_input", "data": {"audio": "<base64>"}}
    - Client sends: {"type": "text_input", "data": {"text": "..."}}
    - Server sends: {"type": "audio_output", "data": {"audio": "<base64>"}}
    - Server sends: {"type": "transcript", "data": {"text": "...", "role": "assistant"}}
    """
    await websocket.accept()

    # Log connection start
    short_id = session_id[:12]
    print(f"\n{'─'*60}")
    print(f"🔗 CONNECTION OPENED | Session: {short_id}")
    print(f"{'─'*60}")

    # Get or create session
    session = session_manager.get_session(session_id)
    if not session:
        context = session_manager.create_session(session_id)
        print(f"📝 NEW SESSION created: {short_id}")
    else:
        context = session["context"]
        print(f"📝 EXISTING SESSION resumed: {short_id}")

    session = session_manager.get_session(session_id)

    # Send session started message
    await send_ws_message(websocket, MessageType.SESSION_STARTED, {
        "session_id": session_id,
        "agent": session["current_agent"],
        "message": f"Welcome to {config.COMPANY_NAME}! Connecting you to our assistant..."
    })

    try:
        # Main message handling loop
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")
                msg_data = message.get("data", {})

                if msg_type == MessageType.TEXT_INPUT.value:
                    # Handle text input - use Runner directly (no TTS)
                    text = msg_data.get("text", "")
                    if text:
                        await handle_text_input(websocket, session_id, text, context)

                elif msg_type == MessageType.AUDIO_INPUT.value:
                    # Handle audio input - use VoicePipeline
                    audio_base64 = msg_data.get("audio", "")
                    if audio_base64:
                        await handle_audio_input(websocket, session_id, audio_base64, context)

                elif msg_type == "voice_mode_start":
                    # Client starting voice mode - initialize pipeline
                    session["is_voice_mode"] = True
                    print(f"🎤 VOICE MODE START | {short_id}")

                elif msg_type == "voice_mode_end":
                    # Client ending voice mode - process accumulated audio
                    session["is_voice_mode"] = False
                    print(f"🎤 VOICE MODE END | {short_id}")
                    await handle_voice_turn_end(websocket, session_id, context)

                elif msg_type == "interrupt":
                    # User interrupt - cancel current processing
                    print(f"⏹️  USER INTERRUPT | {short_id}")
                    await send_ws_message(websocket, MessageType.AGENT_DONE, {
                        "agent": session["current_agent"],
                        "interrupted": True
                    })

                elif msg_type == MessageType.END_SESSION.value:
                    break

            except WebSocketDisconnect:
                raise
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON received | {short_id}")
            except Exception as e:
                print(f"❌ Message handling error | {short_id} | {e}")
                import traceback
                traceback.print_exc()

    except WebSocketDisconnect:
        print(f"\n{'─'*60}")
        print(f"🔌 CONNECTION CLOSED | Session: {short_id} (client disconnected)")
        print(f"{'─'*60}\n")
    except Exception as e:
        print(f"\n❌ ERROR | Session: {short_id} | {e}")
        import traceback
        traceback.print_exc()
        await send_ws_message(websocket, MessageType.ERROR, {"error": str(e)})
    finally:
        print(f"\n📊 SESSION SUMMARY | {short_id}")
        print(f"   Name: {context.name or 'Not provided'}")
        print(f"   Email: {context.email or 'Not provided'}")
        print(f"   Phone: {context.phone or 'Not provided'}")
        print(f"{'─'*60}\n")
        await send_ws_message(websocket, MessageType.SESSION_ENDED, {
            "session_id": session_id,
            "context": {
                "name": context.name,
                "email": context.email,
                "phone": context.phone,
            }
        })


# =============================================================================
# Text Input Handler (uses Runner directly - no TTS)
# =============================================================================

async def handle_text_input(
    websocket: WebSocket,
    session_id: str,
    text: str,
    context: CustomerContext
):
    """Handle text input using Runner directly (cheaper, no audio)."""
    session = session_manager.get_session(session_id)
    if not session:
        return

    short_id = session_id[:12]
    current_agent_name = session["current_agent"]
    current_agent = get_agent_by_name(current_agent_name)

    print(f"📝 TEXT INPUT | {short_id} | \"{text}\"")

    # Add user message to history
    session_manager.add_message(session_id, "user", text)

    # Send user transcript to frontend
    await send_ws_message(websocket, MessageType.USER_TRANSCRIPT, {
        "text": text,
        "role": "user"
    })

    # Notify frontend that agent is processing
    await send_ws_message(websocket, MessageType.AGENT_SPEAKING, {
        "agent": current_agent_name
    })

    try:
        # Get conversation history for context
        history = session_manager.get_history_as_messages(session_id)

        # Build input with conversation history
        # The agent needs the full conversation to maintain context
        if history:
            # Use history as input (includes the current message we just added)
            agent_input = history
        else:
            # First message - just use the text
            agent_input = text

        # Run the agent with streaming to capture tool events
        result = Runner.run_streamed(
            starting_agent=current_agent,
            input=agent_input,
            context=context,
        )

        # Track response text and tool calls
        response_text = ""
        tool_calls_in_progress = set()

        # Track tool names by call_id for results
        tool_names_by_id: Dict[str, str] = {}

        # Process streaming events
        async for event in result.stream_events():
            event_type = type(event).__name__

            # Handle tool call items
            if event_type == "RunItemStreamEvent":
                item = event.item
                item_type = getattr(item, 'type', '')

                if item_type == "tool_call_item":
                    # ToolCallItem has raw_item which is a ResponseFunctionToolCall
                    raw_item = getattr(item, 'raw_item', None)
                    if raw_item:
                        tool_name = getattr(raw_item, 'name', 'tool')
                        call_id = getattr(raw_item, 'call_id', '')
                    else:
                        # Fallback - try to get from item directly
                        tool_name = 'tool'
                        call_id = str(id(item))  # Use object id as fallback

                    if call_id and call_id not in tool_calls_in_progress:
                        tool_calls_in_progress.add(call_id)
                        tool_names_by_id[call_id] = tool_name
                        print(f"🔧 TOOL CALL | {short_id} | {tool_name}")
                        await send_ws_message(websocket, MessageType.TOOL_CALL, {
                            "tool": tool_name,
                            "call_id": call_id,
                        })

                elif item_type == "tool_call_output_item":
                    # ToolCallOutputItem has raw_item which is a FunctionCallOutput
                    raw_item = getattr(item, 'raw_item', None)
                    if raw_item:
                        call_id = getattr(raw_item, 'call_id', '')
                        output = getattr(raw_item, 'output', None)
                    else:
                        call_id = ''
                        output = None

                    # Convert output to string safely (it could be an object like CustomerInfoStatus)
                    output_str = str(output) if output is not None else 'done'
                    tool_name = tool_names_by_id.get(call_id, 'tool')

                    print(f"✅ TOOL RESULT | {short_id} | {tool_name} → {output_str[:50]}...")
                    await send_ws_message(websocket, MessageType.TOOL_RESULT, {
                        "tool": tool_name,
                        "call_id": call_id,
                        "result": output_str[:200],
                        "status": "completed",
                    })
                    if call_id in tool_calls_in_progress:
                        tool_calls_in_progress.discard(call_id)

            # Handle text output
            elif event_type == "RawResponsesStreamEvent":
                # This contains the raw response chunks
                pass

        # Get the final response text
        response_text = result.final_output if hasattr(result, 'final_output') else ""

        if response_text:
            print(f"🤖 AGENT SAID | {short_id} | \"{response_text[:100]}{'...' if len(response_text) > 100 else ''}\"")

            # Add assistant message to history
            session_manager.add_message(session_id, "assistant", response_text, current_agent_name)

            # Send transcript to frontend
            await send_ws_message(websocket, MessageType.TRANSCRIPT, {
                "text": response_text,
                "role": "assistant",
                "agent": current_agent_name
            })

        # Check for handoff
        if hasattr(result, 'last_agent') and result.last_agent:
            new_agent_name = result.last_agent.name
            if new_agent_name != current_agent_name:
                print(f"🔀 HANDOFF | {short_id} | {current_agent_name} → {new_agent_name}")
                session_manager.update_current_agent(session_id, new_agent_name)
                await send_ws_message(websocket, MessageType.HANDOFF, {
                    "from_agent": current_agent_name,
                    "to_agent": new_agent_name,
                    "message": f"Transferring you to {new_agent_name}..."
                })

        # Send context update
        await send_ws_message(websocket, MessageType.CONTEXT_UPDATE, {
            "name": context.name,
            "email": context.email,
            "phone": context.phone,
            "info_complete": context.info_collection_complete,
            "products_discussed": context.products_discussed,
            "selected_product": context.selected_product,
            "current_agent": session["current_agent"],
        })

    except Exception as e:
        print(f"❌ TEXT HANDLER ERROR | {short_id} | {e}")
        import traceback
        traceback.print_exc()
        await send_ws_message(websocket, MessageType.ERROR, {
            "error": str(e)
        })

    finally:
        # Notify frontend that agent is done
        await send_ws_message(websocket, MessageType.AGENT_DONE, {
            "agent": session["current_agent"]
        })


# =============================================================================
# Audio Input Handler (uses VoicePipeline - STT → LLM → TTS)
# =============================================================================

# Buffer for accumulating audio chunks during voice mode
audio_buffers: Dict[str, bytearray] = {}


async def handle_audio_input(
    websocket: WebSocket,
    session_id: str,
    audio_base64: str,
    context: CustomerContext
):
    """Handle audio input - accumulate chunks for VoicePipeline processing."""
    session = session_manager.get_session(session_id)
    if not session:
        return

    # Decode audio and add to buffer
    audio_bytes = base64.b64decode(audio_base64)

    if session_id not in audio_buffers:
        audio_buffers[session_id] = bytearray()

    audio_buffers[session_id].extend(audio_bytes)


async def handle_voice_turn_end(
    websocket: WebSocket,
    session_id: str,
    context: CustomerContext
):
    """Process accumulated audio when voice turn ends."""
    session = session_manager.get_session(session_id)
    if not session:
        return

    short_id = session_id[:12]

    # Get accumulated audio
    if session_id not in audio_buffers or len(audio_buffers[session_id]) == 0:
        print(f"⚠️  NO AUDIO | {short_id} | No audio data to process")
        return

    audio_data = bytes(audio_buffers[session_id])
    audio_buffers[session_id] = bytearray()  # Clear buffer

    current_agent_name = session["current_agent"]
    current_agent = get_agent_by_name(current_agent_name)
    current_voice = session["current_voice"]

    print(f"🎤 VOICE INPUT | {short_id} | Processing {len(audio_data)} bytes of audio")

    # Notify frontend that agent is processing
    await send_ws_message(websocket, MessageType.AGENT_SPEAKING, {
        "agent": current_agent_name
    })

    try:
        # Get existing voice history from session (for maintaining conversation context)
        voice_history = session.get("voice_input_history", [])

        # Define callbacks for real-time updates
        async def on_transcription(text: str):
            """Called immediately when user speech is transcribed."""
            print(f"🎤 USER SAID | {short_id} | \"{text}\"")
            await send_ws_message(websocket, MessageType.USER_TRANSCRIPT, {
                "text": text,
                "role": "user"
            })
            session_manager.add_message(session_id, "user", text)

        async def on_tool_call(tool_name: str, call_id: str):
            """Called when a tool is invoked."""
            print(f"🔧 TOOL CALL | {short_id} | {tool_name}")
            await send_ws_message(websocket, MessageType.TOOL_CALL, {
                "tool": tool_name,
                "call_id": call_id,
            })

        async def on_tool_result(tool_name: str, call_id: str, output: str):
            """Called when a tool returns."""
            print(f"✅ TOOL RESULT | {short_id} | {tool_name} → {output[:50]}...")
            await send_ws_message(websocket, MessageType.TOOL_RESULT, {
                "tool": tool_name,
                "call_id": call_id,
                "result": output[:200],
                "status": "completed",
            })

        # Create VoicePipeline with context-aware workflow and callbacks
        pipeline, workflow = create_voice_pipeline(
            agent=current_agent,
            voice=current_voice,
            context=context,  # Pass CustomerContext for tools!
            input_history=voice_history,  # Maintain conversation history
            on_transcription=on_transcription,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )

        # Create audio input from accumulated audio
        # AudioInput expects a numpy array (int16 or float32), not raw bytes
        import numpy as np
        from agents.voice import AudioInput

        # Convert PCM16 bytes to numpy int16 array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_input = AudioInput(
            buffer=audio_array,
            frame_rate=CONFIG_AUDIO_SAMPLE_RATE,  # 24000 Hz
            sample_width=2,  # 16-bit = 2 bytes
            channels=1,  # Mono
        )

        # Run the pipeline
        result = await pipeline.run(audio_input)

        # Track response text for history
        response_text = ""
        audio_chunk_count = 0

        # Stream events back to client
        async for event in result.stream():
            # Get event type - could be class name or type attribute
            event_class = type(event).__name__
            event_type = getattr(event, 'type', event_class)

            # Debug: print event info
            print(f"📡 EVENT | {short_id} | {event_class} | type={event_type}")

            # Handle audio events
            if event_class == "VoiceStreamEventAudio" or "audio" in event_type.lower():
                # Get audio data - could be 'data' or 'audio' attribute
                # Note: data might be a numpy array, not bytes
                audio_data = getattr(event, 'data', None)
                if audio_data is None:
                    audio_data = getattr(event, 'audio', None)

                if audio_data is not None:
                    # Convert numpy array to bytes if needed
                    import numpy as np
                    if isinstance(audio_data, np.ndarray):
                        # Convert to int16 bytes for PCM audio
                        if audio_data.dtype == np.float32:
                            # Convert float32 [-1, 1] to int16
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                            audio_bytes = audio_int16.tobytes()
                        else:
                            audio_bytes = audio_data.tobytes()
                    elif isinstance(audio_data, bytes):
                        audio_bytes = audio_data
                    else:
                        audio_bytes = bytes(audio_data)

                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    await send_ws_message(websocket, MessageType.AUDIO_OUTPUT, {
                        "audio": audio_b64
                    })
                    audio_chunk_count += 1
                    print(f"🔊 AUDIO CHUNK | {short_id} | {len(audio_bytes)} bytes")

            # Handle lifecycle events
            elif event_class == "VoiceStreamEventLifecycle" or "lifecycle" in event_type.lower():
                lifecycle_event = getattr(event, 'event', '') or getattr(event, 'lifecycle', '')
                if lifecycle_event == "turn_started":
                    print(f"🚀 TURN START | {short_id}")
                elif lifecycle_event == "turn_ended":
                    print(f"✅ TURN END | {short_id} | {audio_chunk_count} audio chunks")

            # Handle error events
            elif event_class == "VoiceStreamEventError" or "error" in event_type.lower():
                error = getattr(event, 'error', '') or getattr(event, 'message', '')
                print(f"❌ VOICE ERROR | {short_id} | {error}")

        # User transcript is already handled by the on_transcription callback
        # which fires immediately when STT completes, before the agent responds

        # Get agent response text - StreamedAudioResult uses total_output_text
        response_text = getattr(result, 'total_output_text', None) or ""
        print(f"📊 RESPONSE | {short_id} | total_output_text={response_text[:100] if response_text else 'None'}...")

        if response_text:
            print(f"🤖 AGENT SAID | {short_id} | \"{response_text[:100]}{'...' if len(response_text) > 100 else ''}\"")
            session_manager.add_message(session_id, "assistant", response_text, current_agent_name)

            # Send transcript to frontend
            await send_ws_message(websocket, MessageType.TRANSCRIPT, {
                "text": response_text,
                "role": "assistant",
                "agent": current_agent_name
            })

        # Update voice history in session for next turn
        session["voice_input_history"] = workflow.input_history

        # Check for handoff using our custom workflow's last_agent property
        if workflow.last_agent and workflow.last_agent.name != current_agent_name:
            new_agent_name = workflow.last_agent.name
            print(f"🔀 HANDOFF | {short_id} | {current_agent_name} → {new_agent_name}")
            session_manager.update_current_agent(session_id, new_agent_name)
            await send_ws_message(websocket, MessageType.HANDOFF, {
                "from_agent": current_agent_name,
                "to_agent": new_agent_name,
                "message": f"Transferring you to {new_agent_name}..."
            })

        # Send context update (tools have now updated context!)
        await send_ws_message(websocket, MessageType.CONTEXT_UPDATE, {
            "name": context.name,
            "email": context.email,
            "phone": context.phone,
            "info_complete": context.info_collection_complete,
            "products_discussed": context.products_discussed,
            "selected_product": context.selected_product,
            "current_agent": session["current_agent"],
        })

    except Exception as e:
        print(f"❌ VOICE HANDLER ERROR | {short_id} | {e}")
        import traceback
        traceback.print_exc()
        await send_ws_message(websocket, MessageType.ERROR, {
            "error": str(e)
        })

    finally:
        # Notify frontend that agent is done
        await send_ws_message(websocket, MessageType.AGENT_DONE, {
            "agent": session["current_agent"]
        })


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
