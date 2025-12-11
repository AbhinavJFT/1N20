/**
 * Voice Sales Agent - Frontend Application
 * Handles WebSocket communication, chat UI, and voice mode
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    WS_URL: `ws://${window.location.hostname}:8000/ws`,
    AUDIO_SAMPLE_RATE: 24000,
    AUDIO_CHANNELS: 1,
    RECONNECT_DELAY: 3000,
    MAX_RECONNECT_ATTEMPTS: 5,
};

// ============================================
// Application State
// ============================================
const state = {
    sessionId: null,
    websocket: null,
    isConnected: false,
    isConversationActive: false,
    currentMode: 'chat', // 'chat' or 'voice'
    currentAgent: 'GreetingAgent',

    // Customer info
    customerInfo: {
        name: null,
        email: null,
        phone: null,
    },

    // Voice mode state
    isRecording: false,
    mediaRecorder: null,
    audioContext: null,
    audioWorklet: null,
    audioQueue: [],
    isPlaying: false,

    // Transcripts
    userTranscript: '',
    agentTranscript: '',
    partialTranscript: '',

    // Products
    productsDiscussed: [],

    // Activity log
    activityLog: [],
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    // Connection & Mode
    connectionStatus: document.getElementById('connectionStatus'),
    headerStatus: document.getElementById('headerStatus'),

    // Agent Info
    agentAvatar: document.getElementById('agentAvatar'),
    currentAgentName: document.getElementById('currentAgentName'),
    currentAgentRole: document.getElementById('currentAgentRole'),
    agentIndicator: document.getElementById('agentIndicator'),

    // Customer Info
    customerName: document.getElementById('customerName'),
    customerEmail: document.getElementById('customerEmail'),
    customerPhone: document.getElementById('customerPhone'),
    infoProgress: document.getElementById('infoProgress'),
    infoProgressText: document.getElementById('infoProgressText'),

    // Activity & Products
    toolActivityList: document.getElementById('toolActivityList'),
    productsList: document.getElementById('productsList'),

    // Chat
    messagesContainer: document.getElementById('messagesContainer'),
    chatSubtitle: document.getElementById('chatSubtitle'),

    // Input
    startBtn: document.getElementById('startBtn'),
    endBtn: document.getElementById('endBtn'),
    chatInputContainer: document.getElementById('chatInputContainer'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),

    // Voice Mode
    voiceToggleBtn: document.getElementById('voiceToggleBtn'),
    voiceStatusBar: document.getElementById('voiceStatusBar'),
    voiceStatusText: document.getElementById('voiceStatusText'),
    interruptBtn: document.getElementById('interruptBtn'),

    // Notifications
    handoffNotification: document.getElementById('handoffNotification'),
    handoffMessage: document.getElementById('handoffMessage'),
    toastContainer: document.getElementById('toastContainer'),
};

// ============================================
// Utility Functions
// ============================================

/**
 * Generate a unique session ID
 */
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

/**
 * Format timestamp for display
 */
function formatTime(date = new Date()) {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });
}

/**
 * Transliterate non-English text to English characters (Hinglish support)
 * This is a simplified version - for production use a proper library
 */
function transliterateToEnglish(text) {
    // Basic Hindi to English transliteration map
    const hindiToEnglish = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
        'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
        'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
        'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
        'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
        'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
        'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
        'ष': 'sha', 'स': 'sa', 'ह': 'ha', 'ा': 'a', 'ि': 'i',
        'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'े': 'e', 'ै': 'ai',
        'ो': 'o', 'ौ': 'au', '्': '', 'ं': 'n', 'ः': 'h',
        '।': '.', '॥': '.',
        // Numbers
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    };

    let result = '';
    for (const char of text) {
        result += hindiToEnglish[char] || char;
    }
    return result;
}

/**
 * Check if text contains non-English characters
 */
function containsNonEnglish(text) {
    // Check for non-ASCII characters (excluding common punctuation)
    return /[^\x00-\x7F]/.test(text);
}

/**
 * Process text for display (transliterate if needed)
 */
function processTextForDisplay(text) {
    if (containsNonEnglish(text)) {
        return transliterateToEnglish(text);
    }
    return text;
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle',
        warning: 'fa-exclamation-triangle',
    };

    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span>${message}</span>
    `;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100px)';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Scroll messages container to bottom
 */
function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// ============================================
// UI Update Functions
// ============================================

/**
 * Update connection status display
 */
function updateConnectionStatus(status) {
    const statusDot = elements.connectionStatus.querySelector('.status-dot');
    const statusText = elements.connectionStatus.querySelector('span');

    statusDot.className = 'status-dot ' + status;

    const statusMessages = {
        connected: 'Connected',
        connecting: 'Connecting...',
        disconnected: 'Disconnected',
    };

    statusText.textContent = statusMessages[status] || status;
}

/**
 * Update current agent display
 */
function updateAgentDisplay(agentName) {
    state.currentAgent = agentName;
    elements.currentAgentName.textContent = agentName;

    if (agentName === 'GreetingAgent') {
        elements.currentAgentRole.textContent = 'Collecting your information';
        elements.agentAvatar.classList.remove('sales');
    } else if (agentName === 'SalesAgent') {
        elements.currentAgentRole.textContent = 'Helping you find products';
        elements.agentAvatar.classList.add('sales');
    }
}

/**
 * Update agent indicator (idle, active, speaking)
 */
function updateAgentIndicator(status) {
    elements.agentIndicator.className = 'agent-indicator ' + status;
}

/**
 * Update customer info display
 */
function updateCustomerInfo(info) {
    state.customerInfo = { ...state.customerInfo, ...info };

    // Update name
    if (info.name) {
        elements.customerName.textContent = info.name;
        elements.customerName.classList.remove('empty');
        elements.customerName.classList.add('filled');
    }

    // Update email
    if (info.email) {
        elements.customerEmail.textContent = info.email;
        elements.customerEmail.classList.remove('empty');
        elements.customerEmail.classList.add('filled');
    }

    // Update phone
    if (info.phone) {
        elements.customerPhone.textContent = info.phone;
        elements.customerPhone.classList.remove('empty');
        elements.customerPhone.classList.add('filled');
    }

    // Update progress
    let collected = 0;
    if (state.customerInfo.name) collected++;
    if (state.customerInfo.email) collected++;
    if (state.customerInfo.phone) collected++;

    const percentage = (collected / 3) * 100;
    elements.infoProgress.style.width = percentage + '%';
    elements.infoProgressText.textContent = `${collected}/3 collected`;
}

/**
 * Add activity item to the log
 */
function addActivity(text, type = 'info') {
    // Remove empty state if present
    const emptyState = elements.toolActivityList.querySelector('.activity-empty');
    if (emptyState) emptyState.remove();

    const activity = document.createElement('div');
    activity.className = `activity-item ${type}`;

    const icons = {
        'tool-start': 'fa-cog fa-spin',
        'tool-end': 'fa-check',
        'tool-error': 'fa-times',
        'info': 'fa-info-circle',
    };

    activity.innerHTML = `
        <i class="fas ${icons[type] || icons.info}"></i>
        <span class="activity-text">${text}</span>
        <span class="activity-time">${formatTime()}</span>
    `;

    elements.toolActivityList.insertBefore(activity, elements.toolActivityList.firstChild);

    // Limit to 20 items
    while (elements.toolActivityList.children.length > 20) {
        elements.toolActivityList.lastChild.remove();
    }

    state.activityLog.push({ text, type, time: new Date() });
}

/**
 * Add product to discussed list
 */
function addProduct(productName) {
    if (state.productsDiscussed.includes(productName)) return;

    state.productsDiscussed.push(productName);

    // Remove empty state if present
    const emptyState = elements.productsList.querySelector('.activity-empty');
    if (emptyState) emptyState.remove();

    const product = document.createElement('div');
    product.className = 'product-item';
    product.innerHTML = `
        <i class="fas fa-box"></i>
        <span>${productName}</span>
    `;

    elements.productsList.appendChild(product);
}

/**
 * Format message text with markdown-like styling
 * Converts numbered lists, bold text, etc. to HTML
 * Handles streaming by hiding incomplete markdown syntax
 */
function formatMessageText(text, isStreaming = false) {
    if (!text) return '';

    let formatted = text;

    // Escape HTML first
    formatted = formatted
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Bold text: **text** - handle complete pairs
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Also handle __text__
    formatted = formatted.replace(/__([^_]+)__/g, '<strong>$1</strong>');

    // Hide incomplete markdown at the end (streaming case)
    formatted = formatted.replace(/\*\*([^*]*)$/, '<strong>$1</strong>');

    // Hide single trailing *
    formatted = formatted.replace(/(?<!\*)\*([^*]*)$/, '<em>$1</em>');

    // Format numbered lists: "1. Title: description" or "1. Title. description"
    // Captures: number, title (up to first : or .), and rest of the line
    formatted = formatted.replace(
        /(\d+)\.\s+([^:\.\n]+[:\.])\s*([^\n]*)/g,
        '<div class="list-item"><span class="list-number">$1.</span><span class="list-title">$2</span> <span class="list-desc">$3</span></div>'
    );

    // Format bullet points
    formatted = formatted.replace(/[-•]\s+([^\n<]+)/g, '<div class="bullet-item"><span class="bullet">•</span>$1</div>');

    // Format lines that look like headers (ending with :)
    formatted = formatted.replace(/^([A-Z][^:\n]{3,}):$/gm, '<div class="section-header">$1</div>');

    // Convert newlines to <br> for proper line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    // Clean up multiple <br> tags
    formatted = formatted.replace(/(<br>){3,}/g, '<br><br>');

    // Remove <br> right after list items
    formatted = formatted.replace(/<\/div><br>/g, '</div>');

    return formatted;
}

/**
 * Add message to chat
 */
function addMessage(text, sender = 'agent', agentName = null, isTyping = false) {
    const processedText = processTextForDisplay(text);
    const formattedText = isTyping ? processedText : formatMessageText(processedText);

    const message = document.createElement('div');
    const isUser = sender === 'user';
    const isSales = agentName === 'SalesAgent';

    message.className = `message ${isUser ? 'user-message' : 'agent-message'} ${isSales ? 'sales' : ''}`;

    const avatar = isUser ?
        '<i class="fas fa-user"></i>' :
        (isSales ? '<i class="fas fa-shopping-bag"></i>' : '<i class="fas fa-headset"></i>');

    const senderName = isUser ? 'You' : (agentName || 'Assistant');

    message.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-bubble">
            <div class="message-header">
                <span class="message-sender">${senderName}</span>
                <span class="message-time">${formatTime()}</span>
            </div>
            <div class="message-text">${formattedText}</div>
        </div>
    `;

    elements.messagesContainer.appendChild(message);
    scrollToBottom();

    return message;
}

/**
 * Add system message to chat
 */
function addSystemMessage(text) {
    const message = document.createElement('div');
    message.className = 'message system-message';
    message.innerHTML = `
        <div class="message-content">
            <i class="fas fa-info-circle"></i>
            <span>${text}</span>
        </div>
    `;

    elements.messagesContainer.appendChild(message);
    scrollToBottom();
}

/**
 * Add handoff message to chat
 */
function addHandoffMessage(fromAgent, toAgent) {
    const message = document.createElement('div');
    message.className = 'message handoff-message';
    message.innerHTML = `
        <div class="message-content">
            <i class="fas fa-exchange-alt"></i>
            <div class="handoff-info">
                <span class="handoff-title">Transferred to ${toAgent}</span>
                <span class="handoff-subtitle">From ${fromAgent}</span>
            </div>
        </div>
    `;

    elements.messagesContainer.appendChild(message);
    scrollToBottom();
}

/**
 * Update partial transcript (typing indicator)
 */
let currentTypingMessage = null;
let currentTypingAgentName = null;

function updatePartialTranscript(text, agentName) {
    const processedText = processTextForDisplay(text);

    if (!currentTypingMessage) {
        currentTypingMessage = addMessage('', 'agent', agentName, true); // true = isTyping
        currentTypingAgentName = agentName;
    }

    const textElement = currentTypingMessage.querySelector('.message-text');
    textElement.innerHTML = formatMessageText(processedText); // Apply formatting in real-time
    textElement.classList.add('typing');

    scrollToBottom();
}

function finalizeTranscript(finalText, agentName) {
    if (currentTypingMessage) {
        const textElement = currentTypingMessage.querySelector('.message-text');
        textElement.classList.remove('typing');

        // Apply final formatted text
        if (finalText) {
            const processedText = processTextForDisplay(finalText);
            textElement.innerHTML = formatMessageText(processedText);
        }

        currentTypingMessage = null;
        currentTypingAgentName = null;
    }
}

/**
 * Show handoff notification
 */
function showHandoffNotification(fromAgent, toAgent) {
    elements.handoffMessage.textContent = `Transferred from ${fromAgent} to ${toAgent}`;
    elements.handoffNotification.classList.remove('hidden');
    elements.handoffNotification.classList.add('show');

    setTimeout(() => {
        elements.handoffNotification.classList.remove('show');
        setTimeout(() => {
            elements.handoffNotification.classList.add('hidden');
        }, 300);
    }, 3000);
}

// ============================================
// WebSocket Communication
// ============================================

/**
 * Connect to WebSocket server
 */
function connectWebSocket() {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        return;
    }

    state.sessionId = generateSessionId();
    const wsUrl = `${CONFIG.WS_URL}/${state.sessionId}`;

    console.log('Connecting to WebSocket:', wsUrl);
    updateConnectionStatus('connecting');

    state.websocket = new WebSocket(wsUrl);

    state.websocket.onopen = handleWebSocketOpen;
    state.websocket.onmessage = handleWebSocketMessage;
    state.websocket.onclose = handleWebSocketClose;
    state.websocket.onerror = handleWebSocketError;
}

/**
 * Disconnect WebSocket
 */
function disconnectWebSocket() {
    if (state.websocket) {
        state.websocket.close();
        state.websocket = null;
    }

    state.isConnected = false;
    state.isConversationActive = false;
    updateConnectionStatus('disconnected');
}

/**
 * Send message through WebSocket
 */
function sendWebSocketMessage(type, data) {
    if (!state.websocket || state.websocket.readyState !== WebSocket.OPEN) {
        console.error('WebSocket not connected');
        return false;
    }

    const message = {
        type: type,
        data: data,
        timestamp: new Date().toISOString(),
    };

    state.websocket.send(JSON.stringify(message));
    return true;
}

/**
 * Handle WebSocket open
 */
function handleWebSocketOpen(event) {
    console.log('WebSocket connected');
    state.isConnected = true;
    updateConnectionStatus('connected');
    showToast('Connected to assistant', 'success');
}

/**
 * Handle WebSocket message
 */
function handleWebSocketMessage(event) {
    try {
        const message = JSON.parse(event.data);
        console.log('Received message:', message.type, message.data);

        switch (message.type) {
            case 'session_started':
                handleSessionStarted(message.data);
                break;

            case 'session_ended':
                handleSessionEnded(message.data);
                break;

            case 'transcript':
                handleTranscript(message.data);
                break;

            case 'partial_transcript':
                handlePartialTranscript(message.data);
                break;

            case 'user_transcript':
                handleUserTranscript(message.data);
                break;

            case 'audio_output':
                handleAudioOutput(message.data);
                break;

            case 'tool_call':
                handleToolCall(message.data);
                break;

            case 'tool_result':
                handleToolResult(message.data);
                break;

            case 'handoff':
                handleHandoff(message.data);
                break;

            case 'context_update':
                handleContextUpdate(message.data);
                break;

            case 'agent_speaking':
                handleAgentSpeaking(message.data);
                break;

            case 'agent_done':
                handleAgentDone(message.data);
                break;

            case 'error':
                handleError(message.data);
                break;

            default:
                console.log('Unknown message type:', message.type);
        }
    } catch (error) {
        console.error('Error parsing WebSocket message:', error);
    }
}

/**
 * Handle WebSocket close
 */
function handleWebSocketClose(event) {
    console.log('WebSocket closed:', event.code, event.reason);
    state.isConnected = false;
    state.isConversationActive = false;
    updateConnectionStatus('disconnected');

    if (event.code !== 1000) {
        showToast('Connection lost. Please restart the conversation.', 'error');
    }

    resetUIState();
}

/**
 * Handle WebSocket error
 */
function handleWebSocketError(error) {
    console.error('WebSocket error:', error);
    showToast('Connection error occurred', 'error');
}

// ============================================
// Message Handlers
// ============================================

function handleSessionStarted(data) {
    console.log('Session started:', data);
    state.isConversationActive = true;
    addSystemMessage(data.message || 'Connected to sales assistant');
    updateAgentIndicator('active');
}

function handleSessionEnded(data) {
    console.log('Session ended:', data);
    state.isConversationActive = false;
    addSystemMessage('Conversation ended. Thank you for chatting with us!');
    resetUIState();
}

function handleTranscript(data) {
    // Finalize the typing message with the final text
    if (data.role === 'assistant') {
        finalizeTranscript(data.text, data.agent || state.currentAgent);
    }

    // Update voice transcript if in voice mode
    if (state.currentMode === 'voice') {
        elements.agentTranscriptText.textContent = processTextForDisplay(data.text);
    }
}

function handlePartialTranscript(data) {
    if (data.role === 'assistant') {
        // Always show in chat (both chat and voice mode)
        updatePartialTranscript(data.text, data.agent || state.currentAgent);

        // Update voice status bar if in voice mode
        if (isVoiceModeActive) {
            isAgentSpeaking = true;
            pauseListening();
            updateVoiceStatus('agent-speaking');
        }
    }

    updateAgentIndicator('speaking');
}

function handleUserTranscript(data) {
    console.log('[USER_TRANSCRIPT] Received:', data.text);

    // Transliterate to English if contains non-English characters
    const displayText = processTextForDisplay(data.text);

    // Always show user message in chat
    addMessage(displayText, 'user');

    // In voice mode, pause listening and wait for agent
    if (isVoiceModeActive) {
        pauseListening();
    }
}

function handleAudioOutput(data) {
    console.log('[AUDIO_OUTPUT] Received, isVoiceModeActive:', isVoiceModeActive, 'hasAudio:', !!data.audio);
    // Play audio when in voice mode
    if (data.audio && isVoiceModeActive) {
        // Ensure we're not listening while agent audio plays
        if (!isAgentSpeaking) {
            console.log('[AUDIO_OUTPUT] Agent started speaking, pausing listening');
            isAgentSpeaking = true;
            pendingAgentDone = false;  // Reset pending flag when new audio starts
            pauseListening();
            updateVoiceStatus('agent-speaking');
        }
        playAudio(data.audio);
    }
    updateAgentIndicator('speaking');
}

function handleToolCall(data) {
    const toolName = formatToolName(data.tool);
    addActivity(`Calling ${toolName}...`, 'tool-start');

    // Special handling for specific tools
    if (data.tool === 'search_products') {
        addActivity('Searching product catalog...', 'info');
    }
}

function handleToolResult(data) {
    const toolName = formatToolName(data.tool);

    if (data.status === 'completed') {
        addActivity(`${toolName} completed`, 'tool-end');
    } else {
        addActivity(`${toolName} failed`, 'tool-error');
    }

    // Handle specific tool results
    if (data.tool === 'save_product_interest' && data.result) {
        // Extract product name from result
        const match = data.result.match(/Recorded interest in: (.+)/);
        if (match) {
            addProduct(match[1]);
        }
    }
}

function handleHandoff(data) {
    console.log('Handoff:', data);

    updateAgentDisplay(data.to_agent);
    addHandoffMessage(data.from_agent, data.to_agent);
    showHandoffNotification(data.from_agent, data.to_agent);
    showToast(`Transferred to ${data.to_agent}`, 'info');
}

function handleContextUpdate(data) {
    updateCustomerInfo({
        name: data.name,
        email: data.email,
        phone: data.phone,
    });

    if (data.current_agent) {
        updateAgentDisplay(data.current_agent);
    }

    // Update products discussed
    if (data.products_discussed && Array.isArray(data.products_discussed)) {
        data.products_discussed.forEach(product => addProduct(product));
    }
}

function handleAgentSpeaking(data) {
    updateAgentIndicator('speaking');
    isAgentSpeaking = true;
    if (isVoiceModeActive) {
        // Pause listening while agent speaks
        pauseListening();
        updateVoiceStatus('agent-speaking');
    }
}

function handleAgentDone(data) {
    console.log('[AGENT_DONE] Received, isVoiceModeActive:', isVoiceModeActive, 'audioQueue:', audioQueue.length, 'isProcessingAudio:', isProcessingAudio);
    finalizeTranscript();
    updateAgentIndicator('active');

    // Check if audio is still playing
    if (audioQueue.length > 0 || isProcessingAudio) {
        console.log('[AGENT_DONE] Audio still playing, marking pendingAgentDone=true');
        pendingAgentDone = true;
        // Don't resume listening yet - the audio processor will handle it
        return;
    }

    // No audio playing, can resume immediately
    isAgentSpeaking = false;
    if (isVoiceModeActive) {
        console.log('[AGENT_DONE] No audio pending, resuming listening');
        resumeListening();
        updateVoiceStatus('listening');
    }
}

function handleError(data) {
    console.error('Server error:', data);

    if (data.type === 'guardrail') {
        showToast(data.message, 'warning');
        addSystemMessage(data.message);
    } else {
        showToast(data.error || 'An error occurred', 'error');
    }
}

/**
 * Format tool name for display
 */
function formatToolName(toolName) {
    return toolName
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

// ============================================
// Audio Handling
// ============================================

let audioContextInstance = null;
let audioQueue = [];
let isProcessingAudio = false;
let pendingAgentDone = false;  // Track if we received agent_done while audio is still playing

/**
 * Initialize audio context
 */
async function initAudioContext() {
    if (!audioContextInstance) {
        audioContextInstance = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: CONFIG.AUDIO_SAMPLE_RATE,
        });
    }

    if (audioContextInstance.state === 'suspended') {
        await audioContextInstance.resume();
    }

    return audioContextInstance;
}

/**
 * Play audio from base64 encoded PCM16 data
 */
async function playAudio(base64Audio) {
    try {
        const audioContext = await initAudioContext();

        // Decode base64 to binary
        const binaryString = atob(base64Audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert PCM16 to Float32
        const pcm16 = new Int16Array(bytes.buffer);
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) {
            float32[i] = pcm16[i] / 32768.0;
        }

        // Create audio buffer
        const audioBuffer = audioContext.createBuffer(
            CONFIG.AUDIO_CHANNELS,
            float32.length,
            CONFIG.AUDIO_SAMPLE_RATE
        );
        audioBuffer.getChannelData(0).set(float32);

        // Queue and play
        audioQueue.push(audioBuffer);
        console.log('[AUDIO] Added to queue, queue length:', audioQueue.length);
        processAudioQueue();

    } catch (error) {
        console.error('Error playing audio:', error);
    }
}

/**
 * Process audio queue
 */
async function processAudioQueue() {
    if (isProcessingAudio || audioQueue.length === 0) return;

    isProcessingAudio = true;
    const audioContext = await initAudioContext();

    while (audioQueue.length > 0) {
        const buffer = audioQueue.shift();
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);

        await new Promise(resolve => {
            source.onended = resolve;
            source.start();
        });
    }

    isProcessingAudio = false;
    console.log('[AUDIO] Queue empty, isProcessingAudio:', isProcessingAudio, 'pendingAgentDone:', pendingAgentDone);

    // Check if we received agent_done while audio was playing
    if (pendingAgentDone) {
        console.log('[AUDIO] Processing pending agent done - resuming listening');
        pendingAgentDone = false;
        isAgentSpeaking = false;
        if (isVoiceModeActive) {
            resumeListening();
            updateVoiceStatus('listening');
        }
    }
}

// ============================================
// Voice Recording
// ============================================

let mediaStream = null;
let audioProcessor = null;
let isVoiceModeActive = false;  // Voice mode is on (UI state)
let isListening = false;         // Currently capturing and sending audio
let isAgentSpeaking = false;     // Track if agent is currently speaking

/**
 * Start voice mode (continuous recording with turn-based control)
 */
async function startVoiceMode() {
    try {
        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: CONFIG.AUDIO_SAMPLE_RATE,
                channelCount: CONFIG.AUDIO_CHANNELS,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        const audioContext = await initAudioContext();
        const source = audioContext.createMediaStreamSource(mediaStream);

        // Create script processor for capturing audio
        const bufferSize = 4096;
        audioProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);

        audioProcessor.onaudioprocess = (event) => {
            // Only send audio when actively listening (not when agent is speaking)
            if (!isVoiceModeActive || !isListening) return;

            const inputData = event.inputBuffer.getChannelData(0);

            // Convert Float32 to PCM16
            const pcm16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                pcm16[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
            }

            // Convert to base64
            const bytes = new Uint8Array(pcm16.buffer);
            let binary = '';
            for (let i = 0; i < bytes.length; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64Audio = btoa(binary);

            // Send to server
            sendWebSocketMessage('audio_input', { audio: base64Audio });
        };

        source.connect(audioProcessor);
        audioProcessor.connect(audioContext.destination);

        isVoiceModeActive = true;
        isListening = true;  // Start listening immediately
        state.isRecording = true;

        // Update UI - show voice status bar, change mic button
        elements.voiceStatusBar.classList.remove('hidden');
        elements.voiceToggleBtn.classList.add('active');
        elements.voiceToggleBtn.innerHTML = '<i class="fas fa-stop"></i>';
        elements.voiceToggleBtn.title = 'Stop voice mode';

        updateHeaderStatus('voice');
        updateVoiceStatus('listening');

        showToast('Voice mode activated - speak now', 'success');

    } catch (error) {
        console.error('Error starting voice mode:', error);
        showToast('Could not access microphone', 'error');
    }
}

/**
 * Pause listening (when user finishes speaking or agent starts)
 */
function pauseListening() {
    console.log('[VOICE] Pausing listening');
    isListening = false;
    if (isVoiceModeActive) {
        updateVoiceStatus('agent-speaking');
    }
}

/**
 * Resume listening (when agent finishes speaking)
 */
function resumeListening() {
    console.log('[VOICE] Resuming listening');
    if (isVoiceModeActive) {
        isListening = true;
        updateVoiceStatus('listening');
    }
}

/**
 * Stop voice mode
 */
function stopVoiceMode() {
    isVoiceModeActive = false;
    isListening = false;
    state.isRecording = false;

    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Update UI - hide voice status bar, reset mic button
    elements.voiceStatusBar.classList.add('hidden');
    elements.voiceToggleBtn.classList.remove('active');
    elements.voiceToggleBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    elements.voiceToggleBtn.title = 'Start voice mode';

    updateHeaderStatus('connected');

    showToast('Voice mode stopped', 'info');
}

/**
 * Update voice status bar
 */
function updateVoiceStatus(status) {
    const statusBar = elements.voiceStatusBar;
    const statusText = elements.voiceStatusText;
    const interruptBtn = elements.interruptBtn;

    statusBar.classList.remove('agent-speaking', 'paused');

    switch (status) {
        case 'listening':
            statusText.textContent = 'Listening...';
            interruptBtn.classList.add('hidden');
            break;
        case 'processing':
            statusText.textContent = 'Processing...';
            interruptBtn.classList.add('hidden');
            break;
        case 'agent-speaking':
            statusBar.classList.add('agent-speaking');
            statusText.textContent = 'Agent speaking...';
            interruptBtn.classList.remove('hidden');
            break;
        case 'paused':
            statusBar.classList.add('paused');
            statusText.textContent = 'Paused';
            interruptBtn.classList.add('hidden');
            break;
    }
}

/**
 * Update header status badge
 */
function updateHeaderStatus(status) {
    const badge = elements.headerStatus;
    const icon = badge.querySelector('i');
    const text = badge.querySelector('span');

    badge.className = 'status-badge';

    switch (status) {
        case 'connected':
            badge.classList.add('connected');
            icon.className = 'fas fa-circle';
            text.textContent = 'Connected';
            break;
        case 'voice':
            badge.classList.add('connected');
            icon.className = 'fas fa-microphone';
            text.textContent = 'Voice Mode';
            break;
        case 'disconnected':
            icon.className = 'fas fa-circle';
            text.textContent = 'Disconnected';
            break;
        default:
            icon.className = 'fas fa-circle';
            text.textContent = 'Ready';
    }
}

// ============================================
// UI State Management
// ============================================

/**
 * Reset UI to initial state
 */
function resetUIState() {
    // Stop any recording first
    if (state.isRecording || isVoiceModeActive) {
        isVoiceModeActive = false;
        isListening = false;
        state.isRecording = false;
        if (audioProcessor) {
            audioProcessor.disconnect();
            audioProcessor = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
    }

    // Reset buttons
    elements.startBtn.classList.remove('hidden');
    elements.endBtn.classList.add('hidden');
    elements.chatInputContainer.classList.add('hidden');
    elements.voiceStatusBar.classList.add('hidden');

    // Reset voice toggle button
    elements.voiceToggleBtn.classList.remove('active');
    elements.voiceToggleBtn.innerHTML = '<i class="fas fa-microphone"></i>';

    // Reset agent indicator
    updateAgentIndicator('');
    updateHeaderStatus('disconnected');

    // Clear typing message
    currentTypingMessage = null;
}

// ============================================
// Event Handlers
// ============================================

/**
 * Start conversation button click
 */
elements.startBtn.addEventListener('click', async () => {
    connectWebSocket();

    elements.startBtn.classList.add('hidden');
    elements.endBtn.classList.remove('hidden');
    elements.chatInputContainer.classList.remove('hidden');
    elements.messageInput.focus();

    updateHeaderStatus('connected');
});

/**
 * End conversation button click
 */
elements.endBtn.addEventListener('click', () => {
    sendWebSocketMessage('end_session', {});
    disconnectWebSocket();
    resetUIState();
});

/**
 * Send message button click
 */
elements.sendBtn.addEventListener('click', sendChatMessage);

/**
 * Message input enter key
 */
elements.messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

/**
 * Send chat message
 */
function sendChatMessage() {
    const text = elements.messageInput.value.trim();
    if (!text) return;

    // Add message to UI
    addMessage(text, 'user');

    // Send to server
    sendWebSocketMessage('text_input', { text });

    // Clear input
    elements.messageInput.value = '';
}

/**
 * Voice toggle button - click to toggle voice mode
 */
elements.voiceToggleBtn.addEventListener('click', async () => {
    if (!state.isConnected) {
        showToast('Please start the conversation first', 'warning');
        return;
    }

    if (isVoiceModeActive) {
        // Stop voice mode
        stopVoiceMode();
    } else {
        // Start voice mode
        await startVoiceMode();
    }
});

/**
 * Interrupt button - stop agent and start listening
 */
elements.interruptBtn.addEventListener('click', () => {
    if (!isVoiceModeActive || !isAgentSpeaking) return;

    console.log('[INTERRUPT] User interrupted agent');

    // Send interrupt signal to backend
    sendWebSocketMessage('interrupt', {});

    // Clear any playing audio
    clearAudioQueue();

    // Reset agent speaking state
    isAgentSpeaking = false;

    // Resume listening immediately
    resumeListening();
    updateVoiceStatus('listening');
    updateAgentIndicator('active');

    showToast('Agent interrupted - listening now', 'info');
});

/**
 * Clear the audio queue to stop playback
 */
function clearAudioQueue() {
    console.log('[AUDIO] Clearing audio queue');
    audioQueue = [];
    isProcessingAudio = false;
    pendingAgentDone = false;
}

/**
 * Wait for all queued audio to finish playing
 */
function waitForAudioComplete() {
    return new Promise((resolve) => {
        const checkAudio = () => {
            if (audioQueue.length === 0 && !isProcessingAudio) {
                resolve();
            } else {
                setTimeout(checkAudio, 100);
            }
        };
        checkAudio();
    });
}

// ============================================
// Initialization
// ============================================

/**
 * Initialize the application
 */
function init() {
    console.log('Voice Sales Agent Frontend initialized');

    // Update connection status
    updateConnectionStatus('disconnected');
    updateHeaderStatus('disconnected');

    // Initialize audio context on first user interaction
    document.addEventListener('click', async () => {
        await initAudioContext();
    }, { once: true });
}

// Start the application
init();
