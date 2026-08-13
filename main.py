from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import urllib.request

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GEMINI 3 FLASH PREVIEW SETUP
# ============================================
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai_client = None
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_AVAILABLE = False

if GOOGLE_API_KEY:
    try:
        genai_client = genai.Client(api_key=GOOGLE_API_KEY)
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Say 'OK'",
            config=types.GenerateContentConfig(max_output_tokens=5)
        )
        GEMINI_AVAILABLE = True
        print(f"✅ Gemini 3 Flash Preview connected!")
    except Exception as e:
        if "429" in str(e):
            print("⚠️ Rate limited - retry in 60s")
        elif "404" in str(e) or "not found" in str(e).lower():
            print(f"⚠️ {GEMINI_MODEL} not available, trying gemini-2.0-flash...")
            GEMINI_MODEL = "gemini-2.0-flash"
            try:
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents="Say 'OK'",
                    config=types.GenerateContentConfig(max_output_tokens=5)
                )
                GEMINI_AVAILABLE = True
                print(f"✅ Using {GEMINI_MODEL}")
            except:
                print("⚠️ Using simulation mode")
        else:
            print(f"⚠️ Gemini error: {e}")
else:
    print("⚠️ GEMINI_API_KEY not found - using simulation mode")

# ============================================
# DEPRESSION SCREENING PROMPT ENGINEERING
# ============================================
DEPRESSION_SCREENING_PROMPT = """You are MindGuard, a compassionate and professional mental health screening assistant. Your purpose is to conduct a structured conversation to assess potential depression symptoms.

## YOUR IDENTITY
- Name: MindGuard Assistant
- Role: Mental health screening companion
- Tone: Warm, empathetic, non-judgmental, professional
- Style: Natural conversation, not clinical interrogation

## SCREENING STRUCTURE
Follow this progressive conversation flow:

### PHASE 1: INTRODUCTION & RAPPORT BUILDING
Start by asking their name and building trust:
"Hi, I'm MindGuard, your mental health companion. Before we begin, I'd love to know - what should I call you?"

After they share their name, acknowledge them warmly:
"Thank you [name]. Remember, this is a safe space. Everything you share stays confidential. How have you been feeling lately, generally speaking?"

### PHASE 2: GENERAL WELLBEING EXPLORATION
Ask open-ended questions one at a time about:
- Overall mood over the past 2 weeks
- Daily energy levels
- General outlook on life
- Recent changes in feelings or behavior

Example: "On a scale of 1-10, how would you rate your overall mood over the past couple of weeks?"

### PHASE 3: CORE DEPRESSION SYMPTOMS (PHQ-9 Based)
Gently explore each area with natural questions:

1. **Anhedonia (Loss of Interest):**
   "Have you been finding less pleasure or interest in activities you used to enjoy? Like hobbies, spending time with others, or even simple things?"

2. **Depressed Mood:**
   "Have you been feeling down, depressed, or hopeless more days than not? How does that show up in your day-to-day life?"

3. **Sleep Disturbances:**
   "How has your sleep been? Are you sleeping too much, struggling to fall asleep, or waking up frequently during the night?"

4. **Fatigue/Energy Loss:**
   "How are your energy levels? Do you feel tired or drained even after resting?"

5. **Appetite Changes:**
   "Have you noticed any changes in your appetite or eating habits? Eating more than usual, or less?"

6. **Self-Worth:**
   "Sometimes when we're struggling, we can be hard on ourselves. Have you been feeling down on yourself, or like you're letting people down?"

7. **Concentration:**
   "Has it been harder to focus lately? Like trouble concentrating on work, studies, or even watching TV?"

8. **Psychomotor Changes:**
   "Have you noticed yourself moving or speaking more slowly than usual? Or feeling unusually restless?"

9. **Thoughts of Self-Harm (ASK WITH EXTREME CARE):**
   ONLY if the conversation naturally leads here and they've shown significant distress:
   "I want to check in about something important - have you had any thoughts about hurting yourself or that life isn't worth living?"

### PHASE 4: SUPPORT SYSTEM ASSESSMENT
- "Who do you talk to when you're feeling down?"
- "Do you feel supported by friends, family, or a community?"
- "Have you spoken to a professional about these feelings before?"

### PHASE 5: STRENGTHS & COPING
- "What helps you feel better, even a little bit?"
- "What keeps you going on difficult days?"
- "Is there anything you're looking forward to?"

## IMPORTANT RULES
1. **ONE QUESTION AT A TIME** - Never ask multiple questions in one message
2. **ACKNOWLEDGE RESPONSES** - Validate their feelings before the next question
3. **NATURAL FLOW** - Don't rush through the list; follow the conversation naturally
4. **CRISIS PROTOCOL**: If they mention suicide, self-harm, or immediate danger:
   - Stay calm and supportive
   - Say: "Thank you for sharing that with me. That takes courage. I want to make sure you're safe. Please consider calling 988 (Suicide & Crisis Lifeline) - they're available 24/7 by call or text. Would you be willing to reach out to them?"
   - Don't end the conversation abruptly
5. **NO DIAGNOSIS** - You screen, you don't diagnose. Always say "suggests" or "indicates" not "you have depression"
6. **CONFIDENTIALITY REMINDER** - Occasionally remind them this is a safe space
7. **KEEP RESPONSES 2-4 SENTENCES** - Be concise but warm

## CONVERSATION STYLE EXAMPLES

❌ BAD: "Do you have anhedonia, sleep disturbance, and psychomotor retardation?"
✅ GOOD: "I appreciate you sharing that. You mentioned feeling tired - how's your sleep been lately? Are you getting enough rest?"

❌ BAD: "Based on PHQ-9 criteria, you have moderate depression."
✅ GOOD: "From what you've shared, it sounds like you're going through a really tough time. These feelings you're describing are common when someone's struggling with low mood. Have you ever talked to a counselor or therapist about this?"

❌ BAD: "Next question: rate your concentration from 0-3."
✅ GOOD: "Thank you for opening up about that. I'm wondering - have you noticed any changes in your ability to focus or concentrate on things recently?"

Remember: You're having a conversation, not filling out a form. Be human, be kind, be present."""

# ============================================
# FACE DETECTION SETUP
# ============================================
print("\nLoading Face Detection Model...")
MODEL_TYPE = "none"
face_detector = None
face_cascade = None

try:
    face_detector = YOLO("yolov8n-face.pt") if os.path.exists("yolov8n-face.pt") else YOLO("yolov8n.pt")
    MODEL_TYPE = "yolov8"
    print("✅ YOLOv8 Face Detection Loaded")
    
    def detect_faces(image):
        results = face_detector(image, conf=0.3, verbose=False)
        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    if x2 > x1+20 and y2 > y1+20:
                        faces.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        return faces

except Exception as e:
    print(f"⚠️ YOLOv8 not available: {e}")
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        MODEL_TYPE = "haar_cascade"
        print("✅ OpenCV Haar Cascade Loaded")
        
        def detect_faces(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            return [{"bbox": [x, y, x+w, y+h], "confidence": 0.8} for (x, y, w, h) in faces]
    except:
        print("❌ No face detection available")
        def detect_faces(image):
            return []

# ============================================
# EMOTION ANALYSIS
# ============================================
def analyze_emotion(face_roi):
    if face_roi is None or face_roi.size == 0:
        return "neutral", 0.5
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        if brightness < 80: return "sad", 0.7
        elif brightness > 170: return "happy", 0.7
        elif contrast > 60: return "surprised", 0.65
        elif 80 <= brightness < 100: return "fearful", 0.6
        else: return "neutral", 0.7
    except:
        return "neutral", 0.5

# ============================================
# SIMULATION MODE RESPONSES
# ============================================
def generate_screening_response(text, conversation_stage):
    """Generate depression screening responses for simulation mode"""
    text_lower = text.lower()
    
    # Extract name if just shared
    if conversation_stage == "introduction":
        # Try to extract name
        words = text.split()
        potential_name = words[-1] if words else ""
        if len(potential_name) > 1 and potential_name[0].isupper():
            return f"Thank you, {potential_name}. This is a safe space. How have you been feeling lately?"
        return "This is a safe space - everything you share stays confidential. How have you been feeling lately, generally speaking?"
    
    if conversation_stage == "general_mood":
        if any(w in text_lower for w in ['bad', 'terrible', 'awful', 'horrible']):
            return "I'm sorry you're feeling that way. On a scale of 1-10, how would you rate your mood over the past couple of weeks?"
        elif any(w in text_lower for w in ['ok', 'okay', 'fine', 'alright']):
            return "I hear you saying things are okay. Have you been finding joy in activities you usually enjoy?"
        else:
            return "Thank you for sharing. Have you been finding less pleasure or interest in activities you used to enjoy?"
    
    if conversation_stage == "anhedonia":
        return "I see. How has your sleep been? Are you getting enough rest, or struggling with sleep?"
    
    if conversation_stage == "sleep":
        if any(w in text_lower for w in ['cant sleep', 'insomnia', 'waking up', 'nightmares']):
            return "Sleep troubles can really affect your mood. How are your energy levels during the day?"
        else:
            return "How are your energy levels? Do you feel tired or drained even after resting?"
    
    if conversation_stage == "energy":
        return "Have you noticed any changes in your appetite or eating habits recently?"
    
    if conversation_stage == "appetite":
        return "Sometimes when we're struggling, we can be hard on ourselves. Have you been feeling down on yourself lately?"
    
    if conversation_stage == "self_worth":
        if any(w in text_lower for w in ['yes', 'yeah', 'sometimes', 'always']):
            return "That must be really difficult. Has it been harder to focus or concentrate on things recently?"
        else:
            return "Has it been harder to focus or concentrate on things recently?"
    
    # Default empathetic responses
    responses = [
        "Thank you for sharing that with me. Can you tell me more about how that affects your daily life?",
        "I appreciate your honesty. How long have you been feeling this way?",
        "That sounds challenging. Do you have someone you can talk to about this?",
        "I hear you. What helps you feel better, even a little bit?",
    ]
    import random
    return random.choice(responses)

# ============================================
# DATA MODELS
# ============================================
class TextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class ImageRequest(BaseModel):
    image: str
    session_id: Optional[str] = None

class AssessmentRequest(BaseModel):
    session_id: str
    sleep_hours: float = 7.0
    mood_score: int = 5

class ConversationResponse(BaseModel):
    response: str
    session_id: str
    emotional_state: dict

# ============================================
# SESSION MANAGEMENT
# ============================================
chat_sessions: Dict[str, dict] = {}

def create_gemini_chat():
    """Create a new Gemini chat with depression screening prompt"""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        chat = genai_client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=DEPRESSION_SCREENING_PROMPT,
                temperature=0.7,
                max_output_tokens=300,
                top_p=0.9,
            )
        )
        return chat
    except Exception as e:
        print(f"Chat creation error: {e}")
        return None

def get_or_create_session(session_id: Optional[str] = None) -> tuple:
    if session_id and session_id in chat_sessions:
        return session_id, chat_sessions[session_id]
    
    new_id = datetime.now().strftime("%Y%m%d%H%M%S") + os.urandom(4).hex()
    
    chat_sessions[new_id] = {
        "chat": create_gemini_chat(),
        "messages": [],
        "face_analyses": [],
        "user_name": None,
        "conversation_stage": "introduction",
        "started_at": datetime.now().isoformat(),
        "depression_indicators": [],
        "risk_factors": []
    }
    
    return new_id, chat_sessions[new_id]

def analyze_depression_indicators(text):
    """Analyze text for depression indicators"""
    text_lower = text.lower()
    indicators = []
    
    # Check for key depression signals
    if any(w in text_lower for w in ['hopeless', 'worthless', 'no point', 'give up']):
        indicators.append("expressions_of_hopelessness")
    if any(w in text_lower for w in ['cant sleep', 'insomnia', 'sleeping too much', 'nightmares']):
        indicators.append("sleep_disturbance")
    if any(w in text_lower for w in ['tired', 'exhausted', 'no energy', 'fatigue']):
        indicators.append("fatigue")
    if any(w in text_lower for w in ['not eating', 'overeating', 'no appetite', 'eating too much']):
        indicators.append("appetite_changes")
    if any(w in text_lower for w in ['cant focus', 'cant concentrate', 'forgetting', 'scattered']):
        indicators.append("concentration_issues")
    if any(w in text_lower for w in ['no interest', 'dont enjoy', 'nothing matters', 'boring']):
        indicators.append("anhedonia")
    if any(w in text_lower for w in ['hate myself', 'failure', 'disappointment', 'let down']):
        indicators.append("low_self_worth")
    if any(w in text_lower for w in ['suicide', 'kill myself', 'end my life', 'want to die']):
        indicators.append("suicidal_ideation")
    if any(w in text_lower for w in ['alone', 'lonely', 'no one', 'isolated']):
        indicators.append("social_isolation")
    if any(w in text_lower for w in ['crying', 'tears', 'break down', 'emotional']):
        indicators.append("emotional_distress")
    
    return indicators

def update_conversation_stage(session):
    """Determine the current conversation stage based on messages"""
    messages = session.get("messages", [])
    user_messages = [m for m in messages if m['user'] != 'SYSTEM']
    count = len(user_messages)
    
    if count == 0:
        return "introduction"
    elif count == 1:
        return "general_mood"
    elif count == 2:
        return "anhedonia"
    elif count == 3:
        return "sleep"
    elif count == 4:
        return "energy"
    elif count == 5:
        return "appetite"
    elif count == 6:
        return "self_worth"
    elif count >= 7:
        return "deep_exploration"
    
    return "general_mood"

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "app": "MindGuard Depression Screening",
        "gemini_available": GEMINI_AVAILABLE,
        "gemini_model": GEMINI_MODEL,
        "face_detection": MODEL_TYPE,
        "mode": "gemini" if GEMINI_AVAILABLE else "simulation"
    }

@app.post("/chat/start")
async def start_chat():
    """Start a new depression screening conversation"""
    session_id, session = get_or_create_session()
    
    if session["chat"] and GEMINI_AVAILABLE:
        try:
            response = session["chat"].send_message(
                "Start the conversation by introducing yourself as MindGuard and asking for their name."
            )
            initial_message = response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            initial_message = "Hi, I'm MindGuard, your mental health companion. I'm here to listen and understand how you've been feeling. What should I call you?"
    else:
        initial_message = "Hi, I'm MindGuard, your mental health companion. I'm here to listen and understand how you've been feeling. What should I call you?"
    
    session["messages"].append({
        "user": "SYSTEM",
        "assistant": initial_message,
        "timestamp": datetime.now().isoformat()
    })
    
    print(f"✅ New screening session: {session_id}")
    
    return ConversationResponse(
        response=initial_message,
        session_id=session_id,
        emotional_state={
            "mode": "gemini" if GEMINI_AVAILABLE else "simulated",
            "stage": "introduction",
            "session_active": True
        }
    )

@app.post("/chat/message")
async def chat_message(request: TextRequest):
    """Handle screening conversation messages"""
    session_id, session = get_or_create_session(request.session_id)
    
    # Analyze depression indicators
    indicators = analyze_depression_indicators(request.text)
    session["depression_indicators"].extend(indicators)
    session["depression_indicators"] = list(set(session["depression_indicators"]))
    
    # Update conversation stage
    session["conversation_stage"] = update_conversation_stage(session)
    
    # Try to extract name from first message
    if session["conversation_stage"] == "introduction" and not session.get("user_name"):
        words = request.text.strip().split()
        if words:
            potential_name = words[-1].strip('.,!?')
            if len(potential_name) > 1 and potential_name[0].isupper():
                session["user_name"] = potential_name
    
    # Generate response
    if session.get("chat") and GEMINI_AVAILABLE:
        try:
            # Add context to help Gemini stay on track
            context = f"\n[Current screening stage: {session['conversation_stage']}]"
            full_message = request.text + context
            response = session["chat"].send_message(full_message)
            response_text = response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            response_text = generate_screening_response(request.text, session["conversation_stage"])
    else:
        response_text = generate_screening_response(request.text, session["conversation_stage"])
    
    # Store message
    session["messages"].append({
        "user": request.text,
        "assistant": response_text,
        "timestamp": datetime.now().isoformat(),
        "stage": session["conversation_stage"]
    })
    
    # Build emotional state
    text_lower = request.text.lower()
    emotional_state = {
        "depression_indicators": session["depression_indicators"],
        "indicator_count": len(session["depression_indicators"]),
        "conversation_stage": session["conversation_stage"],
        "crisis_risk": any(w in text_lower for w in ['suicide', 'kill myself', 'want to die', 'end my life']),
        "message_count": len([m for m in session["messages"] if m['user'] != 'SYSTEM'])
    }
    
    return ConversationResponse(
        response=response_text,
        session_id=session_id,
        emotional_state=emotional_state
    )

@app.post("/analyze/face")
async def analyze_face(request: ImageRequest):
    """Analyze face for emotional indicators"""
    try:
        img_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"faces_detected": 0, "emotions": [], "error": "Invalid image"}
        
        faces = detect_faces(img)
        emotions = []
        
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_roi = img[y1:y2, x1:x2]
            emotion, emotion_conf = analyze_emotion(face_roi)
            emotions.append({
                "emotion": emotion,
                "confidence": face["confidence"],
                "emotion_confidence": emotion_conf,
                "bbox": [x1, y1, x2, y2],
                "depression_relevant": emotion in ['sad', 'fearful', 'neutral']
            })
        
        if request.session_id and request.session_id in chat_sessions:
            chat_sessions[request.session_id].setdefault("face_analyses", []).append({
                "timestamp": datetime.now().isoformat(),
                "faces_detected": len(faces),
                "emotions": emotions
            })
        
        return {
            "faces_detected": len(faces),
            "emotions": emotions,
            "detection_method": MODEL_TYPE
        }
        
    except Exception as e:
        return {"faces_detected": 0, "emotions": [], "error": str(e)}

@app.post("/assess/comprehensive")
async def comprehensive_assessment(request: AssessmentRequest):
    """Generate comprehensive depression screening assessment"""
    session_id, session = get_or_create_session(request.session_id)
    
    # Compile conversation
    conversation = "\n".join([
        f"User: {msg['user']}\nAssistant: {msg['assistant']}"
        for msg in session.get("messages", [])
        if msg['user'] != 'SYSTEM'
    ])
    
    # Face analysis summary
    face_summary = "No facial data collected."
    depression_face_indicators = 0
    if session.get("face_analyses"):
        analyses = session["face_analyses"]
        total_frames = len(analyses)
        total_faces = sum(a["faces_detected"] for a in analyses)
        all_emotions = []
        for a in analyses:
            for e in a.get("emotions", []):
                all_emotions.append(e["emotion"])
                if e.get("depression_relevant"):
                    depression_face_indicators += 1
        
        if all_emotions:
            from collections import Counter
            emotion_counts = Counter(all_emotions)
            dominant = emotion_counts.most_common(1)[0]
            sad_ratio = all_emotions.count('sad') / len(all_emotions)
            face_summary = (
                f"{total_frames} frames analyzed. "
                f"Dominant emotion: {dominant[0]}. "
                f"Sad expressions: {all_emotions.count('sad')} ({sad_ratio*100:.0f}%). "
                f"Depression-relevant indicators: {depression_face_indicators}"
            )
    
    # Calculate comprehensive depression risk score
    text_lower = conversation.lower()
    risk_score = 0
    risk_factors = []
    
    # PHQ-9 based scoring
    # 1. Anhedonia
    if any(w in text_lower for w in ['no interest', 'dont enjoy', 'nothing matters', 'boring', 'lost interest']):
        risk_score += 2
        risk_factors.append("Loss of interest/pleasure (Anhedonia)")
    
    # 2. Depressed mood
    if any(w in text_lower for w in ['sad', 'depressed', 'hopeless', 'down', 'miserable']):
        risk_score += 2
        risk_factors.append("Depressed mood")
    
    # 3. Sleep issues
    if any(w in text_lower for w in ['cant sleep', 'insomnia', 'sleeping too much', 'nightmares', 'waking up']):
        risk_score += 1
        risk_factors.append("Sleep disturbances")
    
    # 4. Fatigue
    if any(w in text_lower for w in ['tired', 'exhausted', 'no energy', 'fatigue', 'drained']):
        risk_score += 1
        risk_factors.append("Fatigue/Low energy")
    
    # 5. Appetite
    if any(w in text_lower for w in ['appetite', 'eating', 'weight']):
        risk_score += 1
        risk_factors.append("Appetite/Weight changes")
    
    # 6. Self-worth
    if any(w in text_lower for w in ['failure', 'disappointment', 'let down', 'hate myself', 'worthless']):
        risk_score += 2
        risk_factors.append("Low self-worth")
    
    # 7. Concentration
    if any(w in text_lower for w in ['focus', 'concentrate', 'forgetting', 'scattered', 'memory']):
        risk_score += 1
        risk_factors.append("Concentration difficulties")
    
    # 8. Psychomotor
    if any(w in text_lower for w in ['slow', 'restless', 'agitated', 'moving slow']):
        risk_score += 1
        risk_factors.append("Psychomotor changes")
    
    # 9. Suicidal ideation (CRITICAL)
    if any(w in text_lower for w in ['suicide', 'kill myself', 'want to die', 'end my life']):
        risk_score += 5
        risk_factors.append("⚠️ Suicidal ideation - IMMEDIATE ATTENTION NEEDED")
    
    # Additional factors
    if request.sleep_hours < 5:
        risk_score += 2
        risk_factors.append(f"Severe sleep deprivation ({request.sleep_hours}h)")
    elif request.sleep_hours < 7:
        risk_score += 1
        risk_factors.append(f"Insufficient sleep ({request.sleep_hours}h)")
    
    if request.mood_score <= 3:
        risk_score += 3
        risk_factors.append(f"Very low self-reported mood ({request.mood_score}/10)")
    elif request.mood_score <= 5:
        risk_score += 1
        risk_factors.append(f"Low self-reported mood ({request.mood_score}/10)")
    
    # Face analysis impact
    if depression_face_indicators > 5:
        risk_score += 2
        risk_factors.append("Facial analysis shows significant depression indicators")
    
    # Depression indicator count from session
    indicator_count = len(session.get("depression_indicators", []))
    risk_score += indicator_count * 0.5
    
    # Determine severity
    if risk_score >= 15:
        severity = "Severe"
        confidence = 0.85
        urgency = "IMMEDIATE - Please seek professional help today"
    elif risk_score >= 10:
        severity = "Moderate to Severe"
        confidence = 0.80
        urgency = "Soon - Schedule an appointment with a mental health professional"
    elif risk_score >= 6:
        severity = "Moderate"
        confidence = 0.75
        urgency = "Consider consulting with a mental health professional"
    elif risk_score >= 3:
        severity = "Mild"
        confidence = 0.70
        urgency = "Monitor and consider talking to someone you trust"
    else:
        severity = "Minimal"
        confidence = 0.75
        urgency = "Continue healthy habits and self-care"
    
    # Generate personalized recommendations
    recommendations = []
    
    if "⚠️ Suicidal ideation" in str(risk_factors):
        recommendations.append("🆘 CRISIS: Call 988 (Suicide & Crisis Lifeline) or text HOME to 741741 immediately")
        recommendations.append("Go to your nearest emergency room or call 911 if in immediate danger")
    
    if request.sleep_hours < 7:
        recommendations.append("😴 Sleep: Establish a consistent sleep schedule. Aim for 7-9 hours nightly")
    
    if request.mood_score < 5:
        recommendations.append("📊 Mood Tracking: Keep a daily mood journal to identify patterns and triggers")
    
    if "Loss of interest" in str(risk_factors):
        recommendations.append("🎯 Activity Scheduling: Try scheduling one small enjoyable activity each day")
    
    if "Fatigue" in str(risk_factors):
        recommendations.append("🔋 Energy Management: Start with short walks or gentle exercise to boost energy")
    
    if severity in ["Moderate", "Moderate to Severe", "Severe"]:
        recommendations.append("👨‍⚕️ Professional Help: Schedule an appointment with a therapist, counselor, or psychiatrist")
        recommendations.append("💊 Medical Evaluation: Consider a check-up to rule out medical causes (thyroid, vitamin deficiencies)")
    
    recommendations.extend([
        "🤝 Social Support: Stay connected with trusted friends and family",
        "🧘 Stress Management: Try relaxation techniques like deep breathing or meditation",
        "📱 Crisis Resources: Save 988 (Lifeline) and 741741 (Crisis Text) in your phone"
    ])
    
    assessment = {
        "severity": severity,
        "risk_level": severity.split()[0] if severity != "Minimal" else "Low",
        "confidence": confidence,
        "risk_score": risk_score,
        "urgency": urgency,
        "phq9_approximation": min(risk_score, 27),  # PHQ-9 max is 27
        "sleep_hours": request.sleep_hours,
        "mood_score": request.mood_score,
        "messages_analyzed": len([m for m in session.get("messages", []) if m['user'] != 'SYSTEM']),
        "depression_indicators": session.get("depression_indicators", []),
        "indicator_count": indicator_count,
        "risk_factors": risk_factors,
        "face_summary": face_summary,
        "recommendations": recommendations,
        "user_name": session.get("user_name", "User")
    }
    
    print(f"\n📊 DEPRESSION SCREENING COMPLETE")
    print(f"   Severity: {severity}")
    print(f"   Risk Score: {risk_score}")
    print(f"   Indicators: {indicator_count}")
    print(f"   Risk Factors: {len(risk_factors)}")
    print(f"   Name: {session.get('user_name', 'Unknown')}")
    
    return {
        "assessment": assessment,
        "session_id": session_id,
        "completed_at": datetime.now().isoformat()
    }

@app.delete("/chat/{session_id}")
async def end_chat(session_id: str):
    """End screening session"""
    if session_id in chat_sessions:
        session_data = chat_sessions[session_id]
        
        # Save session log
        log_dir = "screening_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"screening_{session_id}.json")
        
        try:
            with open(log_file, 'w') as f:
                json.dump({
                    "session_id": session_id,
                    "user_name": session_data.get("user_name"),
                    "started_at": session_data.get("started_at"),
                    "ended_at": datetime.now().isoformat(),
                    "total_messages": len([m for m in session_data.get("messages", []) if m['user'] != 'SYSTEM']),
                    "depression_indicators": session_data.get("depression_indicators", []),
                    "messages": session_data.get("messages", [])
                }, f, indent=2, default=str)
        except:
            pass
        
        del chat_sessions[session_id]
    
    return {"status": "ended", "message": "Screening session completed"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": "MindGuard Depression Screening",
        "gemini_available": GEMINI_AVAILABLE,
        "gemini_model": GEMINI_MODEL,
        "face_detection": MODEL_TYPE,
        "active_sessions": len(chat_sessions),
        "prompt_version": "PHQ-9 Based Depression Screening v2.0"
    }

# ============================================
# STARTUP
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🧠 MINDGUARD - Depression Screening System")
    print("=" * 60)
    print(f"🤖 AI Model: {'Gemini ' + GEMINI_MODEL if GEMINI_AVAILABLE else 'Simulation Mode'}")
    print(f"📷 Face Detection: {MODEL_TYPE}")
    print(f"📋 Screening Protocol: PHQ-9 Based + Emotional Analysis")
    print(f"💾 Session Storage: In-memory + JSON logs")
    print("=" * 60)
    print("\nStarting server at http://0.0.0.0:8000")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)