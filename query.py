from main import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import ast
from typing import List, Optional
import google.generativeai as genai
from supabase import create_client, Client
import numpy as np

# Load environment variables
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
api_key = os.getenv("GEMAI")
assert api_key, "Gemini API key not found in .env"

# Initialize Supabase
supabase: Client = create_client(supabase_url, supabase_key)

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 10000,
        "response_mime_type": "text/plain"
    },
)
chat_session = model.start_chat()

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Request schemas
class ProfileInput(BaseModel):
    user_id: str
    domain: Optional[str] = ""
    tech_input: Optional[str] = ""

class UserMessage(BaseModel):
    user_id: str
    message: str

# Utils


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8) 

def get_similar_chats(user_id, current_embedding, top_k=5):
    response = supabase.table("user_chats") \
        .select("message, role, embedding") \
        .eq("user_id", user_id) \
        .order("timestamp", desc=True) \
        .limit(50) \
        .execute()
    
    chats = response.data or []
    scored = []

    for chat in chats:
        if chat.get("embedding"):
            sim = cosine_similarity(current_embedding, chat["embedding"])
            scored.append((sim, chat))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_chats = [c for _, c in scored[:top_k]]
    return top_chats


def store_chat(user_id, message, role, embedding):
    supabase.table("user_chats").insert({
        "user_id": user_id,
        "message": message,
        "role": role,
        "timestamp": "now()",
        "embedding":embedding
    }).execute()

def get_recent_chats(user_id):
    response = supabase.table("user_chats") \
        .select("*") \
        .order("timestamp", desc=True) \
        .limit(10) \
        .eq("user_id", user_id) \
        .execute()
    return response.data





@app.post("/profile")
def initialize_profile(profile: ProfileInit):
    # Step 1: Check if profile already exists
    existing = supabase.table("user_profiles") \
        .select("*") \
        .eq("user_id", profile.user_id) \
        .limit(1) \
        .execute()
    
    if existing.data:
        existing_profile = existing.data[0]
        return {
            "message": "Profile already exists",
            "profile": {
                "Background": existing_profile.get("background", []),
                "Tech_Stack": existing_profile.get("tech_stack", [])
            }
        }

    # Step 2: If not exists, process the input and create new profile
    prompt = f"""
    The user provided:
    - Domain: {profile.domain}
    - Tech stack: {profile.tech_input}

    Extract and return the domain background and tech stack as two separate lists in the format:
    Background = [...]
    Tech_Stack = [...]
    """
    response = chat_session.send_message(prompt).text

    try:
        local_vars = {}
        exec(response, {}, local_vars)
        Background = list(set(local_vars.get("Background", [])))
        Tech_Stack = list(set(local_vars.get("Tech_Stack", [])))

        supabase.table("user_profiles").insert({
            "user_id": profile.user_id,
            "background": Background,
            "tech_stack": Tech_Stack
        }).execute()

        return {
            "message": "Profile created",
            "profile": {
                "Background": Background,
                "Tech_Stack": Tech_Stack
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Profile parsing failed. Model said: " + response)



@app.post("/chat")
def chat_with_bot(user_msg: UserMessage):
    user_id = user_msg.user_id
    message = user_msg.message

    # 1. Embed the user message
    user_embedding = genai.embed_content(
        model="models/embedding-001",
        content=message,
        task_type="retrieval_query"
    )["embedding"]

    # 2. Store the user message
    store_chat(user_id, message, "user", user_embedding)

    # 3. Get recent similar chats for semantic memory
    relevant_chats = get_similar_chats(user_id, user_embedding, top_k=5)
    memory_snippets = "\n".join([f"{c['role']}: {c['message']}" for c in relevant_chats])

    # 4. Construct context for Gemini
    if memory_snippets:
        context_prompt = f"""Here are some relevant past conversations:\n{memory_snippets}\n\nUser said: "{message}"\nRespond appropriately."""
    else:
        context_prompt = message

    # 5. Get Gemini reply
    reply = chat_session.send_message(context_prompt)

    # 6. Embed and store assistant reply
    assistant_embedding = genai.embed_content(
        model="models/embedding-001",
        content=reply.text,
        task_type="retrieval_query"
    )["embedding"]
    store_chat(user_id, reply.text, "assistant", assistant_embedding)

    # 7. Fetch existing user profile
    profile_data = supabase.table("user_profiles").select("*").eq("user_id", user_id).limit(1).execute()
    existing_profile = profile_data.data[0] if profile_data.data else {"tech_stack": [], "background": []}

    # 8. Enrich profile if needed
    enrichment_prompt = f"""
    Based on this user query: "{message}"

    If it suggests new relevant tech stack items, return them as:
    Tech_Stack = [ ... ]

    If it suggests new background elements, return them as:
    Background = [ ... ]

    If neither, return:
    Tech_Stack = []
    Background = []
    """
    enrichment_response = chat_session.send_message(enrichment_prompt).text

    try:
        local_vars = {}
        exec(enrichment_response, {}, local_vars)
        new_stack = set(local_vars.get("Tech_Stack", []))
        new_bg = set(local_vars.get("Background", []))

        updated_stack = list(set(existing_profile["tech_stack"]) | new_stack)
        updated_bg = list(set(existing_profile["background"]) | new_bg)

        supabase.table("user_profiles").upsert({
            "user_id": user_id,
            "tech_stack": updated_stack,
            "background": updated_bg
        }, on_conflict="user_id").execute()
    except:
        updated_stack = existing_profile["tech_stack"]
        updated_bg = existing_profile["background"] 

    # 9. Return assistant reply + updated profile
    return {
        "response": reply.text,
        "profile": {
            "Tech_Stack": updated_stack,
            "Background": updated_bg
        }
    }

@app.get("/chats/{user_id}")
def get_chats(user_id: str):
    return {"recent_chats": get_recent_chats(user_id)}
