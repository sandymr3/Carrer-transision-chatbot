from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client
import numpy as np
import json
import uuid
import uvicorn
from datetime import datetime

# Load environment variables
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
api_key = os.getenv("GEMAI")

# Validate essential environment variables
if not all([supabase_url, supabase_key, api_key]):
    missing = []
    if not supabase_url: missing.append("SUPABASE_URL")
    if not supabase_key: missing.append("SUPABASE_KEY")
    if not api_key: missing.append("GEMAI")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# Initialize Supabase
try:
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    raise ConnectionError(f"Failed to connect to Supabase: {str(e)}")

# Configure Gemini
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 50000,
        "response_mime_type": "text/plain"
    },
)

# Store chat sessions for users
user_chat_sessions = {}

# Define Pydantic models for request/response validation
class UserBase(BaseModel):
    email: EmailStr

class ProfileInit(BaseModel):
    email: EmailStr
    domain: str
    tech_stack: str

class ChatMessage(BaseModel):
    email: EmailStr
    message: str

class ProfileResponse(BaseModel):
    tech_stack: List[str]
    background: List[str]

class ChatResponse(BaseModel):
    response: str
    profile: Dict[str, List[str]]

# Utility functions from original code
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    try:
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    except Exception as e:
        print(f"Error in cosine similarity calculation: {str(e)}")
        return 0.0

def store_chat(user_id, message, role, embedding):
    """Store chat messages in the database with embeddings"""
    try:
        # Ensure message is a string
        if not isinstance(message, str):
            message = str(message)
        
        # Convert embedding to a proper format if needed
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # Insert chat message into database
        result = supabase.table("user_chats").insert({
            "user_id": user_id,
            "message": message,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding
        }).execute()
        
        return bool(result.data)
    except Exception as e:
        print(f"Error storing chat: {str(e)}")
        return False

def get_similar_chats(user_id, current_embedding, top_k=5):
    """Retrieve similar chats using vector similarity"""
    try:
        # Get recent chats for the user
        response = supabase.table("user_chats") \
            .select("message, role, embedding") \
            .eq("user_id", user_id) \
            .execute()
        
        chats = response.data or []
        scored = []
        
        for chat in chats:
            raw_embedding = chat.get("embedding")
            if raw_embedding:
                try:
                    # Ensure it's a list, not a string
                    if isinstance(raw_embedding, str):
                        raw_embedding = json.loads(raw_embedding)
                    
                    # Calculate similarity
                    sim = cosine_similarity(current_embedding, raw_embedding)
                    scored.append((sim, chat))
                except Exception as e:
                    print(f"Skipping invalid embedding: {e}")
                    continue
        
        # Sort by similarity score (descending)
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [c for _, c in scored[:top_k]]
    except Exception as e:
        print(f"Error getting similar chats: {str(e)}")
        return []

def get_profile(user_id):
    """Get user profile with tech stack and background"""
    try:
        response = supabase.table("user_profiles") \
            .select("*") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]  # This will now contain the 'id'
        return None  # <- key change
    except Exception as e:
        print(f"Error getting profile: {str(e)}")
        return None


def initialize_profile(user_id, domain, tech_input):
    """Initialize or update a user profile with domain background and tech stack"""
    try:
        # Check if profile already exists
        existing = get_profile(user_id)
        
        # If tech stack exists and is not empty, profile already initialized
        if existing and existing.get("tech_stack") and len(existing.get("tech_stack")) > 0:
            return {"message": "Profile already exists", "profile": existing}
        
        # Generate structured data from user input
        chat_session = model.start_chat()
        prompt = f"""
        The user provided:
        - Domain: {domain}
        - Tech stack: {tech_input}

        Return ONLY this format with no explanation or code formatting:

        background = ["..."]
        tech_stack = ["..."]
        """

        response = chat_session.send_message(prompt).text
        
        try:
            local_vars = {}
            exec(response, {}, local_vars)
            background = list(set(local_vars.get("background", [])))
            tech_stack = list(set(local_vars.get("tech_stack", [])))

            if existing:
                # Profile exists but was empty — update it
                result = supabase.table("user_profiles") \
                    .update({
                        "background": background,
                        "tech_stack": tech_stack
                    }) \
                    .eq("user_id", user_id) \
                    .execute()
            else:
                # Profile doesn't exist — insert new one
                result = supabase.table("user_profiles") \
                    .insert({
                        "user_id": user_id,
                        "background": background,
                        "tech_stack": tech_stack,
                        "created_at": datetime.now().isoformat()
                    }) \
                    .execute()
            
            return {
                "message": "Profile created",
                "profile": {
                    "background": background,
                    "tech_stack": tech_stack
                }
            }
        except Exception as e:
            print(f"Profile parsing error: {str(e)}")
            raise Exception("Profile parsing failed. Model said: " + response)
    except Exception as e:
        print(f"Error initializing profile: {str(e)}")
        raise

def get_chat_history(user_id, limit=50):
    """Fetch chat history for a user"""
    try:
        response = supabase.table("user_chats") \
            .select("message, role, timestamp") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=False) \
            .limit(limit) \
            .execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching chat history: {str(e)}")
        return []
    
def delete_chat_history(user_id):
    """Delete all chat messages for a user"""
    try:
        result = supabase.table("user_chats") \
            .delete() \
            .eq("user_id", user_id) \
            .execute()
        return True
    except Exception as e:
        print(f"Error deleting chat history: {str(e)}")
        return False

def chatbot_init(user_id):
    """Initialize chatbot with system instructions based on user profile"""
    try:
        # Get user profile
        profile = get_profile(user_id)
        
        # Create new chat session for the user
        chat_session = model.start_chat()
        
        # Format system instructions with user profile information
        system_instructions = f"""
        You are a career development chatbot assistant for people transitioning into tech.

        The user has knowledge on:

        Tech Stack: {profile.get("tech_stack", [])}

        background: {profile.get("background", [])}

        Your task:

        - Interpret the query **in the context of the user's given skills and background**.

        - Give feedback on how the user knowledge will help for the task or how user can improve their knowledge in that particular field **with respect to the user's query and tech stack**

        - Give a focused, strategic answer tailored to that profile.

        - **Do not explain general advice unless it directly applies to the user's experience.**

        - Provide actionable steps, career path suggestions, or resources — **only if they are relevant to the user's query and tech stack**.

        - Maintain an encouraging and practical tone.

        - No preamble 

        - Keep the content brief so that user can read your message easily

        Example Query: What are the most in-demand skills for transitioning to a tech career?
        Possible Output:
        Based on your background in marketing and current stack (Python, JS, React, Node.js), you're well-positioned to explore roles like Marketing Technologist, Data Analyst (marketing-focused), or Full-Stack Developer. To stand out, consider strengthening your skills in Git, REST APIs, and basic cloud deployment (AWS or Vercel). Projects that combine data insights and web dashboards can help bridge your marketing and dev experience — ideal for employers in digital or e-commerce spaces.
        ---
        """
        
        # Initialize chat session with system instructions
        chat_session.send_message(system_instructions)
        
        # Store the chat session for this user
        user_chat_sessions[user_id] = chat_session
        
        return True
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        return False

def chat_with_bot(user_id, message):
    """Process user message and get AI response with context awareness"""
    try:
        # Initialize chat session if not exists
        if user_id not in user_chat_sessions:
            chatbot_init(user_id)
            
        chat_session = user_chat_sessions[user_id]
        
        # Generate embedding for user message
        embed_response = genai.embed_content(
            model="models/embedding-001",
            content=message,
            task_type="retrieval_query"
        )
        user_embedding = embed_response["embedding"]
        
        # Get similar past conversations for context
        relevant_chats = get_similar_chats(user_id, user_embedding, top_k=5)
        
        # Store the user message
        store_chat(user_id, message, "user", user_embedding)
        
        # Build memory context from relevant chats
        memory_snippets = "\n".join([f"{c['role']}: {c['message']}" for c in relevant_chats])
        
        
        
        # Get user profile
        profile = get_profile(user_id)
        
        # Generate enrichment prompt to update user profile
        enrichment_prompt = f"""
        Based on this user query: "{message}"
        Techstack refers to any programming languages, frameworks, or tools the user is familiar with.

        If it suggests new relevant tech stack items, return them as:
        tech_stack = ["python", "fastapi", "etc"]

        background refers to the user's domain knowledge or experience.
        If it suggests relevant domains like software developer, data scientist, Microsoft intern, etc., return them as:
        background = ["software developer", "data scientist", "etc"]

        If neither, return:
        tech_stack = []
        background = []
        """
        
        # Check for profile updates
        enrichment_response = chat_session.send_message(enrichment_prompt).text
        
        try:
            local_vars = {}
            exec(enrichment_response, {}, local_vars)
            new_stack = set(local_vars.get("tech_stack", []))
            new_bg = set(local_vars.get("background", []))
            
            # Get current profile tech stack and background
            current_tech_stack = profile.get("tech_stack", [])
            current_background = profile.get("background", [])
            
            # Ensure they're lists
            if not isinstance(current_tech_stack, list):
                current_tech_stack = []
            if not isinstance(current_background, list):
                current_background = []
                
            # Update profile with new information
            updated_stack = list(set(current_tech_stack) | new_stack)
            updated_bg = list(set(current_background) | new_bg)
            
            # Update profile in database if changes detected
            if new_stack or new_bg:
                supabase.table("user_profiles") \
                    .update({
                        "tech_stack": updated_stack,
                        "background": updated_bg
                    }) \
                    .eq("user_id", user_id) \
                    .execute()
                
        except Exception as e:
            print(f"Error updating profile: {str(e)}")
            updated_stack = profile.get("tech_stack", [])
            updated_bg = profile.get("background", [])
        
        
        
        ms="No relevant past conversations found."
        
        if memory_snippets:
            ms= memory_snippets
            
            
        # Create context-aware prompt
        
        context_prompt = (
            f"""Here are some relevant past conversations:n {ms} User said: "{message}" 
            the user is has knowledge on {updated_stack} and has a bockground domain knowledge on feilds like{updated_bg}
            Respond for the user with the same tone and style as the previous conversations,
            but make sure to give a focused,strategic answer tailored to that profile.
            
            You must respond in a way that is relevant to the user's query and tech stack.
            
            
            
            *You shall repond as if a typical chatbot is responding to the user*
            Make your response in a way that it is easy to read and understand for the user.
            "Respond as a helpful chatbot. Use natural, friendly language without
            special characters like asterisks or markdown formatting. Separate paragraphs with line breaks
            for clarity."
            
            
            
            An example of a user query:
            
            What are the most in-demand skills for transitioning to a tech career?
            
            
            
            An example of a possible output:
            
            Based on your background in software development and your interest in data science,
            
            I would recommend focusing on Python and R for data analysis, as well as SQL for database management.
            
            Additionally, consider learning about machine learning frameworks like TensorFlow or 
            PyTorch to enhance your skill set.
            
            Projects that involve data visualization and predictive modeling
            can be particularly beneficial for your career transition.
            
            
            
            """
            
        )
        # Get AI response to actual user message
        reply = chat_session.send_message(context_prompt)
        
        
        # Generate embedding for assistant response
        assistant_embedding = genai.embed_content(
            model="models/embedding-001",
            content=reply.text,
            task_type="retrieval_query"
        )["embedding"]
        
        # Store assistant response
        store_chat(user_id, reply.text, "assistant", assistant_embedding)
        
        return {
            "response": reply.text,
            "profile": {
                "tech_stack": updated_stack,
                "background": updated_bg
            }
        }
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return {
            "response": f"I'm having trouble processing your request. Please try again later. (Error: {str(e)})",
            "profile": profile if 'profile' in locals() else {"tech_stack": [], "background": []}
        }

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Career Transition Chatbot API"}

@app.post("/check-profile", response_model=Dict[str, Any])
async def check_user_profile(user: UserBase):
    """Check if a user profile exists"""
    user_id = user.email.lower()
    profile = get_profile(user_id)

    if profile and profile.get("tech_stack") and len(profile.get("tech_stack")) > 0:
        # Profile exists
        return {
            "exists": True,
            "profile": {
                "tech_stack": profile.get("tech_stack", []),
                "background": profile.get("background", [])
            }
        }

    else:
        # Profile doesn't exist
        return {
            "exists": False
        }

@app.post("/initialize-profile", response_model=Dict[str, Any])
async def create_user_profile(profile_data: ProfileInit):
    """Initialize a user profile with domain and tech stack"""
    user_id = profile_data.email.lower()
    
    # Check if profile already exists
    profile = get_profile(user_id)
    if profile and profile.get("tech_stack") and len(profile.get("tech_stack")) > 0:
        return {
            "message": "Profile already exists",
            "profile": {
                "tech_stack": profile.get("tech_stack", []),
                "background": profile.get("background", [])
            }
        }
    
    # Initialize profile
    result = initialize_profile(user_id, profile_data.domain, profile_data.tech_stack)
    
    # Initialize chatbot for this user
    chatbot_init(user_id)
    
    return {
        "message": "Profile created successfully",
        "profile": result["profile"]
    }

@app.post("/chat", response_model=ChatResponse)
async def process_chat(chat_data: ChatMessage):
    """Process a chat message and return AI response"""
    user_id = chat_data.email.lower()
    
    # Check if profile exists
    profile = get_profile(user_id)
    if not profile or not profile.get("tech_stack") or len(profile.get("tech_stack")) == 0:
        raise HTTPException(
            status_code=404,
            detail="User profile not found. Please initialize your profile first."
        )
    
    # Process chat message
    result = chat_with_bot(user_id, chat_data.message)
    
    return {
        "response": result["response"],
        "profile": result["profile"]
    }
    
@app.get("/chat-history", response_model=List[Dict[str, Any]])
async def chat_history(email: EmailStr):
    """Return the chat history for a user"""
    user_id = email.lower()
    chats = get_chat_history(user_id)
    return chats

@app.delete("/chat/{email}", response_model=Dict[str, bool])
async def delete_user_chat_history(email: EmailStr):
    """Delete chat history for a user"""
    user_id = email.lower()
    success = delete_chat_history(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete chat history")
    return {"success": True}


# Run with: uvicorn main:app --reload
"""if __name__ == "__main__":

    uvicorn.run(app)"""