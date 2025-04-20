import os
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client
import numpy as np
import json
import uuid
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

chat_session = model.start_chat()

# Utility functions
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


def get_recent_chats(user_id):
    """Get most recent chats for a user"""
    try:
        response = supabase.table("user_chats") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=True) \
            .limit(10) \
            .execute()
        
        return response.data or []
    except Exception as e:
        print(f"Error getting recent chats: {str(e)}")
        return []

def get_user_by_email(email):
    """Find a user by email address - using the direct email as user_id in this case"""
    try:
        # Since we're using email as user_id directly, just return the email as the ID
        return email
    except Exception as e:
        print(f"Error finding user by email: {str(e)}")
        return None

def get_profile(user_id):
    """Get user profile with tech stack and background"""
    try:
        response = supabase.table("user_profiles") \
            .select("*") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return {"user_id": user_id, "background": [], "tech_stack": []}
    except Exception as e:
        print(f"Error getting profile: {str(e)}")
        return {"user_id": user_id, "background": [], "tech_stack": []}

def initialize_profile(user_id, domain, tech_input):
    """Initialize or update a user profile with domain background and tech stack"""
    try:
        # Check if profile already exists
        existing = get_profile(user_id)
        
        # If tech stack exists and is not empty, profile already initialized
        if existing.get("tech_stack") and len(existing.get("tech_stack")) > 0:
            return {"message": "Profile already exists", "profile": existing}
        
        # Generate structured data from user input
        prompt = f"""
        The user provided:
        - Domain: {domain}
        - Tech stack: {tech_input}

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
            
            # Insert or update profile in database
            if "id" in existing:
                # Profile exists, update it
                result = supabase.table("user_profiles") \
                    .update({
                        "background": Background,
                        "tech_stack": Tech_Stack
                    }) \
                    .eq("user_id", user_id) \
                    .execute()
            else:
                # Profile doesn't exist, create it
                result = supabase.table("user_profiles") \
                    .insert({
                        "user_id": user_id,
                        "background": Background,
                        "tech_stack": Tech_Stack,
                        "created_at": datetime.now().isoformat()
                    }) \
                    .execute()
            
            return {
                "message": "Profile created",
                "profile": {
                    "Background": Background,
                    "Tech_Stack": Tech_Stack
                }
            }
        except Exception as e:
            print(f"Profile parsing error: {str(e)}")
            raise Exception("Profile parsing failed. Model said: " + response)
    except Exception as e:
        print(f"Error initializing profile: {str(e)}")
        raise

def chat_with_bot(user_id, message):
    """Process user message and get AI response with context awareness"""
    try:
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
        
        # Create context-aware prompt
        context_prompt = (
            f"""Here are some relevant past conversations:\n{memory_snippets}\n\nUser said: "{message}"\nRespond appropriately."""
            if memory_snippets else message
        )
        
        # Get user profile
        profile = get_profile(user_id)
        
        # Generate enrichment prompt to update user profile
        enrichment_prompt = f"""
        Based on this user query: "{message}"
        Techstack refers to any programming languages, frameworks, or tools the user is familiar with.

        If it suggests new relevant tech stack items, return them as:
        Tech_Stack = ["python", "fastapi", "etc"]

        Background refers to the user's domain knowledge or experience.
        If it suggests relevant domains like software developer, data scientist, Microsoft intern, etc., return them as:
        Background = ["software developer", "data scientist", "etc"]

        If neither, return:
        Tech_Stack = []
        Background = []
        """
        
        # Check for profile updates
        enrichment_response = chat_session.send_message(enrichment_prompt).text
        
        try:
            local_vars = {}
            exec(enrichment_response, {}, local_vars)
            new_stack = set(local_vars.get("Tech_Stack", []))
            new_bg = set(local_vars.get("Background", []))
            
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
        
        # Get AI response
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
                "Tech_Stack": updated_stack,
                "Background": updated_bg
            }
        }
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return {
            "response": f"I'm having trouble processing your request. Please try again later. (Error: {str(e)})",
            "profile": profile if 'profile' in locals() else {"Tech_Stack": [], "Background": []}
        }

def chatbot_init(user_id):
    """Initialize chatbot with system instructions based on user profile"""
    try:
        print("💼 Career Transition Coach Chatbot Setup\n")
        
        # Get user profile
        profile = get_profile(user_id)
        
        # Format system instructions with user profile information
        system_instructions = f"""
        You are a career development chatbot assistant for people transitioning into tech.

        The user has knowledge on:

        Tech Stack: {profile.get("tech_stack", [])}

        Background: {profile.get("background", [])}

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
        
        print("✅ Chatbot initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        return False

def main():
    """Main function to run the chatbot"""
    print("🤖 Welcome to Career Transition Coach Chatbot!")
    
    try:
        # Get user email as ID
        email = input("Enter your email: ").strip().lower()
        
        if not email or '@' not in email:
            print("❌ Please enter a valid email address.")
            return
            
        # Use email as user_id
        user_id = email
        
        # Check if user profile exists
        profile = get_profile(user_id)
        
        if profile and profile.get("tech_stack") and len(profile.get("tech_stack")) > 0:
            print("✅ Welcome back! Found your profile.")
            print("📁 Tech Stack:", profile.get("tech_stack", []))
            print("📁 Background:", profile.get("background", []))
        else:
            # Create new profile for the user
            print(f"🆕 Creating new profile for {email}...")
            domain = input("Enter your domain (e.g., Healthcare, Finance, Education): ").strip()
            tech_input = input("Enter your current tech stack (comma-separated, e.g., Python, FastAPI): ").strip()
            
            result = initialize_profile(user_id, domain, tech_input)
            profile = result["profile"]
            
            print("✅ Profile created.")
            print("📁 Tech Stack:", profile["Tech_Stack"])
            print("📁 Background:", profile["Background"])
        
        # Initialize chatbot
        chatbot_init(user_id)
        
        print("\n💬 Ready to chat with your coach! Type 'exit' or 'quit' to end the conversation.")
        
        # Main chat loop
        while True:
            message = input("\nYou: ").strip()
            
            if message.lower() in {"exit", "quit", "bye", "goodbye"}:
                print("👋 Exiting chatbot. Have a great day!")
                break
            
            # Process message and get response
            result = chat_with_bot(user_id, message)
            
            print("\n🤖 Bot:", result["response"])
            print("\n📈 Profile Info:")
            print("   Tech Stack:", result["profile"]["Tech_Stack"])
            print("   Background:", result["profile"]["Background"])
            
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot terminated. Have a great day!")
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)}")
        print("The chatbot encountered an error and must exit.")

if __name__ == "__main__":
    main()