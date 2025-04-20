import os
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client
import numpy as np
import json

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



# Utility functions
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
def store_chat(user_id, message, role, embedding):
    # Ensure message is a string (prevent saving embeddings by mistake)
    if not isinstance(message, str):
        message = str(message)  # fallback, but ideally should never happen
    supabase.table("user_chats").insert({
        "user_id": user_id,
        "message": message,
        "role": role,
        "timestamp": "now()",
        "embedding": embedding
    }).execute()
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
        raw_embedding = chat.get("embedding")
        if raw_embedding:
            # Parse stringified list if needed
            if isinstance(raw_embedding, str):
                try:
                    raw_embedding = json.loads(raw_embedding)
                except json.JSONDecodeError:
                    continue  # skip if it's invalid
            sim = cosine_similarity(current_embedding, raw_embedding)
            scored.append((sim, chat))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:top_k]]
def get_recent_chats(user_id):
    response = supabase.table("user_chats") \
        .select("*") \
        .order("timestamp", desc=True) \
        .limit(10) \
        .eq("user_id", user_id) \
        .execute()
    return response.data
def get_profile(user_id):
    existing = supabase.table("user_profiles") \
        .select("*") \
        .eq("user_id", user_id) \
        .limit(1) \
        .execute()
    if existing.data:
        profile = existing.data[0]
        return profile
    return {"background": [], "tech_stack": []}
def initialize_profile(user_id, domain, tech_input):
    
    existing = get_profile(user_id)
    if existing.get("tech_stack"):
        return {"message": "Profile already exists", "profile": existing}
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

        supabase.table("user_profiles").insert({
            "user_id": user_id,
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
        raise Exception("Profile parsing failed. Model said: " + response)
def chat_with_bot(user_id, message):
    user_embedding = genai.embed_content(
        model="models/embedding-001",
        content=message,
        task_type="retrieval_query"
    )["embedding"]
    
    relevant_chats = get_similar_chats(user_id, user_embedding, top_k=5)
    store_chat(user_id, message, "user", user_embedding)
    memory_snippets = "\n".join([f"{c['role']}: {c['message']}" for c in relevant_chats])
    
    context_prompt = (
        f"""Here are some relevant past conversations:\n{memory_snippets}\n\nUser said: "{message}"\nRespond appropriately."""
        if memory_snippets else message
    )   

    

    profile = get_profile(user_id)
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

        updated_stack = list(set(profile.get("tech_stack", [])) | new_stack)
        updated_bg = list(set(profile.get("background", [])) | new_bg)
        supabase.table("user_profiles").upsert({
            "user_id": user_id,
            "tech_stack": updated_stack,
            "background": updated_bg
        }, on_conflict="user_id").execute()
    except:
        updated_stack = profile.get("tech_stack", [])
        updated_bg = profile.get("background", [])
    
    reply = chat_session.send_message(context_prompt)
    
    assistant_embedding = genai.embed_content(
        model="models/embedding-001",
        content=reply.text,
        task_type="retrieval_query"
    )["embedding"]
    
    store_chat(user_id, reply.text, "assistant", assistant_embedding)
    
    
    
    return {
        "response": reply.text,
        "profile": {
            "Tech_Stack": updated_stack,
            "Background": updated_bg
        }
    
    }

        
        
if __name__ == "__main__":
    print("💼 Career Transition Coach Chatbot Setup\n")

    user_id = input("Enter your email/user ID: ").strip()

    # Try to fetch existing profile
    profile = get_profile(user_id)
    if profile.get("tech_stack") or profile.get("background"):
        print("✅ Found existing profile.")
        print("📁 Tech Stack:", profile.get("tech_stack"))
        print("📁 Background:", profile.get("background"))
    else:
        print("🆕 No profile found. Let's create one.")
        domain = input("Enter your domain (e.g., Healthcare, Finance, Education): ").strip()
        tech_input = input("Enter your current tech stack (comma-separated, e.g., Python, FastAPI): ").strip()
        result = initialize_profile(user_id, domain, tech_input)
        profile = result["profile"]
        print("✅ Profile created.")
        print("📁 Tech Stack:", profile["Tech_Stack"])
        print("📁 Background:", profile["Background"])
        
    system_instructions=f"""
    You are a career development chatbot assistant for people transitioning into tech.


    The user has a knowledge on:

    Tech Stack: {profile.get("tech_stack")}

    Background: {profile.get("background")}


    Your task:

    - Interpret the query **in the context of the user’s given skills and background**.

    - Give feedback on how the user knowledge will help for the task or how user can improve their knowledge in that particular field **with respect to the user's query and tech stack**

    - Give a focused, strategic answer tailored to that profile.

    - **Do not explain general advice unless it directly applies to the user's experience.**

    - Provide actionable steps, career path suggestions, or resources — **only if they are relevant to the user's query and tech stack**.

    - Maintain an encouraging and practical tone.

    - No preamble 

    - Keep the content brief so that user can read you message easilycl

    Example Query: What are the most in-demand skills for transitioning to a tech career?
    Possible Output:
    Based on your background in marketing and current stack (Python, JS, React, Node.js), you’re well-positioned to explore roles like Marketing Technologist, Data Analyst (marketing-focused), or Full-Stack Developer. To stand out, consider strengthening your skills in Git, REST APIs, and basic cloud deployment (AWS or Vercel). Projects that combine data insights and web dashboards can help bridge your marketing and dev experience — ideal for employers in digital or e-commerce spaces.
    ---
    """
    reply = chat_session.send_message(system_instructions)
    print("\n💬 Ready to chat with your coach!")
    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"exit", "quit"}:
            print("👋 Exiting chatbot. Have a great day!")
            break

        result = chat_with_bot(user_id, message)
        print("\n🤖 Bot:", result["response"])
        print("📈 Updated Profile:", result["profile"])


