from flask import Flask, request, jsonify
from queue import Queue
from threading import Thread
import time
import requests
import os
from datetime import datetime, timedelta, timezone

# --- LANGCHAIN IMPORTS (LEGACY COMPATIBLE) ---
# We use try/except to handle whatever version the server gives us
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback for very old versions
    from langchain.chat_models import ChatOpenAI 

from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- THE CRITICAL FIX ---
# Instead of the new 'create_tool_calling_agent', we use the older 'initialize_agent'
from langchain.agents import initialize_agent, AgentType

app = Flask(__name__)
request_queue = Queue()

# --- CONSTANTS ---
PROFILE_API = "https://fresh-linkedin-scraper-api.p.rapidapi.com/api/v1/user/profile"
POSTS_API = "https://fresh-linkedin-scraper-api.p.rapidapi.com/api/v1/user/posts"

# --- HELPER FUNCTIONS ---
def make_api_call(url, headers, params, max_retries=3):
    """Helper to handle rate limits and retries inside the tool"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 429:
                time.sleep(5)
                continue
            if response.status_code == 200:
                return response.json()
        except Exception:
            time.sleep(1)
    return None

def is_recent(timestamp_str, days=30):
    try:
        if not timestamp_str: return False
        if timestamp_str.endswith('Z'): timestamp_str = timestamp_str.replace('Z', '+00:00')
        ts = datetime.fromisoformat(timestamp_str)
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        return ts >= (datetime.now(timezone.utc) - timedelta(days=days))
    except:
        return False

# --- THE AGENT TOOL ---
@tool
def scrape_linkedin_posts(username: str, rapid_api_key: str, days: int = 30) -> str:
    """
    Use this tool to fetch recent LinkedIn posts for a specific username.
    It requires the LinkedIn username (e.g., 'williamhgates') and the RapidAPI Key.
    Returns a raw string of the text content of their recent posts.
    """
    print(f"🛠️ TOOL: Agent is scraping LinkedIn for {username}...")
    
    headers = {
        'x-rapidapi-key': rapid_api_key,
        'x-rapidapi-host': 'fresh-linkedin-scraper-api.p.rapidapi.com'
    }

    # 1. Get Profile to find URN
    profile_data = make_api_call(PROFILE_API, headers, {'username': username})
    if not profile_data or 'data' not in profile_data:
        return "Error: Could not find user profile or URN."
    
    urn = profile_data['data'].get('urn')
    
    if not urn:
        return "Error: Profile found but URN is missing."

    # 2. Get Posts using URN
    posts_data = make_api_call(POSTS_API, headers, {'urn': urn, 'page': 1})
    if not posts_data or 'data' not in posts_data:
        return "Error: Could not fetch posts."

    raw_posts = posts_data['data']
    cleaned_posts = []
    
    for p in raw_posts:
        if is_recent(p.get('created_at'), days) and p.get('text'):
            cleaned_posts.append(f"Date: {p.get('created_at')[:10]}\nContent: {p.get('text')}\n---")

    if not cleaned_posts:
        return "User exists, but has no posts within the specified date range."

    return "\n".join(cleaned_posts)


# --- AGENT FACTORY ---
def run_agent_job(job_data):
    """
    Sets up the agent dynamically using the provided OpenAI API Key.
    """
    try:
        # 1. Setup LLM
        llm = ChatOpenAI(
            api_key=job_data['openai_api_key'], 
            model="gpt-4o", 
            temperature=0
        )

        # 2. Setup Tools
        tools = [scrape_linkedin_posts]

        # 3. Construct Agent (THE OLDER, SAFE WAY)
        # This uses the OPENAI_FUNCTIONS agent type which works on older versions
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS, 
            verbose=True,
            handle_parsing_errors=True
        )

        # 4. Invoke Agent
        input_text = (
            f"Please fetch the LinkedIn posts for username '{job_data['username']}' "
            f"using the RapidAPI key '{job_data['rapid_api_key']}'. "
            f"Look for posts from the last {job_data.get('posted_max_days_ago', 30)} days. "
            "Once you have the data, ignore the raw JSON format and write a professional 3-bullet point summary "
            "of their recent activity, tone, and main topics."
        )

        # Older agents use .run or .invoke depending on version, .invoke is usually safe now
        # but .run is the safest for legacy.
        try:
            response_text = agent_executor.run(input_text)
        except:
            response = agent_executor.invoke({"input": input_text})
            response_text = response.get("output")

        return {
            "success": True,
            "username": job_data['username'],
            "summary": response_text
        }

    except Exception as e:
        print(f"❌ Agent Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# --- WORKER & FLASK ---
def sequential_worker():
    while True:
        if not request_queue.empty():
            job = request_queue.get()
            webhook_url = job.get('webhook_url')
            
            print(f"🤖 Starting Agent Job for {job.get('username')}")
            
            result = run_agent_job(job)
            
            if webhook_url:
                try:
                    print(f"📤 Sending webhook to {webhook_url}")
                    requests.post(webhook_url, json=result, timeout=10)
                except Exception as e:
                    print(f"❌ Webhook failed: {e}")
            
            time.sleep(1)
        else:
            time.sleep(0.5)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    
    # VALIDATION
    required_fields = ['username', 'rapid_api_key', 'openai_api_key']
    missing = [f for f in required_fields if not data.get(f)]
    
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400
    
    request_queue.put({
        'username': data.get('username'),
        'rapid_api_key': data.get('rapid_api_key'),
        'openai_api_key': data.get('openai_api_key'),
        'webhook_url': data.get('webhook_url'),
        'posted_max_days_ago': data.get('posted_max_days_ago', 30)
    })
    
    return jsonify({'status': 'queued', 'queue_size': request_queue.qsize()}), 202

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    worker = Thread(target=sequential_worker, daemon=True)
    worker.start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
