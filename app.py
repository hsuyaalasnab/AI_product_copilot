#Adding functionality of Find (Establishing RAG)

import streamlit as st
import google.generativeai as genai
import json
import numpy as np

# ==========================================
# 1. SETUP & API KEY
# ==========================================
# Read the key securely from Streamlit Cloud's secrets vault
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-3.5-flash')
embedding_model = 'models/gemini-embedding-001'

# ==========================================
# 2. THE SCREEN DICTIONARY
# ==========================================
SCREEN_DICTIONARY = {
    "home_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/home_Screen.png?raw=true",
    "file_menu_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/file_menu_screen.png?raw=true",
    "data_selected_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/data_selected_screen.png?raw=true",
    "insert_menu_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/insert_menu_screen.png?raw=true",
    "pivot_menu_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/pivot_menu_screen.png?raw=true",
    "pivot_implemented_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/pivot_implemented_screen.png?raw=true",
    "formula_typing_gif": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/formula_typing_gif.gif?raw=true",
    "formula_applied_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/formula_applied_screen.png?raw=true",
    "find_dropdown": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/find_dropdown.png?raw=true",
    "find_dialog_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/find_dialog_screen.png?raw=true",
    "find_results_screen": "https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/find_results_Screen.png?raw=true"
    
}

# ==========================================
# 3. THE KNOWLEDGE BASE
# ==========================================
KNOWLEDGE_BASE = [
    {
        "intent_description": "How to save a file as a new document or save as.",
        "ui_map": """
            Screen 1 (home_screen):
            - File Menu: x: '1%', y: '8%'
            Screen 2 (file_menu_screen):
            - Save As Button: x: '3%', y: '32%'
        """
    },
    {
        "intent_description": "How to export data to a CSV file.",
        "ui_map": """
            Screen 1 (home_screen):
            - File Menu: x: '1%', y: '8%'
            Screen 2 (file_menu_screen):
            - Export CSV Button: x: '3%', y: '40%'
        """
    },
    {
        "intent_description": "How to create a pivot table from existing data.",
        "ui_map": """
            Screen 1 (data_selected_screen):
            - Select Data : x: '20%', y: '75%'
            - Go to Insert Menu : x: '10%', y:'8%'
            Screen 2 (insert_menu_screen):
            - Pivot Table Button: x: '3%', y: '12%'
            Screen 3 (pivot_menu_screen):
            - Existing Worksheet Radio Button: x: '10%', y: '50%'
            - OK Button: x: '32%', y: '68%'
            Screen 5 (pivot_implemented_screen):
            - View Completed Pivot Table: x: '1%', y: '1%'
        """
    },
    {
        "intent_description": "How to apply a formula on a cell.",
        "ui_map": """
            Screen 1 (home_screen):
            - Target Cell: x: '30%', y: '45%'
            Screen 2 (formula_typing_gif):
            - Wait for formula to type: x: '30%', y: '40%', wait_time: 8000
            Screen 3 (formula_applied_screen):
            - View Applied Formula: x: '30%', y: '45%'
        """
    },
    {
        "intent_description": "How to search or find a word, text, or value within the active worksheet or current sheet.",
        "ui_map": """
            Screen 1 (home_screen):
            - Find & Select Button: x: '88%', y: '15%'
            Screen 2 (find_dropdown):
            - Select Find from Dropdown: x: '88%', y: '23%'
            Screen 3 (find_dialog_screen):
            - Search Input Box: x: '45%', y: '47%'
            - Find Next Button: x: '65%', y: '63%'
            Screen 4 (find_results_screen):
            - Highlight Found Match: x: '16%', y: '64%'
        """
    }
]

# ==========================================
# 4. OPTIMIZED CACHING (RATE LIMIT FIX)
# ==========================================
def get_embedding(text):
    result = genai.embed_content(model=embedding_model, content=text, task_type="retrieval_document")
    return result['embedding']

# @st.cache_data forces Streamlit to run this only ONCE on startup.
@st.cache_data
def precompute_database_embeddings():
    embeddings = []
    for item in KNOWLEDGE_BASE:
        embeddings.append(get_embedding(item["intent_description"]))
    return embeddings

# Load embeddings into memory immediately
kb_embeddings = precompute_database_embeddings()

# ==========================================
# 5. STRICT SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """
You are an expert UI navigation assistant. Your job is to guide users step-by-step through a web application.
Your ONLY source of truth is the [RETRIEVED CONTEXT] provided below. 

GUARDRAILS:
1. If the user's question cannot be answered using the [RETRIEVED CONTEXT], reply that you can only assist with documented application features and return an empty "actions" list. Do not hallucinate steps.
2. You must output ONLY a valid, raw JSON object. Do not include markdown blocks (like ```json), comments, or any conversational text outside the JSON structure.
3. AMBIGUITY HANDLING: If the user asks to "find text" or "search" without specifying whether it is for the current sheet or the entire workbook, follow the active worksheet workflow, but mention in your 'reply' field that they can also search the entire workbook by opening Options.

INSTRUCTIONS FOR ACTIONS:
Translate the retrieved workflow into a sequential list of actions. For EVERY step in the workflow, provide a brief instruction, the exact x and y coordinates, the exact screen_id, and the wait_time (default to 0 unless a wait_time is explicitly provided in the context).

JSON OUTPUT FORMAT:
{
  "reply": "Here is how you do it. First, click the cell. Then type the formula and hit enter.",
  "actions": [
    {
      "text": "Click the target cell.",
      "x": "30%",
      "y": "45%",
      "screen_id": "home_screen",
      "wait_time": 0
    },
    {
      "text": "Wait for the formula to finish typing.",
      "x": "30%",
      "y": "40%",
      "screen_id": "formula_typing_gif",
      "wait_time": 8000
    },
    {
      "text": "The formula is applied and the result is displayed.",
      "x": "30%",
      "y": "45%",
      "screen_id": "formula_applied_screen",
      "wait_time": 0
    }
  ]
}

[RETRIEVED CONTEXT]:
"""

# ==========================================
# 6. STREAMLIT APP LAYOUT & LOGIC
# ==========================================
st.set_page_config(layout="wide")
st.title("AI Product Copilot: Prototype")

col1, col2 = st.columns([3, 2])

# Initialize session state for chat history and animation trigger
if "messages" not in st.session_state:
    st.session_state.messages = []
if "animation_data" not in st.session_state:
    st.session_state.animation_data = "[]"

with col2:
    st.subheader("Copilot Chat")
    
    # Create a fixed-height, scrollable container (matches the 450px height of our app view)
    chat_container = st.container(height=450)
    
    # 1. Render existing chat history INSIDE the container
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 2. Handle new user input (Input box stays pinned below the container)
    if prompt := st.chat_input("Try asking: 'How do I apply a formula?'"):
        
        # Save and display user message in the container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generate and display AI response in the container
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        # --- RAG RETRIEVAL ---
                        user_embedding = get_embedding(prompt)
                        best_match = None
                        highest_similarity = 0
                        
                        for i, item in enumerate(KNOWLEDGE_BASE):
                            item_embedding = kb_embeddings[i]
                            similarity = np.dot(user_embedding, item_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding))
                            
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match = item
                        
                        # --- GUARDRAIL CHECK ---
                        if highest_similarity < 0.65:
                            retrieved_context = "NO RELEVANT CONTEXT FOUND."
                        else:
                            retrieved_context = best_match["ui_map"]

                        # --- TEXT GENERATION ---
                        final_prompt = SYSTEM_PROMPT + retrieved_context + "\n\nUser Question: " + prompt
                        response = model.generate_content(final_prompt)
                        
                        clean_json_str = response.text.replace("```json", "").replace("```", "").strip()
                        ai_payload = json.loads(clean_json_str)
                        
                        # Display AI text reply
                        st.markdown(ai_payload["reply"])
                        
                        # Save AI reply to history
                        st.session_state.messages.append({"role": "assistant", "content": ai_payload["reply"]})
                        
                        # --- URL MAPPING ---
                        actions = ai_payload.get("actions", [])
                        for action in actions:
                            screen_id = action.get("screen_id")
                            action["image"] = SCREEN_DICTIONARY.get(screen_id, "") 
                        
                        st.session_state.animation_data = json.dumps(actions)

                    except Exception as e:
                        st.error(f"Error processing request. Details: {e}")

with col1:
    st.subheader("Application Interface")
    
    # ==========================================
    # 7. FRONTEND UI & ANIMATION LOGIC
    # ==========================================
    html_code = f"""
    <div id="app-screen" style="
        position: relative; width: 100%; aspect-ratio: 850 / 458;
        background-color: #ecf0f1; 
        background-image: url('https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/home_Screen.png?raw=true');
        background-size: 100% 100%; background-position: center;
        border-radius: 8px; border: 2px solid #bdc3c7; overflow: hidden;
        transition: background-image 0.3s ease-in-out;
    ">
        <div id="virtual-cursor" style="
            position: absolute; width: 24px; height: 24px;
            background: rgba(231, 76, 60, 0.8); border: 2px solid #e74c3c; border-radius: 50%;
            pointer-events: none; z-index: 100;
            transition: left 1s ease-in-out, top 1s ease-in-out, transform 0.2s;
            left: -50px; top: -50px;
        "></div>
    </div>

    <script>
        const steps = {st.session_state.animation_data};
        const screen = document.getElementById('app-screen');
        const cursor = document.getElementById('virtual-cursor');
        const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

        async function runAnimation() {{
            // Handle off-topic guardrail block
            if (!steps || steps.length === 0) {{
                screen.style.backgroundImage = "url('https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/default_image.png?raw=true')";
                return;
            }}

            // Loop through the sequential steps
            for (const step of steps) {{
                // Change background if it differs from the current one
                if (step.image) screen.style.backgroundImage = `url('${{step.image}}')`;
                await sleep(600);
                
                // Glide cursor
                cursor.style.left = step.x;
                cursor.style.top = step.y;
                await sleep(1000);
                
                // Click effect
                cursor.style.transform = 'scale(0.5)';
                await sleep(200);
                cursor.style.transform = 'scale(1)';
                await sleep(800);
                
                if (step.wait_time && step.wait_time > 0) await sleep(step.wait_time);
                
            
            }}
            
            
            // Cleanup and reset
            await sleep(1000);
            cursor.style.left = '-50px'; 
            cursor.style.top = '-50px';
            screen.style.backgroundImage = "url('[https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/defult_image.png?raw=true](https://github.com/hsuyaalasnab/AI_product_copilot/blob/main/images/default_image.png?raw=true)')";
        }}

        runAnimation();
    </script>
    """
    
    st.components.v1.html(html_code, height=500)
