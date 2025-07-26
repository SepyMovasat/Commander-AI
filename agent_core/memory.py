"""
Memory module: Stores context, previous actions, and user preferences.
"""
import os
import json


MEMORY_FILE = os.path.join(os.path.dirname(__file__), '../cache/memory.json')
NOTEPAD_FILE = os.path.join(os.path.dirname(__file__), '../cache/notepad.json')
RAG_FILE = os.path.join(os.path.dirname(__file__), '../cache/rag.json')
CHAT_HISTORY_FILE = os.path.join(os.path.dirname(__file__), '../cache/chat_history.json')

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f)

# Notepad memory: persistent notes
def add_to_notepad(note: str):
    notes = []
    if os.path.exists(NOTEPAD_FILE):
        with open(NOTEPAD_FILE, 'r') as f:
            notes = json.load(f)
    notes.append(note)
    with open(NOTEPAD_FILE, 'w') as f:
        json.dump(notes, f)

def get_notepad():
    if os.path.exists(NOTEPAD_FILE):
        with open(NOTEPAD_FILE, 'r') as f:
            return json.load(f)
    return []

# RAG memory: efficient retrieval-augmented memory
def add_to_rag(text: str):
    # For simplicity, store as list of dicts with text and embedding (embedding is a placeholder)
    rag = []
    if os.path.exists(RAG_FILE):
        with open(RAG_FILE, 'r') as f:
            rag = json.load(f)
    rag.append({"text": text})
    with open(RAG_FILE, 'w') as f:
        json.dump(rag, f)

def rag_query(query: str):
    # Simple RAG: return the most relevant note by string match (can be replaced with embedding search)
    if not os.path.exists(RAG_FILE):
        return "[No RAG memory yet]"
    with open(RAG_FILE, 'r') as f:
        rag = json.load(f)
    # Find the note with the most word overlap
    query_words = set(query.lower().split())
    best = None
    best_score = 0
    for item in rag:
        item_words = set(item["text"].lower().split())
        score = len(query_words & item_words)
        if score > best_score:
            best = item["text"]
            best_score = score
    return best if best else "[No relevant memory found]"

# Chat history persistence
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_chat_history(history):
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
