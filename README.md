# Commander AI

A modular, hybrid AI agent for desktop automation, question answering, and system interaction.

## Features
- Uses local LLMs (Ollama, llama.cpp, etc.) for most tasks
- Optionally uses cloud APIs (OpenAI, Gemini, Claude) for complex queries
- Reads and edits files, runs commands, controls mouse/keyboard, reads screen, searches web
- Extensible plugin system

## Setup
1. **Python 3.9+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Local LLM:**
   - Recommended: [Ollama](https://ollama.com/) (download and run a 3B model, e.g. `ollama run llama3:3b`)
   - Or use [llama.cpp](https://github.com/ggerganov/llama.cpp) with a compatible 3B model
4. **API keys (optional):**
   - OpenAI: https://platform.openai.com/account/api-keys
   - Gemini: https://ai.google.dev/gemini-api/docs/api-key
   - Claude: https://claude.ai/
   - Bing/DuckDuckGo Web Search API (optional)
5. **OCR:**
   - Install Tesseract: `sudo apt install tesseract-ocr`

## Run
```bash
python main.py
```

## Directory Structure
- `agent_core/` - Orchestration, task manager
- `models/` - LLM wrappers
- `modules/` - System interaction
- `ui/` - CLI/GUI
- `plugins/` - Extensions
- `config/` - Config, API keys
- `cache/` - Caching

---

**See code comments for more details.**
