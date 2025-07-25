"""
LLMManager: Handles local and API LLMs, prompt engineering, and RAG.
"""
import os
import requests


import yaml
import json
import subprocess

class LLMManager:
    """
    Handles both local (Ollama) and OpenAI API LLMs for planning and tool-use.
    """
    def __init__(self):
        # Load config
        with open("config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        self.use_api = cfg.get("USE_API", False)
        self.api_key = cfg.get("OPENAI_API_KEY", None)
        self.local_llm = cfg.get("LOCAL_LLM", "ollama")

    def plan(self, request: str, chat_history=None) -> dict:
        """
        Use LLM to parse user request into a plan (tool use, steps, etc.)
        Returns a dict: {"tool": ..., "args": ...}
        
        Args:
            request: The user's request
            chat_history: List of previous interactions in the format [{"role": role, "content": content}, ...]
        """
        # Add UI message about tool execution
        def add_tool_description(plan: dict) -> dict:
            if not plan or 'tool' not in plan:
                return plan
                
            tool = plan['tool']
            args = plan.get('args', {})
            
            if tool == 'read_file':
                path = args.get('path', '')
                desc = f"📖 Reading file: {path}"
            elif tool == 'write_file' or tool == 'append_file':
                path = args.get('path', '')
                desc = f"✍️ {'Writing to' if tool == 'write_file' else 'Appending to'} file: {path}"
            elif tool == 'run_command':
                cmd = args.get('cmd', '')
                if len(cmd) > 50:
                    cmd = cmd[:47] + "..."
                desc = f"🔧 Running command: {cmd}"
            elif tool == 'search_web':
                query = args.get('query', '')
                if len(query) > 50:
                    query = query[:47] + "..."
                desc = f"🔍 Searching web for: {query}"
            elif tool == 'screen_ocr':
                desc = "👀 Capturing screen text"
            elif tool == 'move_mouse':
                x, y = args.get('x', 0), args.get('y', 0)
                desc = f"🖱️ Moving mouse to ({x}, {y})"
            elif tool == 'click':
                desc = "🖱️ Clicking mouse"
            elif tool == 'type_text':
                text = args.get('text', '')
                if len(text) > 30:
                    text = text[:27] + "..."
                desc = f"⌨️ Typing: {text}"
            elif tool == 'memory_notepad_add':
                note = args.get('note', '')
                if len(note) > 50:
                    note = note[:47] + "..."
                desc = f"📝 Adding note: {note}"
            elif tool == 'memory_rag_query':
                query = args.get('query', '')
                if len(query) > 50:
                    query = query[:47] + "..."
                desc = f"🧠 Querying memory: {query}"
            else:
                return plan

            if 'ui' not in plan:
                plan['ui'] = {}
            plan['ui']['description'] = desc
            return plan

        # Memory triggers
        if request.lower().startswith("remember that") or request.lower().startswith("remember this"):
            plan = {"tool": "memory_notepad_add", "args": {"note": request}}
            return add_tool_description(plan)

        # Direct answer shortcut (if question is simple)
        if self._is_simple_question(request):
            answer = self.answer_question(request)
            return {"tool": "none", "args": {}, "message": answer}
            
        # Try local LLM first, fallback to API if needed
        plan = self._plan_with_local(request, chat_history=chat_history)
        if plan.get("fallback_to_api") and self.use_api and self.api_key:
            plan = self._plan_with_api(request, chat_history)
            
        return add_tool_description(plan)

    def _is_simple_question(self, request: str) -> bool:
        # Heuristic: if it looks like a factual or short question, answer directly
        q = request.strip().lower()
        return q.endswith('?') and not any(x in q for x in ["file", "screen", "mouse", "type", "command", "search", "web", "run", "move", "click", "read", "write", "append"])

    def _plan_with_local(self, request: str, chat_history=None) -> dict:
        prompt = self._get_prompt(request, local=True, chat_history=chat_history)
        try:
            # Use Popen for streaming output
            process = subprocess.Popen(
                ["ollama", "run", "llama3.2:3b"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Send prompt
            process.stdin.write(prompt)
            process.stdin.close()
            
            # Stream output
            output = ""
            for line in process.stdout:
                print(line, end='', flush=True)  # Print streaming response
                output += line
                
            process.stdout.close()
            process.wait(timeout=60)
            
            plan = self._parse_plan_from_output(output.strip())
            return plan
        except Exception as e:
            return {"tool": "none", "args": {}, "error": str(e), "fallback_to_api": True}

    def _plan_with_api(self, request: str, chat_history=None) -> dict:
        import openai
        openai.api_key = self.api_key
        prompt = self._get_prompt(request, local=False, chat_history=chat_history)

        # Build the messages array with chat history
        messages = [{"role": "system", "content": self._system_prompt(api=True)}]
        
        # Add chat history if provided
        if chat_history:
            for entry in chat_history:
                role = entry.get('role', '')
                content = entry.get('content', '')
                # Convert roles to OpenAI format
                if role == 'user':
                    messages.append({"role": "user", "content": content})
                elif role == 'assistant' or role.startswith('llm_'):
                    messages.append({"role": "assistant", "content": content})
                    
        # Add the current request
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Use streaming for real-time output
            output = ""
            for chunk in openai.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=messages,
                temperature=0.2,
                max_tokens=512,
                stream=True  # Enable streaming
            ):
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end='', flush=True)  # Print streaming response
                    output += content
                    
            plan = self._parse_plan_from_output(output.strip())
            return plan
        except Exception as e:
            return {"tool": "none", "args": {}, "error": str(e)}

    def _get_prompt(self, request: str, local: bool, chat_history=None) -> str:
        """
        Returns a detailed prompt for the LLM, describing available tools and expected output format.
        Includes chat history for context if available.
        """
        tool_desc = self._tool_description()
        
        # Format chat history if available
        history_context = ""
        if chat_history:
            history_context = "Previous conversation:\n"
            for entry in chat_history:
                role = "User" if entry.get('role') == 'user' else "Assistant"
                content = entry.get('content', '').strip()
                if content:
                    history_context += f"{role}: {content}\n"
            history_context += "\n"  # separate from the new request

        return f"""
You are an AI agent operating in a REAL Linux environment with REAL consequences. This is not a simulation.
CRITICAL: Never invent or imagine file paths, outputs, or responses - you are working with actual files and systems.

{tool_desc}
{history_context}Focus only on the NEW user message below. Use conversation history and tool outputs just for reference, handling one message at a time.

USER REQUEST: {request}

TASK EXECUTION FLOW:
1. Begin every response with a short message from you.
2. After your text, output EXACTLY ONE tool command as JSON.
3. Continue executing one tool at a time until the task is fully complete.
4. When finished, provide a final summary message and output {{"tool": "none", "args": {{}}}}.

KEY POINTS:
1. Don't stop until the task is actually finished
2. Treat every tool result or user reply as a new message in the conversation (but not as a task)
3. Each response = ONE tool command as JSON
4. Wait for real output before the next step
5. Keep messages brief but informative
6. Share progress updates in your text as needed

Examples of good progress updates:
✓ "I'll help you find large files in the system"
✓ "Found potential large files, now checking their exact sizes"
✓ "Complete! Found 3 files over 1GB: [actual file list]"

❌ Too frequent:
"Searching directory..."
"Checking next directory..."
"Moving to next folder..."
"""

    def _system_prompt(self, api: bool) -> str:
        return (
            "You are operating in a REAL Linux environment - not a simulation. "
            "Follow these CRITICAL rules exactly:\n\n"
            "CORE RULES:\n"
            "1. Start every response with a short message from you.\n"
            "2. After your message, output EXACTLY ONE tool command as JSON.\n"
            "3. Never combine multiple actions - do one step at a time.\n"
            "4. Do not invent tool names or argument names.\n"
            "5. Provide progress updates in your text as needed.\n"
            "6. Finish with a summary message and the JSON {{\"tool\": \"none\", \"args\": {{}}}} when done.\n"
            "7. Use the 'none' tool only when no further action is needed.\n"
            "8. Treat each user message as a NEW instruction; use chat history only as context.\n"
            "9. Tool outputs appear as their own messages; don't confuse them with the user's request.\n\n"
            "Example JSON:\n"
            '{"tool": "tool_name", "args": {"arg_name": "value"}}\n\n'
            "Always gather necessary information before acting."
        )

    def _tool_description(self) -> str:
        return (
            "AVAILABLE TOOLS:\n\n"
            "Input/Output:\n"
            '{"tool": "screen_ocr", "args": {}} - Capture screen text\n'
            '{"tool": "move_mouse", "args": {"x": 123, "y": 456}} - Move mouse\n'
            '{"tool": "click", "args": {}} - Click mouse\n'
            '{"tool": "type_text", "args": {"text": "text"}} - Type text\n\n'
            "Files:\n"
            '{"tool": "read_file", "args": {"path": "/path"}} - Read file\n'
            '{"tool": "write_file", "args": {"path": "/path", "content": "text"}} - Write file\n'
            '{"tool": "append_file", "args": {"path": "/path", "content": "text"}} - Append to file\n\n'
            "System & Web:\n"
            '{"tool": "search_web", "args": {"query": "terms"}} - Web search\n'
            '{"tool": "run_command", "args": {"cmd": "command"}} - Shell command\n\n'
            "Memory & Communication:\n"
            '{"tool": "memory_notepad_add", "args": {"note": "text"}} - Add note\n'
            '{"tool": "memory_rag_query", "args": {"query": "text"}} - Query memory\n'
            '{"tool": "inquiry", "args": {"text": "question"}} - Ask user\n'
            '{"tool": "none", "args": {}} - No action needed\n'
        )

    def answer_question(self, question: str) -> str:
        """Answer a question directly using the best available LLM, routing easy questions to Ollama and harder ones to OpenAI if available."""
        # Heuristic: if question is very short/simple, use Ollama; else use OpenAI if available
        q = question.strip().lower()
        easy_greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening", "good night"]
        is_easy = (
            q in easy_greetings or
            (len(q.split()) <= 8 and not any(x in q for x in ["explain", "why", "how", "summarize", "analyze", "compare", "difference", "write", "code", "generate", "complex", "difficult"]))
        )
        if is_easy:
            try:
                prompt = f"Answer the following question concisely and factually.\nQuestion: {question}"
                result = subprocess.run([
                    "ollama", "run", "llama3.2:3b"
                ], input=prompt, capture_output=True, text=True, timeout=60)
                return result.stdout.strip()
            except Exception as e:
                return f"[Local LLM error: {e}]"
        # Otherwise, use OpenAI if available
        if self.use_api and self.api_key:
            try:
                import openai
                openai.api_key = self.api_key
                response = openai.chat.completions.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": question}],
                    temperature=0.2,
                    max_tokens=512
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"[API error: {e}]"
        # Fallback to Ollama if OpenAI is not available
        try:
            prompt = f"Answer the following question concisely and factually.\nQuestion: {question}"
            result = subprocess.run([
                "ollama", "run", "llama3.2:3b"
            ], input=prompt, capture_output=True, text=True, timeout=60)
            return result.stdout.strip()
        except Exception as e:
            return f"[Local LLM error: {e}]"

    def _output_format(self) -> str:
        # Note: This method is kept for compatibility but no longer used
        # All format information is now included in _tool_description
        return ""

    def _parse_plan_from_output(self, output: str) -> dict:
        """Extract a JSON plan and any accompanying text."""
        try:
            start = output.find('{')
            end = output.rfind('}') + 1
            if start != -1 and end != -1:
                plan = json.loads(output[start:end])
                text = (output[:start] + output[end:]).strip()
                if text:
                    plan['message'] = text
                return plan
        except Exception:
            pass
        return {"tool": "none", "args": {}, "message": output}
