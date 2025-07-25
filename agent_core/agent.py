"""
Agent core: Orchestrates tasks, manages memory, tool selection, and LLM interaction.
"""
from models.llm import LLMManager
from modules import screen, input, file, web, command


from agent_core import memory
from modules import screen, input as mod_input, file as mod_file, web, command

class Agent:
    def __init__(self):
        self.llm = LLMManager()
        self.memory = memory.load_memory()

    def handle_request(self, request: str) -> str:
        """Main entry for user requests. Implements agentic multi-step loop and saves chat history."""
        chat_history = []
        try:
            plan = self.llm.plan(request)
            chat_history.append({"role": "user", "content": request})
            chat_history.append({"role": "llm_plan", "content": str(plan)})
            result = self.execute_plan(plan, request)
            chat_history.append({"role": "tool", "content": str(result)})
            # Agentic multi-step loop: keep reasoning until LLM says done
            max_steps = 10
            steps = 0
            while self._should_continue(plan, result) and steps < max_steps:
                steps += 1
                followup_prompt = self._agentic_followup_prompt(request, plan, result)
                followup_plan = self._robust_parse_plan(self.llm._plan_with_api(followup_prompt))
                chat_history.append({"role": "llm_followup_plan", "content": str(followup_plan)})
                followup_result = self.execute_plan(followup_plan, request)
                chat_history.append({"role": "tool_followup", "content": str(followup_result)})
                plan, result = followup_plan, followup_result
        finally:
            # Save chat history even if error
            self.save_chat_history(chat_history)
            # Optionally update memory
            self.memory['last_request'] = request
            self.memory['last_plan'] = plan if 'plan' in locals() else None
            memory.save_memory(self.memory)
        # Always return a string: just return the result (direct_answer, inquiry, etc.)
        if not chat_history or not result:
            return 'No response from AI.'
        # If the plan/tool is 'none' return any final message
        if plan.get('tool') == 'none':
            args = plan.get('args', {})
            if 'text' in args:
                return args['text']
            if 'message' in plan:
                return plan['message']
        return str(result)

    def _robust_parse_plan(self, plan):
        """If plan is not a dict or is a string, try to parse it as JSON."""
        import ast, json
        if isinstance(plan, dict):
            return plan
        if isinstance(plan, str):
            try:
                # Try JSON first
                return json.loads(plan.replace("'", '"'))
            except Exception:
                try:
                    # Try Python dict literal
                    return ast.literal_eval(plan)
                except Exception:
                    return {"tool": "none", "args": {}, "message": plan}
        return {"tool": "none", "args": {}, "message": str(plan)}

    def _should_continue(self, plan, result):
        # If the LLM says it's done, or the plan is 'none', stop. Otherwise, continue.
        if not plan or plan.get('tool') == 'none':
            return False
        # If the result contains a clear done/completion signal, stop.
        done_signals = ['done', 'complete', 'finished', 'no further action', 'task accomplished']
        if any(sig in str(result).lower() for sig in done_signals):
            return False
        return True

    def _agentic_followup_prompt(self, user_request, last_plan, last_result):
        # Compose a prompt for the LLM to reason about the next step
        return (
            f"You are an autonomous AI agent. The user asked: '{user_request}'.\n"
            f"You just executed the tool '{last_plan.get('tool')}' with result: {last_result}\n"
            "Treat this tool result as a new message in the conversation."
            " If the overall task is not yet complete, plan the next step as a single JSON object (tool+args). "
            "When the task is fully done, provide your final summary and then return a JSON with tool:'none' and empty args. "
            "Remember: output exactly one JSON object and nothing else."
        )

    def save_chat_history(self, chat_history):
        import os, json, datetime
        chat_dir = os.path.join(os.path.dirname(__file__), '../cache/chats')
        os.makedirs(chat_dir, exist_ok=True)
        session_file = os.path.join(chat_dir, f'session_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(session_file, 'w') as f:
            json.dump(chat_history, f, indent=2)

    def execute_plan(self, plan, request=None):
        tool = plan.get('tool')
        args = plan.get('args', {})
        try:
            if tool == 'screen_ocr':
                screen.capture_screen()
                return "Captured the screen and performed OCR."
            elif tool == 'move_mouse':
                x, y = args.get('x'), args.get('y')
                if x is not None and y is not None:
                    mod_input.move_mouse(x, y)
                    return f"Moved mouse to ({x}, {y})."
                return "Missing x or y argument."
            elif tool == 'click':
                mod_input.click()
                return "Clicked mouse."
            elif tool == 'type_text':
                text = args.get('text', '')
                mod_input.type_text(text)
                return f"Typed: {text}"
            elif tool == 'read_file':
                path = args.get('path')
                if path:
                    content = mod_file.read_file(path)
                    return content
                return "Missing file path."
            elif tool == 'write_file':
                path, content = args.get('path'), args.get('content')
                if path is not None and content is not None:
                    mod_file.write_file(path, content)
                    return f"Wrote to {path}."
                return "Missing path or content."
            elif tool == 'append_file':
                path, content = args.get('path'), args.get('content')
                if path is not None and content is not None:
                    mod_file.append_file(path, content)
                    return f"Appended to {path}."
                return "Missing path or content."
            elif tool == 'search_web':
                query = args.get('query')
                if query:
                    web.search_web(query)
                    return f"Searched the web for: {query}"
                return "Missing query."
            elif tool == 'run_command':
                cmd = args.get('cmd')
                if cmd:
                    output = command.run_command(cmd)
                    return output
                return "Missing command."
            elif tool == 'memory_notepad_add':
                note = args.get('note')
                if note:
                    memory.add_to_notepad(note)
                    return "Added to notepad memory."
                return "Missing note."
            elif tool == 'memory_rag_query':
                query = args.get('query')
                if query:
                    memory.rag_query(query)
                    return f"Queried memory for: {query}"
                return "Missing query."
            elif tool == 'direct_answer':
                question = args.get('question')
                if question:
                    return self.llm.answer_question(question)
                return "Missing question."
            elif tool == 'inquiry':
                inquiry = args.get('question') or args.get('inquiry') or args.get('text')
                if not inquiry:
                    return 'The AI is requesting clarification.'
                # Return a special signal that this is an inquiry that needs user input
                return {"__type": "inquiry", "text": inquiry}
            elif tool == 'none':
                return ''
            else:
                return f"Unknown tool: {tool}"
        except Exception as e:
            return f"Error in tool execution: {e}"
