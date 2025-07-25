"""
Command module: Run shell commands and capture output.
"""
import subprocess

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            # Include error message if command failed
            output = f"Error: {result.stderr.strip()}"
        return output or "Command executed successfully (no output)"
    except Exception as e:
        return f"Error executing command: {str(e)}"
