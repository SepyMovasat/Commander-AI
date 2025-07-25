"""
Input module: Mouse and keyboard control.
"""
"""
Input module: Mouse and keyboard control.
Gracefully degrades if pyautogui cannot be used (e.g. headless env).
"""

try:
    import pyautogui  # Requires a GUI environment
    _pyautogui_error = None
except Exception as e:  # Import or display errors
    pyautogui = None
    _pyautogui_error = e

def move_mouse(x, y):
    if not pyautogui:
        raise RuntimeError(f"pyautogui unavailable: {_pyautogui_error}")
    pyautogui.moveTo(x, y)

def click():
    if not pyautogui:
        raise RuntimeError(f"pyautogui unavailable: {_pyautogui_error}")
    pyautogui.click()

def type_text(text):
    if not pyautogui:
        raise RuntimeError(f"pyautogui unavailable: {_pyautogui_error}")
    pyautogui.write(text)
