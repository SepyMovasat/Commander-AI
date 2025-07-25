"""
Input module: Mouse and keyboard control.
"""
import pyautogui

def move_mouse(x, y):
    pyautogui.moveTo(x, y)

def click():
    pyautogui.click()

def type_text(text):
    pyautogui.write(text)
