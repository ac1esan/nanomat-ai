"""Entry point for Hugging Face Spaces (Gradio SDK). Locally: python screen_bandgap.py --app"""
from screen_bandgap import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
