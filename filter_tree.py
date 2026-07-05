import os
import re

# Sensitive keys to replace
REPLACEMENTS = {
    # Cerebras API Key
    "YOUR_CEREBRAS_API_KEY_HERE": "YOUR_CEREBRAS_API_KEY_HERE",
    # Gemini API Key
    "YOUR_GEMINI_API_KEY_HERE": "YOUR_GEMINI_API_KEY_HERE",
    # ChatGPT Bearer Token
    "YOUR_CHATGPT_BEARER_HERE": "YOUR_CHATGPT_BEARER_HERE",
    # The 5 sharded keys
    "YOUR_GEMINI_KEY_1": "YOUR_GEMINI_KEY_1",
    "YOUR_GEMINI_KEY_2": "YOUR_GEMINI_KEY_2",
    "YOUR_GEMINI_KEY_3": "YOUR_GEMINI_KEY_3",
    "YOUR_GEMINI_KEY_4": "YOUR_GEMINI_KEY_4",
    "YOUR_GEMINI_KEY_5": "YOUR_GEMINI_KEY_5"
}

def process_file(filepath):
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        
        modified = False
        for key, replacement in REPLACEMENTS.items():
            key_bytes = key.encode("utf-8")
            if key_bytes in content:
                content = content.replace(key_bytes, replacement.encode("utf-8"))
                modified = True
        
        if modified:
            with open(filepath, "wb") as f:
                f.write(content)
            print(f"Processed and cleaned: {filepath}")
    except Exception as e:
        pass

def main():
    for root, dirs, files in os.walk("."):
        if ".git" in root.split(os.sep):
            continue
        for file in files:
            filepath = os.path.join(root, file)
            # Scan text files and scripts
            if file.endswith((".py", ".env", ".json", ".txt", ".md", ".sh", ".yml", ".yaml")):
                process_file(filepath)

if __name__ == "__main__":
    main()
