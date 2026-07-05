import os

# The 5 hardcoded keys to replace
REPLACEMENTS = {
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
            # Scan all code, scripts, shell, env and text files
            if file.endswith((".py", ".env", ".json", ".txt", ".md", ".sh", ".yml", ".yaml")):
                process_file(filepath)

if __name__ == "__main__":
    main()
