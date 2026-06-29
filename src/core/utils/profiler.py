import os
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config

LOG_FILE = os.path.join(config.STORAGE_DIR, "performance_logs.jsonl")

def log_timing(step_name, duration_seconds, metadata=None):
    """
    Logs execution duration for a specific pipeline step.
    Prints a formatted log to the console and appends a structured entry to the JSONL log file.
    """
    if metadata is None:
        metadata = {}
        
    duration_ms = round(duration_seconds * 1000, 2)
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": step_name,
        "duration_ms": duration_ms,
        **metadata
    }
    
    # Print clean console log
    print(f"[PROFILER] Step '{step_name}' completed in {duration_ms}ms")
    
    try:
        # Append to performance logs file
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[PROFILER WARNING] Failed to write timing log: {e}")
