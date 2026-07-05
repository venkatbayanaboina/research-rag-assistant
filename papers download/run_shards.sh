#!/bin/bash
# run_shards.sh
# Runs 5 sharded API dataset generators in parallel.

# Define API Keys
KEY1="YOUR_GEMINI_KEY_1"
KEY2="YOUR_GEMINI_KEY_2"
KEY3="YOUR_GEMINI_KEY_3"
KEY4="YOUR_GEMINI_KEY_4"
KEY5="YOUR_GEMINI_KEY_5"

echo "⚡ Starting 5-shard parallel QA dataset generation..."
echo "📂 Logging output to shard_1.log, shard_2.log, etc."

# Launch each shard in the background
GEMINI_API_KEY="$KEY1" python3 generate_dataset_api.py --shard 1 --num-shards 5 > shard_1.log 2>&1 &
PID1=$!
GEMINI_API_KEY="$KEY2" python3 generate_dataset_api.py --shard 2 --num-shards 5 > shard_2.log 2>&1 &
PID2=$!
GEMINI_API_KEY="$KEY3" python3 generate_dataset_api.py --shard 3 --num-shards 5 > shard_3.log 2>&1 &
PID3=$!
GEMINI_API_KEY="$KEY4" python3 generate_dataset_api.py --shard 4 --num-shards 5 > shard_4.log 2>&1 &
PID4=$!
GEMINI_API_KEY="$KEY5" python3 generate_dataset_api.py --shard 5 --num-shards 5 > shard_5.log 2>&1 &
PID5=$!

echo "🚀 Background processes launched. PIDs: $PID1, $PID2, $PID3, $PID4, $PID5"
echo "⏳ Waiting for all shards to complete (~4.5 minutes)..."

# Wait for all background tasks to finish
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo "📂 All processes complete. Merging shards..."

# Merge all generated shards into gold_qa_dataset.json and delete temp files
python3 -c "
import json, glob, os
merged = {}
if os.path.exists('gold_qa_dataset.json'):
    try:
        with open('gold_qa_dataset.json') as f:
            merged = json.load(f)
    except:
        pass

for fpath in glob.glob('gold_qa_dataset_shard_*.json'):
    try:
        with open(fpath) as f:
            data = json.load(f)
            merged.update(data)
        os.remove(fpath)
    except Exception as e:
        print(f'Error merging {fpath}: {e}')

with open('gold_qa_dataset.json', 'w') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
"

echo "🎉 All shards completed successfully! Output merged in gold_qa_dataset.json."
