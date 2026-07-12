#!/bin/bash
# run_parallel_generation.sh
# Runs 5 sharded LLM generation processes in parallel.

# Define API Keys
KEY1="YOUR_GEMINI_KEY_1"
KEY2="YOUR_GEMINI_KEY_2"
KEY3="YOUR_GEMINI_KEY_3"
KEY4="YOUR_GEMINI_KEY_4"
KEY5="YOUR_GEMINI_KEY_5"

echo "⚡ Starting 5-shard parallel LLM generation..."
echo "📂 Logging output to gen_shard_1.log, gen_shard_2.log, etc."

# Launch each shard in the background
GEMINI_API_KEY="$KEY1" ../rag/bin/python3 -u run_llm_generation.py --shard 1 --num-shards 5 > gen_shard_1.log 2>&1 &
PID1=$!
GEMINI_API_KEY="$KEY2" ../rag/bin/python3 -u run_llm_generation.py --shard 2 --num-shards 5 > gen_shard_2.log 2>&1 &
PID2=$!
GEMINI_API_KEY="$KEY3" ../rag/bin/python3 -u run_llm_generation.py --shard 3 --num-shards 5 > gen_shard_3.log 2>&1 &
PID3=$!
GEMINI_API_KEY="$KEY4" ../rag/bin/python3 -u run_llm_generation.py --shard 4 --num-shards 5 > gen_shard_4.log 2>&1 &
PID4=$!
GEMINI_API_KEY="$KEY5" ../rag/bin/python3 -u run_llm_generation.py --shard 5 --num-shards 5 > gen_shard_5.log 2>&1 &
PID5=$!

echo "🚀 Background processes launched. PIDs: $PID1, $PID2, $PID3, $PID4, $PID5"
echo "⏳ Waiting for all shards to complete..."

# Wait for all background tasks to finish
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo "📂 All processes complete. Merging generation shards..."

# Merge all generated shards into rag_retrieved_answers.json and delete temp files
../rag/bin/python3 -c "
import json, glob, os
merged = {}
if os.path.exists('rag_retrieved_answers.json'):
    try:
        with open('rag_retrieved_answers.json') as f:
            merged = json.load(f)
    except:
        pass

for fpath in glob.glob('rag_retrieved_answers_shard_*.json'):
    try:
        with open(fpath) as f:
            data = json.load(f)
            # Update individual key values (answers) inside the dict
            for qid, item in data.items():
                if qid in merged:
                    merged[qid]['rag_answer'] = item.get('rag_answer', '')
                else:
                    merged[qid] = item
        os.remove(fpath)
    except Exception as e:
        print(f'Error merging {fpath}: {e}')

with open('rag_retrieved_answers.json', 'w') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
"

echo "🎉 All LLM answer shards completed successfully and merged into rag_retrieved_answers.json!"
