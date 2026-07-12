#!/bin/bash
# run_parallel_web_generation.sh
# Runs 3 parallel web automation processes on Gemini sharing the query load.
# Pre-merge check: Save and merge progress from any aborted/killed shard runs
echo "📂 Checking for existing progress in shard files to merge..."
../rag/bin/python3 -c "
import json, glob, os
merged = {}
if os.path.exists('rag_retrieved_answers.json'):
    try:
        with open('rag_retrieved_answers.json') as f:
            merged = json.load(f)
    except Exception as e:
        print('Error reading main database:', e)

shard_files = glob.glob('rag_retrieved_answers_shard_*.json')
if shard_files:
    print(f'Found {len(shard_files)} existing shard files. Merging completed answers...')
    merged_count = 0
    for fpath in shard_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
                for qid, item in data.items():
                    if qid in merged and item.get('rag_answer', '').strip() and not item.get('rag_answer', '').startswith('[ERROR'):
                        merged[qid]['rag_answer'] = item['rag_answer']
                        merged_count += 1
            os.remove(fpath)
        except Exception as e:
            print(f'Error merging {fpath}: {e}')
    
    if merged_count > 0:
        print(f'✓ Successfully merged {merged_count} answers from previous run.')
        with open('rag_retrieved_answers.json', 'w') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
    else:
        print('No new completed answers found in existing shard files.')
else:
    print('No existing shard files found. Starting fresh.')
"

echo "⚡ Starting 1-shard Web Automation LLM Generation..."
echo "📂 Logging output to web_shard_1.log"

# Launch the single shard in the background
../rag/bin/python3 -u automate_llm_generation.py --browser gemini --port 9222 --shard 1 --num-shards 1 > web_shard_1.log 2>&1 &
PID1=$!

echo "🚀 Web automation process launched. PID: $PID1"
echo "⏳ Waiting for the shard to complete..."

# Wait for the background task to finish
wait $PID1

echo "📂 All processes complete. Merging web shards..."

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

echo "🎉 All web-automated LLM answers completed successfully and merged into rag_retrieved_answers.json!"
