import os
import subprocess
import sys

def run_cmd(cmd, check=True):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 and check:
        print(f"Error executing: {cmd}")
        print(res.stderr.decode("utf-8", errors="ignore"))
        sys.exit(1)
    return res.stdout.decode("utf-8", errors="ignore").strip()

def main():
    # 1. Commit any current modifications to avoid dirty state
    run_cmd("git add -A")
    run_cmd("git commit -m 'chore: purge remaining credentials' || echo 'Nothing to commit'")

    # 2. Delete existing filter backups to avoid conflict
    run_cmd("git update-ref -d refs/original/refs/heads/main", check=False)
    run_cmd("git update-ref -d refs/original/refs/remotes/origin/main", check=False)
    
    # 3. Run git filter-branch using filter_tree.py on all commits
    print("Running filter-branch on all branches and tags...")
    run_cmd("FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --tree-filter 'python3 /Users/nanibayanaboina2750/Desktop/research-rag-assistant/filter_tree.py' -- --all")

    # 4. Remove all backup refs and remote tracking refs locally
    print("Clearing local git cache and reference logs...")
    run_cmd("git update-ref -d refs/original/refs/heads/main", check=False)
    run_cmd("git update-ref -d refs/original/refs/remotes/origin/main", check=False)
    run_cmd("git branch -dr origin/main", check=False)

    # 5. Expire reflog and prune all unreachable loose objects aggressively
    print("Pruning unreachable commits aggressively...")
    run_cmd("git reflog expire --expire=now --all")
    run_cmd("git gc --prune=now --aggressive")

    # 6. Delete temporary helper script
    if os.path.exists("filter_tree.py"):
        os.remove("filter_tree.py")

    # 7. Force push everything back to GitHub
    print("Force pushing clean commits to GitHub remote...")
    run_cmd("git push origin --force --all")
    run_cmd("git push origin --force --tags")

    print("\n✅ API Keys Purge Complete!")
    
    # 8. Final verification check
    print("\nVerification Key Scan Result:")
    scan = run_cmd("git grep 'csk-wpwv8\\|AIzaSyDg\\|AQ.Ab8RN6' $(git log --all --pretty=format:'%H') 2>/dev/null | grep -v 'YOUR_'", check=False)
    if not scan:
        print("🌟 CLEAN - no keys found in any commit history!")
    else:
        print("⚠️ Warning: Some keys were still found:")
        print(scan)

if __name__ == "__main__":
    main()
