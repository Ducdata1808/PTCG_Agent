import os
import sys
import shutil
import subprocess

def run_command(command, cwd=None):
    """Run a shell command and stream output in real-time."""
    print(f"\n> Running: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            shell=True
        )
        process.wait()
        if process.returncode != 0:
            print(f"Error: Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)
    except Exception as e:
        print(f"Exception while running command: {e}")
        sys.exit(1)

def select_deck(deck_name):
    """Copy a deck CSV file to submission/deck.csv."""
    if not deck_name.endswith(".csv"):
        deck_name += ".csv"
        
    src_path = os.path.join("decks", "csv_file", deck_name)
    dest_path = os.path.join("submission", "deck.csv")
    
    if not os.path.exists(src_path):
        print(f"Error: Deck file '{src_path}' not found.")
        print("Available decks in decks/csv_file/:")
        for f in os.listdir(os.path.join("decks", "csv_file")):
            if f.endswith(".csv"):
                print(f"  - {f[:-4]}")
        sys.exit(1)
        
    print(f"Selecting deck '{deck_name}'...")
    shutil.copyfile(src_path, dest_path)
    print(f"Successfully updated '{dest_path}' with deck '{deck_name}'.")

def train_pipeline(num_games):
    """Execute the data collection, training, and packaging pipeline."""
    # 1. Run Data Collection
    print(f"\n--- STEP 1: Collecting {num_games} Self-Play Matches ---")
    run_command(["python", "scripts/collect_data.py", str(num_games)])
    
    # 2. Train Networks
    print("\n--- STEP 2: Training Value and Policy Networks ---")
    run_command(["python", "scripts/train_value_net.py"])
    
    # 3. Sync Weights
    print("\n--- STEP 3: Syncing Weights to Models Directory ---")
    weights = ["value_net_weights.json", "policy_net_weights.json"]
    for w in weights:
        src = os.path.join("submission", "src", "search", w)
        dest = os.path.join("models", "v2", "src", "search", w)
        if os.path.exists(src):
            shutil.copyfile(src, dest)
            print(f"Copied {w} to models/v2/src/search/")
        else:
            print(f"Warning: {src} not found, skipping sync for {w}.")
            
    # 4. Package Submission
    print("\n--- STEP 4: Packaging Submission into submission.tar.gz ---")
    run_command(["tar", "-czf", "submission.tar.gz", "-C", "submission", "."])
    print("Pipeline completed successfully! Package 'submission.tar.gz' is ready.")

def evaluate_pipeline():
    """Run V4 vs V1 and V4 vs V2 benchmarks."""
    print("\n--- STEP 1: Evaluating V4 vs Heuristic Agent (V1) ---")
    run_command(["python", "scripts/evaluate.py"])
    
    print("\n--- STEP 2: Evaluating V4 vs MCTS Agent (V2) ---")
    run_command(["python", "scripts/evaluate_v4_vs_v2.py"])

def print_usage():
    print("Usage: python manage_agent.py <command> [arguments]")
    print("\nCommands:")
    print("  select-deck <deck_name>  - Set active deck from decks/csv_file/")
    print("  train <num_games>        - Run self-play data collection, train nets, sync, and package")
    print("  evaluate                 - Run benchmarks: V4 vs V1 and V4 vs V2")
    print("\nExample:")
    print("  python manage_agent.py select-deck Alakazam")
    print("  python manage_agent.py train 20000")
    print("  python manage_agent.py evaluate")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
        
    cmd = sys.argv[1].lower()
    
    if cmd == "select-deck":
        if len(sys.argv) < 3:
            print("Error: select-deck requires a deck name.")
            print_usage()
            sys.exit(1)
        select_deck(sys.argv[2])
        
    elif cmd == "train":
        num_games = 1000
        if len(sys.argv) >= 3:
            try:
                num_games = int(sys.argv[2])
            except ValueError:
                print(f"Invalid game count '{sys.argv[2]}', defaulting to 1000.")
        train_pipeline(num_games)
        
    elif cmd == "evaluate":
        evaluate_pipeline()
        
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
