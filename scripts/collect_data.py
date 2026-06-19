import os
import sys
import random
import json
import time
import glob
from multiprocessing import Pool, cpu_count

# Ensure root and submission folders are in the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
submission_dir = os.path.join(root_dir, "submission")
sys.path.insert(0, root_dir)
sys.path.insert(0, submission_dir)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, LogType, SelectContext
from main import (
    evaluate_setup_active,
    evaluate_setup_bench,
    evaluate_main_phase,
    evaluate_discard_context,
    evaluate_to_hand_context,
    get_attack_damage_by_id,
    db
)
from search.features import extract_features
from search.option_features import extract_option_features
from search.mcts_search import perform_mcts, get_possible_actions

def load_deck(path):
    deck = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                deck.append(int(line.strip()))
    return deck

def play_and_collect(deck0_path, deck1_path):
    from main import agent as v3_agent
    from models.v2 import agent as v2_agent

    deck0 = load_deck(deck0_path)
    deck1 = load_deck(deck1_path)
    
    # Identify which player is V3 based on deck path
    d0_norm = deck0_path.replace("\\", "/").strip("./")
    d1_norm = deck1_path.replace("\\", "/").strip("./")
    if "submission/deck.csv" in d0_norm:
        agent0, agent1 = v3_agent, v2_agent
    elif "submission/deck.csv" in d1_norm:
        agent0, agent1 = v2_agent, v3_agent
    else:
        role_assignment = random.choice([0, 1])
        if role_assignment == 0:
            agent0, agent1 = v3_agent, v2_agent
        else:
            agent0, agent1 = v2_agent, v3_agent
    
    value_data = []  # List of (state_feats, player_idx)
    policy_data = [] # List of (option_feats, target_probability)
    
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        battle_finish()
        return None, [], []
 
    obs = to_observation_class(obs_dict)
    winner = -1
    turns = 0
    
    try:
        while obs.current is None or obs.current.result == -1:
            current_player = obs.current.yourIndex if (obs.current is not None) else 0
            
            # Extract state features for value network training
            if obs.current is not None:
                value_data.append((extract_features(obs, 0), 0))
                value_data.append((extract_features(obs, 1), 1))

            # Select action using the assigned agent
            if current_player == 0:
                action = agent0(obs_dict)
            else:
                action = agent1(obs_dict)
                
            obs_dict = battle_select(action)
            obs = to_observation_class(obs_dict)
            
            for log in obs.logs:
                if log.type == LogType.RESULT:
                    winner = log.result
                    
            turns += 1
            if turns > 1000:
                break
                
        battle_finish()
        return winner, value_data, policy_data
    except Exception as e:
        battle_finish()
        return None, [], []

def run_single_game(args):
    deck0_path, deck1_path, seed = args
    random.seed(seed)
    os.environ["MCTS_TIME_LIMIT_MS"] = "150.0"
    
    # We run the game with a localized retry/timeout safety
    try:
        winner, value_data, policy_data = play_and_collect(deck0_path, deck1_path)
    except Exception:
        battle_finish()
        return [], []
        
    labeled_value_samples = []
    if winner is not None and value_data:
        for feat, player_idx in value_data:
            if winner == player_idx:
                label = 1.0
            elif winner == 1 - player_idx:
                label = 0.0
            else:
                label = 0.5
            labeled_value_samples.append({
                "features": feat,
                "label": label
            })
            
    return labeled_value_samples, []

def main():
    decks_dir = "decks/csv_file"
    deck_files = glob.glob(os.path.join(decks_dir, "*.csv"))
    if not deck_files:
        print("Error: No CSV deck files found in decks/csv_file")
        sys.exit(1)
        
    num_games = 1000
    if len(sys.argv) > 1:
        try:
            num_games = int(sys.argv[1])
        except ValueError:
            pass

    cores = cpu_count()
    print(f"Starting self-play data collection: {num_games} games...")
    print(f"Found {len(deck_files)} decks in {decks_dir}")
    print(f"Utilizing {cores} CPU cores...")
    
    tasks = []
    submission_deck = "submission/deck.csv"
    for i in range(num_games):
        role_assignment = random.choice([0, 1])
        if role_assignment == 0:
            deck0_path = submission_deck
            deck1_path = random.choice(deck_files)
        else:
            deck0_path = random.choice(deck_files)
            deck1_path = submission_deck
        seed = int(time.time() * 1000) % 2**32 + i
        tasks.append((deck0_path, deck1_path, seed))
        
    all_value_samples = []
    start_time = time.time()
    
    # We will write/append chunks of samples to a temporary file,
    # or write them periodically to prevent losing all data on crash/hang.
    out_path = "data/self_play_data.json"
    os.makedirs("data", exist_ok=True)
    
    completed = 0
    with Pool(processes=cores) as pool:
        # We set a timeout per game to prevent hanging indefinitely
        iterator = pool.imap_unordered(run_single_game, tasks, chunksize=1)
        while completed < num_games:
            try:
                # 30 seconds limit per game execution in queue
                val_samples, pol_samples = iterator.next(timeout=30.0)
                all_value_samples.extend(val_samples)
                completed += 1
            except StopIteration:
                break
            except Exception as e:
                # If a task times out or fails, skip it
                completed += 1
                
            if completed % 50 == 0 or completed == num_games:
                elapsed = time.time() - start_time
                games_per_sec = completed / elapsed if elapsed > 0 else 0
                print(f"Game {completed}/{num_games} completed. Total samples: {len(all_value_samples)}. Elapsed time: {elapsed:.1f}s ({games_per_sec:.2f} games/s)")
                
                # Checkpoint saving every 1000 games
                if completed % 1000 == 0:
                    with open(out_path, "w") as f:
                        json.dump(all_value_samples, f)
                    print(f"Checkpoint saved: {len(all_value_samples)} samples written to {out_path}")

    # Final Save
    with open(out_path, "w") as f:
        json.dump(all_value_samples, f)
        
    print(f"Data collection complete! Saved {len(all_value_samples)} samples to {out_path}")

if __name__ == "__main__":
    main()
