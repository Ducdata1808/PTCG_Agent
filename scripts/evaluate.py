import os
import sys
import random
import time

# Ensure we only append the submission folder so that the cg library is only imported once.
submission_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")
sys.path.insert(0, submission_dir)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
from main import agent as heuristic_agent

# Random Agent for benchmarking
def random_agent(obs_dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        from main import read_deck_csv
        return read_deck_csv()
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

def play_single_game(agent0, agent1, deck0, deck1):
    """Simulate a single match between agent0 (Player 0) and agent1 (Player 1)."""
    obs_dict, start_data = battle_start(deck0, deck1)
    
    if start_data.errorType != 0:
        battle_finish()
        return -1, "StartError"

    obs = to_observation_class(obs_dict)
    turn_count = 0
    
    try:
        while obs.current is None or obs.current.result == -1:
            current_player = obs_dict["selectPlayer"] if "selectPlayer" in obs_dict else 0
            
            # Select action based on active player
            if current_player == 0:
                action = agent0(obs_dict)
            else:
                action = agent1(obs_dict)
                
            obs_dict = battle_select(action)
            obs = to_observation_class(obs_dict)
            
            turn_count += 1
            if turn_count > 1000:
                battle_finish()
                return -1, "Timeout"
                
        winner = obs.current.result
        battle_finish()
        return winner, "Success"
        
    except Exception as e:
        battle_finish()
        return -1, f"RuntimeError: {str(e)}"

def run_benchmark(num_games=100):
    print(f"=== RUNNING BENCHMARK ({num_games} Games) ===")
    print("Agent 0: Heuristic Agent (submission/main.py)")
    print("Agent 1: Random Agent")
    
    from main import read_deck_csv
    deck0 = read_deck_csv()
    deck1 = read_deck_csv()
    
    agent0_wins = 0
    agent1_wins = 0
    errors = 0
    
    # Run games, swap player index half the time to ensure fairness (1st vs 2nd turn bias)
    for i in range(num_games):
        sys.stdout.write(f"\rPlaying Game {i+1}/{num_games}...")
        sys.stdout.flush()
        
        # Alternate who goes first/second in the engine
        if i % 2 == 0:
            winner, status = play_single_game(heuristic_agent, random_agent, deck0, deck1)
            if winner == 0:
                agent0_wins += 1
            elif winner == 1:
                agent1_wins += 1
            else:
                errors += 1
        else:
            # Swap decks and agents
            winner, status = play_single_game(random_agent, heuristic_agent, deck1, deck0)
            if winner == 1:
                agent0_wins += 1
            elif winner == 0:
                agent1_wins += 1
            else:
                errors += 1
                
    total_valid_games = agent0_wins + agent1_wins
    win_rate = (agent0_wins / total_valid_games * 100) if total_valid_games > 0 else 0
    
    print("\n\n=== BENCHMARK RESULTS ===")
    print(f"Heuristic Agent Wins: {agent0_wins}")
    print(f"Random Agent Wins   : {agent1_wins}")
    print(f"Errors / Timeouts   : {errors}")
    print(f"Heuristic Win Rate  : {win_rate:.2f}%")
    print("=========================")

if __name__ == "__main__":
    # Run 10 games as a quick check
    run_benchmark(num_games=10)
