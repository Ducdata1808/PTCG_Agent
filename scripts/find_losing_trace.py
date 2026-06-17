import os
import sys
import random

submission_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")
sys.path.insert(0, submission_dir)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, LogType

from main import agent as heuristic_agent
from main import read_deck_csv

def random_agent(obs_dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

def run_trace_and_check():
    deck0 = read_deck_csv()
    deck1 = read_deck_csv()
    
    # Run games until we find a loss
    for game_idx in range(50):
        obs_dict, start_data = battle_start(deck0, deck1)
        obs = to_observation_class(obs_dict)
        
        history = []
        turn = 0
        winner = -1
        
        try:
            while obs.current is None or obs.current.result == -1:
                current_player = obs.current.yourIndex if obs.current else 0
                state = obs.current
                
                # Record step state
                step_log = ""
                if state:
                    active0 = state.players[0].active[0].id if state.players[0].active and state.players[0].active[0] else None
                    active1 = state.players[1].active[0].id if state.players[1].active and state.players[1].active[0] else None
                    step_log += f"\n[Turn {state.turn}] Player {current_player}'s Choice | Context: {obs.select.context} | Type: {obs.select.type}\n"
                    step_log += f"  P0 Active: {active0} (HP: {state.players[0].active[0].hp if active0 else 0}) | Hand: {state.players[0].handCount} | Deck: {state.players[0].deckCount}\n"
                    step_log += f"  P1 Active: {active1} (HP: {state.players[1].active[0].hp if active1 else 0}) | Hand: {state.players[1].handCount} | Deck: {state.players[1].deckCount}\n"
                
                if current_player == 0:
                    action = heuristic_agent(obs_dict)
                else:
                    action = random_agent(obs_dict)
                    
                chosen_opt = obs.select.option[action[0]] if obs.select and action else None
                step_log += f"  Options: {[opt for opt in obs.select.option]}\n"
                step_log += f"  > Player {current_player} selected option {action} -> type: {chosen_opt.type if chosen_opt else 'Deck'}\n"
                
                obs_dict = battle_select(action)
                obs = to_observation_class(obs_dict)
                
                for log in obs.logs:
                    if log.type == LogType.RESULT:
                        winner = log.result
                        step_log += f"  *** Game Result: Player {log.result} won by reason {log.reason} ***\n"
                    elif log.type == LogType.HP_CHANGE:
                        step_log += f"  * HP Change: Player {log.playerIndex} card {log.cardId} change {log.value}\n"
                    elif log.type == LogType.PLAY:
                        step_log += f"  * Play Card: Player {log.playerIndex} card {log.cardId}\n"
                    elif log.type == LogType.ATTACH:
                        step_log += f"  * Attach: Player {log.playerIndex} attached card {log.cardId} to {log.cardIdTarget}\n"
                        
                history.append(step_log)
                turn += 1
                if turn > 500:
                    winner = 2 # Timeout
                    break
            
            battle_finish()
            
            # If Heuristic Agent (Player 0) lost, print the history!
            if winner == 1:
                print(f"=== FOUND A LOSS IN GAME {game_idx+1} ===")
                for step in history[-30:]: # Print last 30 actions leading to defeat
                    print(step)
                return
                
        except Exception as e:
            battle_finish()
            print(f"Game error: {e}")
            
    print("No losses found in 50 games.")

if __name__ == "__main__":
    run_trace_and_check()
