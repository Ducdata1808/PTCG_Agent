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

def run_trace():
    deck0 = read_deck_csv()
    deck1 = read_deck_csv()
    
    obs_dict, start_data = battle_start(deck0, deck1)
    obs = to_observation_class(obs_dict)
    
    turn = 0
    while obs.current is None or obs.current.result == -1:
        current_player = obs.current.yourIndex if obs.current else 0
        
        # Log active state
        state = obs.current
        if state:
            active0 = state.players[0].active[0].id if state.players[0].active and state.players[0].active[0] else None
            active1 = state.players[1].active[0].id if state.players[1].active and state.players[1].active[0] else None
            print(f"\n[Turn {state.turn}] Player {current_player}'s Choice | Context: {obs.select.context} | Type: {obs.select.type}")
            print(f"  P0 Active: {active0} (HP: {state.players[0].active[0].hp if active0 else 0}) | Hand: {state.players[0].handCount} | Deck: {state.players[0].deckCount}")
            print(f"  P1 Active: {active1} (HP: {state.players[1].active[0].hp if active1 else 0}) | Hand: {state.players[1].handCount} | Deck: {state.players[1].deckCount}")
        
        if current_player == 0:
            action = heuristic_agent(obs_dict)
            chosen_opt = obs.select.option[action[0]] if obs.select and action else None
            print(f"  > Heuristic Agent selected option {action} -> type: {chosen_opt.type if chosen_opt else 'Deck'}")
        else:
            action = random_agent(obs_dict)
            chosen_opt = obs.select.option[action[0]] if obs.select and action else None
            print(f"  > Random Agent selected option {action} -> type: {chosen_opt.type if chosen_opt else 'Deck'}")
            
        obs_dict = battle_select(action)
        obs = to_observation_class(obs_dict)
        
        # Print logs from this action
        for log in obs.logs:
            if log.type == LogType.RESULT:
                print(f"  *** Game Result: Player {log.result} won by reason {log.reason} ***")
            elif log.type == LogType.HP_CHANGE:
                print(f"  * HP Change: Player {log.playerIndex} card {log.cardId} change {log.value}")
            elif log.type == LogType.PLAY:
                print(f"  * Play Card: Player {log.playerIndex} card {log.cardId}")
            elif log.type == LogType.ATTACH:
                print(f"  * Attach: Player {log.playerIndex} attached card {log.cardId} to {log.cardIdTarget}")
            elif log.type == LogType.EVOLVE:
                print(f"  * Evolve: Player {log.playerIndex} evolved {log.cardIdTarget} to {log.cardId}")

        turn += 1
        if turn > 150:
            print("Trace stopped due to max turn limit.")
            break
            
    battle_finish()

if __name__ == "__main__":
    run_trace()
