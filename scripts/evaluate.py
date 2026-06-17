import os
import sys
import random
import time
from collections import Counter

# Ensure the submission folder is resolved
submission_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")
sys.path.insert(0, submission_dir)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, LogType

from main import agent as heuristic_agent

# Random Agent for benchmarking
def random_agent(obs_dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        from main import read_deck_csv
        return read_deck_csv()
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

def play_single_game(agent0, agent1, deck0, deck1):
    """Play a game and return winner, reason, decision durations, damage, and actions."""
    stats = {
        'winner': -1,
        'win_reason': 'Unknown',
        'turns': 0,
        'times': {0: [], 1: []},
        'damage_dealt': {0: 0, 1: 0},
        'cards_played': {0: 0, 1: 0},
        'cards_attached': {0: 0, 1: 0},
        'cards_evolved': {0: 0, 1: 0},
    }
    
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        battle_finish()
        return None

    obs = to_observation_class(obs_dict)
    
    try:
        while obs.current is None or obs.current.result == -1:
            current_player = obs.current.yourIndex if (obs.current is not None) else 0
            
            # Measure decision time
            t0 = time.perf_counter()
            if current_player == 0:
                action = agent0(obs_dict)
            else:
                action = agent1(obs_dict)
            t1 = time.perf_counter()
            
            stats['times'][current_player].append(t1 - t0)
            
            obs_dict = battle_select(action)
            obs = to_observation_class(obs_dict)
            
            # Parse logs
            for log in obs.logs:
                if log.type == LogType.HP_CHANGE:
                    target_player = log.playerIndex
                    change_val = log.value if log.value else 0
                    if change_val < 0:
                        damage = abs(change_val)
                        stats['damage_dealt'][1 - target_player] += damage
                elif log.type == LogType.PLAY:
                    stats['cards_played'][log.playerIndex] += 1
                elif log.type == LogType.ATTACH:
                    stats['cards_attached'][log.playerIndex] += 1
                elif log.type == LogType.EVOLVE:
                    stats['cards_evolved'][log.playerIndex] += 1
                elif log.type == LogType.RESULT:
                    stats['winner'] = log.result
                    reason_map = {
                        1: "Prize Out",
                        2: "Deck Out",
                        3: "Bench Out",
                        4: "Card Effect"
                    }
                    stats['win_reason'] = reason_map.get(log.reason, f"Code {log.reason}")

            stats['turns'] += 1
            if stats['turns'] > 1000:
                stats['win_reason'] = "Timeout"
                break
                
        battle_finish()
        return stats
    except Exception as e:
        battle_finish()
        return None

def run_benchmark(num_games=20):
    print(f"=== RUNNING INTEGRATED EVALUATION BENCHMARK ({num_games} Games) ===")
    print("Agent 0: Heuristic Agent (submission/main.py)")
    print("Agent 1: Random Agent\n")
    
    from main import read_deck_csv
    deck0 = read_deck_csv()
    deck1 = read_deck_csv()
    
    # Aggregated metrics
    agent0_wins = 0
    agent1_wins = 0
    errors = 0
    
    win_reasons = Counter()
    total_turns = 0
    total_damage_dealt = {0: 0, 1: 0}
    total_played = {0: 0, 1: 0}
    total_attached = {0: 0, 1: 0}
    total_evolved = {0: 0, 1: 0}
    
    all_decision_times = {0: [], 1: []}
    
    for i in range(num_games):
        sys.stdout.write(f"\rSimulating Game {i+1}/{num_games}...")
        sys.stdout.flush()
        
        # Swapping player order for fairness
        if i % 2 == 0:
            stats = play_single_game(heuristic_agent, random_agent, deck0, deck1)
            if stats and stats['winner'] == 0:
                agent0_wins += 1
                win_reasons[stats['win_reason']] += 1
            elif stats and stats['winner'] == 1:
                agent1_wins += 1
            else:
                errors += 1
        else:
            stats = play_single_game(random_agent, heuristic_agent, deck1, deck0)
            # When swapped, heuristic agent is Player 1 in the engine
            if stats and stats['winner'] == 1:
                agent0_wins += 1
                win_reasons[stats['win_reason']] += 1
            elif stats and stats['winner'] == 0:
                agent1_wins += 1
            else:
                errors += 1
                
        if stats:
            total_turns += stats['turns']
            # Align engine player index to logical agents (0=heuristic, 1=random)
            h_idx, r_idx = (0, 1) if (i % 2 == 0) else (1, 0)
            
            total_damage_dealt[0] += stats['damage_dealt'][h_idx]
            total_damage_dealt[1] += stats['damage_dealt'][r_idx]
            
            total_played[0] += stats['cards_played'][h_idx]
            total_played[1] += stats['cards_played'][r_idx]
            total_attached[0] += stats['cards_attached'][h_idx]
            total_attached[1] += stats['cards_attached'][r_idx]
            total_evolved[0] += stats['cards_evolved'][h_idx]
            total_evolved[1] += stats['cards_evolved'][r_idx]
            
            all_decision_times[0].extend(stats['times'][h_idx])
            all_decision_times[1].extend(stats['times'][r_idx])

    total_valid_games = agent0_wins + agent1_wins
    win_rate = (agent0_wins / total_valid_games * 100) if total_valid_games > 0 else 0
    avg_turns = total_turns / num_games if num_games > 0 else 0
    
    print("\n\n" + "="*41)
    print("           BENCHMARK RESULTS           ")
    print("="*41)
    print(f"Heuristic Agent Wins : {agent0_wins} ({win_rate:.2f}%)")
    print(f"Random Agent Wins    : {agent1_wins} ({100 - win_rate:.2f}%)")
    print(f"Errors / Timeouts    : {errors}")
    print(f"Average Turn Length  : {avg_turns:.1f} actions/game")
    
    print("\n--- VICTORY TYPE BREAKDOWN (Heuristic) ---")
    for reason, count in win_reasons.items():
        pct = count / agent0_wins * 100 if agent0_wins > 0 else 0
        print(f"* {reason}: {count} games ({pct:.1f}%)")
        
    print("\n--- AVERAGE METRICS PER GAME (Heuristic vs Random) ---")
    print(f"Damage Dealt     : {total_damage_dealt[0]/num_games:.1f} vs {total_damage_dealt[1]/num_games:.1f}")
    print(f"Cards Played     : {total_played[0]/num_games:.1f} vs {total_played[1]/num_games:.1f}")
    print(f"Energy Attached  : {total_attached[0]/num_games:.1f} vs {total_attached[1]/num_games:.1f}")
    print(f"Evolutions Done  : {total_evolved[0]/num_games:.1f} vs {total_evolved[1]/num_games:.1f}")
    
    print("\n--- DECISION TIMINGS ---")
    for name, p_idx in [("Heuristic Agent", 0), ("Random Agent", 1)]:
        times = all_decision_times[p_idx]
        if times:
            avg_time = sum(times) / len(times) * 1000
            max_time = max(times) * 1000
            print(f"* {name}: Avg={avg_time:.2f}ms, Max={max_time:.2f}ms")
    print("="*41)

if __name__ == "__main__":
    run_benchmark(num_games=10)
