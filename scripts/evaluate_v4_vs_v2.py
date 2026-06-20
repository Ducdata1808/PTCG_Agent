import os
import sys
import time
from collections import Counter

# Add workspace root to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)

# Add submission to path for V4 import
submission_dir = os.path.join(workspace_dir, "submission")
sys.path.insert(0, submission_dir)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, LogType

# Import V4 Agent
from main import agent as v4_agent

# Import V2 Agent
from models.v2 import agent as v2_agent

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
        import traceback
        traceback.print_exc()
        battle_finish()
        return None

def run_benchmark(num_games=10):
    print(f"=== RUNNING V4 VS V2 EVALUATION BENCHMARK ({num_games} Games) ===")
    print("Agent 0 (V4): MCTS Search Agent (submission/main.py)")
    print("Agent 1 (V2): IS-MCTS Search Agent (models/v2/main.py)\n")
    
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
            stats = play_single_game(v4_agent, v2_agent, deck0, deck1)
            if stats and stats['winner'] == 0:
                agent0_wins += 1
                win_reasons[stats['win_reason']] += 1
            elif stats and stats['winner'] == 1:
                agent1_wins += 1
            else:
                errors += 1
        else:
            stats = play_single_game(v2_agent, v4_agent, deck1, deck0)
            if stats and stats['winner'] == 1:
                agent0_wins += 1
                win_reasons[stats['win_reason']] += 1
            elif stats and stats['winner'] == 0:
                agent1_wins += 1
            else:
                errors += 1
                
        if stats:
            total_turns += stats['turns']
            # Align engine player index to logical agents (0=v4_agent, 1=v2_agent)
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
    print(f"V4 Search Agent Wins      : {agent0_wins} ({win_rate:.2f}%)")
    print(f"V2 Search Agent Wins      : {agent1_wins} ({100 - win_rate:.2f}%)")
    print(f"Errors / Timeouts         : {errors}")
    print(f"Average Turn Length       : {avg_turns:.1f} actions/game")
    
    print("\n--- VICTORY TYPE BREAKDOWN (V4 Search Agent) ---")
    for reason, count in win_reasons.items():
        pct = count / agent0_wins * 100 if agent0_wins > 0 else 0
        print(f"* {reason}: {count} games ({pct:.1f}%)")
        
    print("\n--- AVERAGE METRICS PER GAME (V4 vs V2) ---")
    print(f"Damage Dealt     : {total_damage_dealt[0]/num_games:.1f} vs {total_damage_dealt[1]/num_games:.1f}")
    print(f"Cards Played     : {total_played[0]/num_games:.1f} vs {total_played[1]/num_games:.1f}")
    print(f"Energy Attached  : {total_attached[0]/num_games:.1f} vs {total_attached[1]/num_games:.1f}")
    print(f"Evolutions Done  : {total_evolved[0]/num_games:.1f} vs {total_evolved[1]/num_games:.1f}")
    
    print("\n--- DECISION TIMINGS ---")
    for name, p_idx in [("V4 Agent", 0), ("V2 Agent", 1)]:
        times = all_decision_times[p_idx]
        if times:
            avg_time = sum(times) / len(times) * 1000
            max_time = max(times) * 1000
            print(f"* {name}: Avg={avg_time:.2f}ms, Max={max_time:.2f}ms")
    print("="*41)

if __name__ == "__main__":
    run_benchmark(num_games=10)
