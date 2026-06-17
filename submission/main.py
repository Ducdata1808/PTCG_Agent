import os
import random
import sys

# Append cg to path to import SDK
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cg.api import Observation, to_observation_class, OptionType, SelectContext, AreaType

def read_deck_csv() -> list[int]:
    """Read deck.csv and return list of 60 card IDs."""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = os.path.join("/kaggle_simulations/agent/", file_path)
    if not os.path.exists(file_path):
        # Fallback if both local relative and kaggle path fail
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
        
    deck = []
    with open(file_path, "r") as file:
        lines = file.read().split("\n")
        for line in lines[:60]:
            if line.strip():
                deck.append(int(line.strip()))
    return deck

def evaluate_setup_active(options, obs: Observation) -> list[int]:
    """Choose the best active Pokémon during Setup phase."""
    # Priority: Mewtwo ex (431) > Kangaskhan ex (24) > Articuno (414) > Wobbuffet (432) > Mimikyu (434) > Tarountula (400)
    priority = {431: 10, 24: 9, 414: 8, 432: 7, 434: 6, 400: 5}
    
    best_idx = 0
    best_score = -1
    for idx, opt in enumerate(options):
        card_id = opt.cardId
        score = priority.get(card_id, 0)
        if score > best_score:
            best_score = score
            best_idx = idx
            
    return [best_idx]

def evaluate_setup_bench(options, obs: Observation) -> list[int]:
    """Select Pokémon to place on the Bench during Setup."""
    # Put all basic Pokémon onto the bench if option allows
    # We should return as many as possible up to maxCount
    max_count = obs.select.maxCount
    min_count = obs.select.minCount
    
    # Priority for benching: Spidops doesn't setup directly, we bench Tarountula, Mimikyu, etc.
    # Just select the first max_count options representing cards
    selected_indices = []
    for idx, opt in enumerate(options):
        if len(selected_indices) < max_count:
            selected_indices.append(idx)
            
    return selected_indices if len(selected_indices) >= min_count else list(range(min_count))

def evaluate_main_phase(options, obs: Observation) -> list[int]:
    """Choose the best option during the Main Phase (PLAY, ATTACH, EVOLVE, ATTACK, etc.)."""
    # Group options by their OptionType
    play_options = []
    attach_options = []
    evolve_options = []
    ability_options = []
    attack_options = []
    retreat_options = []
    end_option_idx = None

    for idx, opt in enumerate(options):
        if opt.type == OptionType.PLAY:
            play_options.append((idx, opt))
        elif opt.type == OptionType.ATTACH:
            attach_options.append((idx, opt))
        elif opt.type == OptionType.EVOLVE:
            evolve_options.append((idx, opt))
        elif opt.type == OptionType.ABILITY:
            ability_options.append((idx, opt))
        elif opt.type == OptionType.ATTACK:
            attack_options.append((idx, opt))
        elif opt.type == OptionType.RETREAT:
            retreat_options.append((idx, opt))
        elif opt.type == OptionType.END:
            end_option_idx = idx

    # Priority 1: Use attacks if we can KO or deal big damage
    if attack_options:
        # Choose the attack with the highest damage or index
        # Usually, first attack option is preferred
        return [attack_options[0][0]]

    # Priority 2: Evolve Pokémon to stronger stages
    if evolve_options:
        # Prioritize Spidops (ID 401)
        for idx, opt in evolve_options:
            if opt.cardId == 401:
                return [idx]
        return [evolve_options[0][0]]

    # Priority 3: Attach Energy to attackers
    if attach_options and not obs.current.energyAttached:
        # Prioritize Mewtwo ex (ID 431) first, then Spidops (ID 401), then others
        # We look at the target Pokémon on the field (inPlayIndex)
        for idx, opt in attach_options:
            target_area = opt.inPlayArea
            target_idx = opt.inPlayIndex
            player_idx = obs.current.yourIndex
            
            # Find the Pokémon at target
            target_pkmn = None
            if target_area == AreaType.ACTIVE:
                active_list = obs.current.players[player_idx].active
                if active_list:
                    target_pkmn = active_list[0]
            elif target_area == AreaType.BENCH:
                bench = obs.current.players[player_idx].bench
                if target_idx < len(bench):
                    target_pkmn = bench[target_idx]
            
            if target_pkmn and target_pkmn.id == 431:  # Mewtwo ex
                return [idx]
                
        # Fallback to attaching to anything
        return [attach_options[0][0]]

    # Priority 4: Use Abilities
    if ability_options:
        # e.g., Spidops's "Charging Up" to grab energy from discard
        return [ability_options[0][0]]

    # Priority 5: Play items/supporters from hand
    if play_options:
        # Play Supporters/Items (like Lillie's Determination, Ariana, Transceiver)
        # Transceiver finds Proton/Ariana, Proton/Ariana draws cards.
        return [play_options[0][0]]

    # Priority 6: End Turn
    if end_option_idx is not None:
        return [end_option_idx]

    # Default fallback: random single option
    return [0]

def agent(obs_dict: dict) -> list[int]:
    """Rule-Based AI Training Agent for PTCG."""
    obs: Observation = to_observation_class(obs_dict)
    
    # If select is None, we must return our starting deck list
    if obs.select is None:
        return read_deck_csv()

    options = obs.select.option
    context = obs.select.context
    
    # Check current state/context to determine policy branch
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        return evaluate_setup_active(options, obs)
        
    elif context == SelectContext.SETUP_BENCH_POKEMON:
        return evaluate_setup_bench(options, obs)
        
    elif context == SelectContext.MAIN:
        return evaluate_main_phase(options, obs)
        
    elif context == SelectContext.ATTACK:
        # Select an attack (always use the first attack option)
        return [0]
        
    elif context == SelectContext.TO_ACTIVE:
        # When active is KO'd, bring up Mewtwo ex, Spidops, or Kangaskhan ex
        priority = {431: 10, 401: 9, 24: 8, 414: 7, 432: 6, 434: 5, 400: 4}
        best_idx = 0
        best_score = -1
        for idx, opt in enumerate(options):
            card_id = opt.cardId
            score = priority.get(card_id, 0)
            if score > best_score:
                best_score = score
                best_idx = idx
        return [best_idx]

    # Default fallback: pick maxCount options sequentially
    max_count = obs.select.maxCount
    return list(range(max_count))
