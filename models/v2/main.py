import os
import sys
import re

try:
    AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    AGENT_DIR = "/kaggle_simulations/agent"
    if not os.path.exists(AGENT_DIR):
        AGENT_DIR = os.getcwd()

# Append cg and src to path to import SDK and modules
sys.path.append(AGENT_DIR)
sys.path.append(os.path.join(AGENT_DIR, "src"))

from cg.api import Observation, to_observation_class, OptionType, SelectContext, AreaType, all_attack
from core.card_database import CardDatabase
from search.mcts_search import perform_mcts

# Initialize Card Database globally
db = CardDatabase()

# Load all attacks from the SDK and index by attackId
ATTACK_DB = {a.attackId: a for a in all_attack()}

def read_deck_csv() -> list[int]:
    """Read deck.csv and return list of 60 card IDs."""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = os.path.join("/kaggle_simulations/agent/", file_path)
    if not os.path.exists(file_path):
        file_path = os.path.join(AGENT_DIR, "deck.csv")
        
    deck = []
    with open(file_path, "r") as file:
        lines = file.read().split("\n")
        for line in lines[:60]:
            if line.strip():
                deck.append(int(line.strip()))
    return deck

def evaluate_setup_active(options, obs: Observation) -> list[int]:
    """Choose the best active Pokémon during Setup phase (highest HP/strongest)."""
    best_idx = 0
    best_hp = -1
    for idx, opt in enumerate(options):
        card = db.get_card(opt.cardId)
        hp = card['hp'] if card and card['hp'] else 0
        if hp > best_hp:
            best_hp = hp
            best_idx = idx
    return [best_idx]

def evaluate_setup_bench(options, obs: Observation) -> list[int]:
    """Select Pokémon to place on the Bench during Setup."""
    max_count = obs.select.maxCount
    min_count = obs.select.minCount
    selected_indices = []
    for idx, opt in enumerate(options):
        if len(selected_indices) < max_count:
            selected_indices.append(idx)
    return selected_indices if len(selected_indices) >= min_count else list(range(min_count))

def get_card_id_from_option(opt, obs: Observation) -> int | None:
    """Resolve cardId from Option structure using current game state and selection data."""
    if opt.cardId is not None:
        return opt.cardId
        
    if not obs or not obs.current:
        return None
        
    player_idx = obs.current.yourIndex
    
    # Check opt.area first
    area = opt.area
    index = opt.index
    
    # Fallback to inPlayArea/inPlayIndex if area is None
    if area is None:
        if opt.type in {OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE}:
            area = AreaType.HAND
            index = opt.index
        else:
            area = opt.inPlayArea
            index = opt.inPlayIndex
        
    if area is None or index is None:
        return None
        
    # Resolve based on AreaType
    if area == AreaType.HAND:
        hand = obs.current.players[player_idx].hand
        if hand and index < len(hand):
            return hand[index].id
            
    elif area == AreaType.ACTIVE:
        p_idx = opt.playerIndex if opt.playerIndex is not None else player_idx
        active = obs.current.players[p_idx].active
        if active and index < len(active) and active[index]:
            return active[index].id
            
    elif area == AreaType.BENCH:
        p_idx = opt.playerIndex if opt.playerIndex is not None else player_idx
        bench = obs.current.players[p_idx].bench
        if bench and index < len(bench):
            return bench[index].id
            
    elif area == AreaType.DISCARD:
        p_idx = opt.playerIndex if opt.playerIndex is not None else player_idx
        discard = obs.current.players[p_idx].discard
        if discard and index < len(discard):
            return discard[index].id
            
    elif area == AreaType.DECK:
        if obs.select and obs.select.deck and index < len(obs.select.deck):
            return obs.select.deck[index].id
            
    elif area == AreaType.LOOKING:
        looking = obs.current.looking
        if looking and index < len(looking) and looking[index]:
            return looking[index].id
            
    elif area == AreaType.STADIUM:
        stadium = obs.current.stadium
        if stadium and index < len(stadium):
            return stadium[index].id
            
    return None

def get_attack_damage_by_id(attack_id):
    """Retrieve damage of an attack by attack ID using the SDK database."""
    attack = ATTACK_DB.get(attack_id)
    return attack.damage if attack else 0

def score_attach_option(opt, obs: Observation) -> float:
    """Score an energy attachment option based on matching types and priority targets."""
    energy_id = get_card_id_from_option(opt, obs)
    energy_card = db.get_card(energy_id) if energy_id else None
    if not energy_card:
        return 0.0
        
    energy_type = energy_card.get('type', '')
    
    player_idx = obs.current.yourIndex
    target_id = None
    target_in_play_area = opt.inPlayArea
    target_in_play_index = opt.inPlayIndex
    
    if target_in_play_area is None:
        target_in_play_area = AreaType.ACTIVE
        target_in_play_index = 0
        
    if target_in_play_area == AreaType.ACTIVE:
        active = obs.current.players[player_idx].active
        if active and target_in_play_index < len(active) and active[target_in_play_index]:
            target_id = active[target_in_play_index].id
    elif target_in_play_area == AreaType.BENCH:
        bench = obs.current.players[player_idx].bench
        if bench and target_in_play_index < len(bench):
            target_id = bench[target_in_play_index].id
            
    if not target_id:
        return 0.0
        
    target_card = db.get_card(target_id)
    if not target_card:
        return 0.0
        
    target_type = target_card.get('type', '')
    
    # Base score
    score = 1.0
    
    # Matching type bonus
    if energy_type and target_type and energy_type == target_type:
        score += 10.0
        
    # Team Rocket's Energy matching
    if energy_id == 15:
        target_name = target_card.get('name', '').lower()
        target_category = target_card.get('category', '').lower()
        if "rocket" in target_name or "rocket" in target_category:
            score += 15.0
            
    # Prefer Active spot
    if target_in_play_area == AreaType.ACTIVE:
        score += 2.0
        
    # Prefer ex attackers
    target_rule = target_card.get('rule', '').lower()
    target_name = target_card.get('name', '').lower()
    if "ex" in target_rule or "ex" in target_name:
        score += 5.0
        
    return score

def evaluate_main_phase(options, obs: Observation) -> list[int]:
    """Choose the best option during the Main Phase."""
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

    player_idx = obs.current.yourIndex
    deck_count = obs.current.players[player_idx].deckCount
    is_low_deck = deck_count < 15
    is_critical_deck = deck_count < 8

    # Priority 1: Play Basic Pokémon to Bench if we have space
    current_bench_size = len(obs.current.players[player_idx].bench)
    max_bench_size = obs.current.players[player_idx].benchMax
    
    if current_bench_size < max_bench_size and play_options:
        for idx, opt in play_options:
            card = db.get_card(get_card_id_from_option(opt, obs))
            if card and 'Pokémon' in card['category'] and 'Basic' in card['stage']:
                return [idx]

    # Priority 2: Evolve Pokémon to stronger stages
    if evolve_options:
        for idx, opt in evolve_options:
            if opt.inPlayArea == AreaType.ACTIVE:
                return [idx]
        return [evolve_options[0][0]]

    # Priority 3: Attach Energy (Scored dynamically)
    if attach_options and not obs.current.energyAttached:
        best_opt_idx = attach_options[0][0]
        best_score = -1.0
        for idx, opt in attach_options:
            score = score_attach_option(opt, obs)
            if score > best_score:
                best_score = score
                best_opt_idx = idx
        return [best_opt_idx]

    # Priority 4: Use Abilities (safe check for low deck size)
    if ability_options:
        for idx, opt in ability_options:
            card = db.get_card(get_card_id_from_option(opt, obs))
            name = card['name'] if card else ""
            is_draw_ability = "Factory" in name or "Dudunsparce" in name or "Drakloak" in name or "Chandelure" in name or "Clefairy" in name
            if is_critical_deck and is_draw_ability:
                continue
            return [idx]

    # Priority 5: Play items/supporters from hand (Draw/Search)
    if play_options:
        best_play_idx = None
        best_score = -1
        
        for idx, opt in play_options:
            card = db.get_card(get_card_id_from_option(opt, obs))
            if not card:
                continue
                
            name = card['name']
            score = 1
            
            # Substring matching for draw card name pattern (including copycat, trail, and helmet)
            is_draw_card = any(keyword in name.lower() for keyword in ['draw', 'research', 'determination', 'ariana', 'proton', 'copycat', 'trail', 'helmet'])
            
            if is_low_deck and is_draw_card:
                continue
                
            if card['stage'] == 'Supporter' and is_draw_card:
                score = 10
            elif card['stage'] == 'Item' and ('ball' in name.lower() or 'transceiver' in name.lower() or 'catching' in name.lower()):
                score = 8
            elif card['stage'] == 'Supporter':
                score = 5
                
            if score > best_score:
                best_score = score
                best_play_idx = idx
                
        if best_play_idx is not None and best_score > 1:
            return [best_play_idx]

    # Priority 6: Smart Retreat
    if not attack_options and retreat_options and current_bench_size > 0:
        active_pokemon = obs.current.players[player_idx].active[0] if obs.current.players[player_idx].active else None
        active_energy = len(active_pokemon.energies) if active_pokemon else 0
        
        bench = obs.current.players[player_idx].bench
        best_bench_energy = max([len(b.energies) for b in bench]) if bench else 0
        active_is_articuno = (active_pokemon.id == 414) if active_pokemon else False
        
        if (best_bench_energy > active_energy and best_bench_energy > 0) or (active_is_articuno and best_bench_energy > 0):
            return [retreat_options[0][0]]

    # Priority 7: Attack if available (ends turn)
    if attack_options:
        valid_attacks = []
        for idx, opt in attack_options:
            if opt.attackId == 1046 and deck_count < 8:
                prizes_left = len(obs.current.players[player_idx].prize)
                opp_active = obs.current.players[1 - player_idx].active
                opp_hp = opp_active[0].hp if (opp_active and opp_active[0]) else 999
                is_ex = False
                if opp_active and opp_active[0]:
                    opp_card = db.get_card(opp_active[0].id)
                    if opp_card and ("ex" in opp_card.get('rule', '').lower() or "ex" in opp_card.get('name', '').lower()):
                        is_ex = True
                prizes_needed_to_win = 2 if is_ex else 1
                if prizes_left <= prizes_needed_to_win and opp_hp <= 100:
                    pass
                else:
                    continue
            valid_attacks.append((idx, opt))
            
        if valid_attacks:
            best_idx = valid_attacks[0][0]
            max_damage = -1
            for idx, opt in valid_attacks:
                damage = get_attack_damage_by_id(opt.attackId)
                if damage > max_damage:
                    max_damage = damage
                    best_idx = idx
            return [best_idx]

    # Priority 8: Play remaining non-priority items/supporters
    if play_options:
        card = db.get_card(get_card_id_from_option(play_options[0][1], obs))
        name = card['name'] if card else ""
        is_draw_card = any(keyword in name.lower() for keyword in ['draw', 'research', 'determination', 'ariana', 'proton', 'copycat', 'trail', 'helmet'])
        if not (is_low_deck and is_draw_card):
            return [play_options[0][0]]

    # Priority 9: End Turn
    if end_option_idx is not None:
        return [end_option_idx]

    return [0]

def evaluate_discard_context(options, obs: Observation) -> list[int]:
    """Intelligently select which cards to discard from hand (generic)."""
    scored_options = []
    
    player_idx = obs.current.yourIndex
    hand_cards = obs.current.players[player_idx].hand
    hand_names = [db.get_card(c.id)['name'] for c in hand_cards if db.get_card(c.id)] if hand_cards else []
    
    for idx, opt in enumerate(options):
        card = db.get_card(get_card_id_from_option(opt, obs))
        if not card:
            scored_options.append((0, idx))
            continue
            
        name = card['name']
        score = 5
        
        if "Energy" in card['stage']:
            score = 1
        elif "ex" in card['rule'] or "ex" in name.lower():
            score = 1
        elif hand_names.count(name) > 1:
            score = 8
        elif card['stage'] == 'Stadium':
            score = 7
            
        scored_options.append((score, idx))
        
    scored_options.sort(key=lambda x: x[0], reverse=True)
    
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    selected_indices = [idx for _, idx in scored_options[:max_count]]
    return selected_indices if len(selected_indices) >= min_count else list(range(min_count))

def evaluate_to_hand_context(options, obs: Observation) -> list[int]:
    """Choose which card to search and put into hand (generic)."""
    scored_options = []
    for idx, opt in enumerate(options):
        card = db.get_card(get_card_id_from_option(opt, obs))
        if not card:
            scored_options.append((0, idx))
            continue
            
        name = card['name']
        score = 1
        
        if "ex" in card['rule'] or "ex" in name.lower():
            score = 10
        elif "Stage 1" in card['stage'] or "Stage 2" in card['stage']:
            score = 8
        elif card['stage'] == 'Supporter' and ('draw' in name.lower() or 'research' in name.lower() or 'determination' in name.lower()):
            score = 7
        elif "Basic Pokémon" in card['stage'] or ("Basic" in card['stage'] and "Pokémon" in card['category']):
            score = 5
            
        scored_options.append((score, idx))
        
    scored_options.sort(key=lambda x: x[0], reverse=True)
    
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    selected_indices = [idx for _, idx in scored_options[:max_count]]
    return selected_indices if len(selected_indices) >= min_count else list(range(min_count))

def agent(obs_dict: dict) -> list[int]:
    """Rule-Based AI Training Agent for PTCG."""
    obs: Observation = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()

    options = obs.select.option
    context = obs.select.context
    
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        return evaluate_setup_active(options, obs)
        
    elif context == SelectContext.SETUP_BENCH_POKEMON:
        return evaluate_setup_bench(options, obs)
        
    elif context == SelectContext.MAIN:
        heuristic_action = evaluate_main_phase(options, obs)
        if heuristic_action:
            chosen_opt = options[heuristic_action[0]]
            if chosen_opt.type in {OptionType.ATTACK, OptionType.ATTACH, OptionType.EVOLVE}:
                return heuristic_action
        try:
            deck = read_deck_csv()
            time_limit = float(os.getenv("MCTS_TIME_LIMIT_MS", "800.0"))
            action = perform_mcts(obs_dict, deck, time_limit_ms=time_limit)
            if action:
                return action
        except Exception:
            pass
        return heuristic_action
        
    elif context == SelectContext.DISCARD:
        return evaluate_discard_context(options, obs)
        
    elif context in {SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD}:
        return evaluate_to_hand_context(options, obs)
        
    elif context == SelectContext.ATTACK:
        best_idx = 0
        max_damage = -1
        for idx, opt in enumerate(options):
            damage = get_attack_damage_by_id(opt.attackId)
            if damage > max_damage:
                max_damage = damage
                best_idx = idx
        return [best_idx]
        
    elif context == SelectContext.TO_ACTIVE:
        player_idx = obs.current.yourIndex
        bench = obs.current.players[player_idx].bench
        
        best_idx = 0
        best_score = -1
        for idx, opt in enumerate(options):
            bench_idx = opt.index
            attached_energy = 0
            
            if bench_idx is not None and bench_idx < len(bench):
                attached_energy = len(bench[bench_idx].energies)
                
            score = attached_energy * 10
            if score > best_score:
                best_score = score
                best_idx = idx
        return [best_idx]
        
    elif context == SelectContext.ATTACH_TO:
        for idx, opt in enumerate(options):
            if opt.inPlayArea == AreaType.ACTIVE:
                return [idx]
        return [0]

    max_count = obs.select.maxCount
    return list(range(max_count))
