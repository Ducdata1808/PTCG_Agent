import os
import random
from cg.api import Observation, AreaType, Card

import json

# Load meta-decks list dynamically
META_DECKS = {}
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "meta_decks.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            META_DECKS = json.load(f)
except Exception:
    pass

def guess_opponent_decklist(obs: Observation, own_deck: list[int]) -> list[int]:
    """Guesses the opponent's starting 60-card list based on visible cards."""
    player_idx = obs.current.yourIndex
    opp_idx = 1 - player_idx
    
    # Collect all visible card IDs from opponent's side
    visible_ids = []
    
    # Active Pokémon
    opp_active = obs.current.players[opp_idx].active
    if opp_active and opp_active[0]:
        visible_ids.append(opp_active[0].id)
        for ec in opp_active[0].energyCards:
            visible_ids.append(ec.id)
        for tc in opp_active[0].tools:
            visible_ids.append(tc.id)
            
    # Bench Pokémon
    for b in obs.current.players[opp_idx].bench:
        visible_ids.append(b.id)
        for ec in b.energyCards:
            visible_ids.append(ec.id)
        for tc in b.tools:
            visible_ids.append(tc.id)
            
    # Discard pile
    for c in obs.current.players[opp_idx].discard:
        visible_ids.append(c.id)
        
    best_deck_name = None
    best_score = -1
    
    for name, deck_ids in META_DECKS.items():
        score = 0
        temp_deck = list(deck_ids)
        for cid in visible_ids:
            if cid in temp_deck:
                score += 1
                temp_deck.remove(cid)
        if score > best_score:
            best_score = score
            best_deck_name = name
            
    if best_deck_name and best_score > 0:
        return list(META_DECKS[best_deck_name])
        
    # Fallback to the first deck in META_DECKS that is not identical to our own deck
    if META_DECKS:
        for name in META_DECKS:
            return list(META_DECKS[name])
            
    # Universal fallback if JSON fails to load
    return [
        418, 418, 418, 418, 723, 723, 723, 723, 721, 721, 478, 1227, 1227, 1227, 1227,
        1182, 1182, 1197, 1145, 1145, 1087, 1087, 1092, 1121, 1262, 1262, 1262, 1102,
        1102, 1213, 1213, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3
    ]

def get_remaining_pool(starting_list: list[int], visible_ids: list[int]) -> list[int]:
    """Computes starting_list - visible_ids safely with duplicates."""
    pool = list(starting_list)
    for cid in visible_ids:
        if cid in pool:
            pool.remove(cid)
    return pool

def determinize_state(obs: Observation, own_deck_full: list[int]) -> dict:
    """
    Creates a random determinized state compatible with search_begin.
    Returns a dictionary of parameter arguments for search_begin.
    """
    player_idx = obs.current.yourIndex
    opp_idx = 1 - player_idx
    
    # 1. Resolve own visible cards
    own_visible = []
    # Hand
    if obs.current.players[player_idx].hand:
        own_visible.extend([c.id for c in obs.current.players[player_idx].hand])
    # Active
    active = obs.current.players[player_idx].active
    if active and active[0]:
        own_visible.append(active[0].id)
        own_visible.extend([ec.id for ec in active[0].energyCards])
        own_visible.extend([tc.id for tc in active[0].tools])
        own_visible.extend([pe.id for pe in active[0].preEvolution])
    # Bench
    for b in obs.current.players[player_idx].bench:
        own_visible.append(b.id)
        own_visible.extend([ec.id for ec in b.energyCards])
        own_visible.extend([tc.id for tc in b.tools])
        own_visible.extend([pe.id for pe in b.preEvolution])
    # Discard
    own_visible.extend([c.id for c in obs.current.players[player_idx].discard])
    
    # Remaining own cards (deck + prize cards)
    own_pool = get_remaining_pool(own_deck_full, own_visible)
    random.shuffle(own_pool)
    
    # Distribute own remaining cards to Deck and Prizes
    prize_count_own = len(obs.current.players[player_idx].prize)
    deck_count_own = obs.current.players[player_idx].deckCount
    
    # Take from pool
    your_prize = own_pool[:prize_count_own]
    your_deck = own_pool[prize_count_own:prize_count_own + deck_count_own]
    
    # 2. Resolve opponent visible cards
    opp_visible = []
    opp_active = obs.current.players[opp_idx].active
    opp_active_id = []
    if opp_active and opp_active[0]:
        opp_visible.append(opp_active[0].id)
        opp_visible.extend([ec.id for ec in opp_active[0].energyCards])
        opp_visible.extend([tc.id for tc in opp_active[0].tools])
        opp_visible.extend([pe.id for pe in opp_active[0].preEvolution])
    elif opp_active and len(opp_active) > 0 and opp_active[0] is None:
        # Opponent active is face-down at Setup phase
        pass
        
    for b in obs.current.players[opp_idx].bench:
        opp_visible.append(b.id)
        opp_visible.extend([ec.id for ec in b.energyCards])
        opp_visible.extend([tc.id for tc in b.tools])
        opp_visible.extend([pe.id for pe in b.preEvolution])
        
    opp_visible.extend([c.id for c in obs.current.players[opp_idx].discard])
    
    # Opponent deck archetype guessing
    opp_deck_full = guess_opponent_decklist(obs, own_deck_full)
    opp_pool = get_remaining_pool(opp_deck_full, opp_visible)
    random.shuffle(opp_pool)
    
    # Distribute opponent remaining cards to Hand, Deck, and Prizes
    hand_count_opp = obs.current.players[opp_idx].handCount
    prize_count_opp = len(obs.current.players[opp_idx].prize)
    deck_count_opp = obs.current.players[opp_idx].deckCount
    
    opponent_hand = opp_pool[:hand_count_opp]
    opponent_prize = opp_pool[hand_count_opp:hand_count_opp + prize_count_opp]
    opponent_deck = opp_pool[hand_count_opp + prize_count_opp:hand_count_opp + prize_count_opp + deck_count_opp]
    
    # Resolve face-down opponent active Pokémon (if any)
    opponent_active = []
    if opp_active and len(opp_active) > 0 and opp_active[0] is None:
        # Opponent active is face-down. Predict a Basic Pokémon from their remaining hand/deck
        basics = [cid for cid in opp_pool if cid in {400, 431, 414, 432, 24, 434, 272, 721, 722}]
        if basics:
            opponent_active = [basics[0]]
        else:
            opponent_active = [722] # Default fallback Snover
            
    # Return dict of arguments for search_begin
    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active
    }
