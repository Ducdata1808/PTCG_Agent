import os
import sys
from cg.api import Observation, Option, OptionType, AreaType

# Resolve relative path for card database
try:
    from core.card_database import CardDatabase
except ImportError:
    # Add root/submission to path if running inside scripts
    submission_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if submission_dir not in sys.path:
        sys.path.append(submission_dir)
    from core.card_database import CardDatabase

db = CardDatabase()

def get_card_id_from_option(opt: Option, obs: Observation) -> int | None:
    """Resolve cardId from Option structure using current game state and selection data."""
    if opt.cardId is not None:
        return opt.cardId
        
    if not obs or not obs.current:
        return None
        
    player_idx = obs.current.yourIndex
    area = opt.area
    index = opt.index
    
    if area is None:
        if opt.type in {OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE}:
            area = AreaType.HAND
            index = opt.index
        else:
            area = opt.inPlayArea
            index = opt.inPlayIndex
        
    if area is None or index is None:
        return None
        
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
            
    return None

def extract_option_features(opt: Option, obs: Observation) -> list[float]:
    """
    Extracts a 16-dimensional feature vector for a candidate action option.
    """
    features = [0.0] * 16
    
    # 1. Option type one-hot encoding (indices 0-6)
    if opt.type == OptionType.PLAY:
        features[0] = 1.0
    elif opt.type == OptionType.ATTACH:
        features[1] = 1.0
    elif opt.type == OptionType.EVOLVE:
        features[2] = 1.0
    elif opt.type == OptionType.ABILITY:
        features[3] = 1.0
    elif opt.type == OptionType.ATTACK:
        features[4] = 1.0
    elif opt.type == OptionType.RETREAT:
        features[5] = 1.0
    elif opt.type == OptionType.END:
        features[6] = 1.0
    else:
        # Fallback/other type
        pass

    # Resolve card info
    card_id = get_card_id_from_option(opt, obs)
    card = db.get_card(card_id) if card_id else None
    
    if card:
        # 2. Card category / stage features (indices 7-10)
        category = card.get('category', '')
        stage = card.get('stage', '')
        name = card.get('name', '').lower()
        
        # Is Trainer (Supporter/Item/Stadium)?
        if 'Trainer' in category:
            features[7] = 1.0
            if 'Supporter' in stage:
                features[8] = 1.0
            elif 'Item' in stage:
                features[9] = 1.0
        # Is Pokemon?
        elif 'Pokémon' in category:
            features[10] = 1.0
            
        # HP (index 11)
        hp = card.get('hp', 0)
        if hp:
            features[11] = float(hp) / 300.0

    # Target info
    player_idx = obs.current.yourIndex if (obs and obs.current) else 0
    target_area = opt.inPlayArea
    target_idx = opt.inPlayIndex
    
    if target_area is None:
        target_area = AreaType.ACTIVE
        target_idx = 0
        
    # 3. Target destination features (indices 12-14)
    if target_area == AreaType.ACTIVE:
        features[12] = 1.0
        # Target attached energy
        if obs and obs.current and obs.current.players[player_idx].active:
            active = obs.current.players[player_idx].active[0]
            if active and active.energies:
                features[14] = float(len(active.energies)) / 5.0
    elif target_area == AreaType.BENCH:
        features[13] = 1.0
        if obs and obs.current and obs.current.players[player_idx].bench and target_idx is not None:
            bench = obs.current.players[player_idx].bench
            if target_idx < len(bench) and bench[target_idx] and bench[target_idx].energies:
                features[14] = float(len(bench[target_idx].energies)) / 5.0

    # 4. Context specific match (index 15)
    # Energy type matching or evolution matching
    if opt.type == OptionType.ATTACH and card:
        energy_type = card.get('type', '')
        # Check target type
        target_card_id = None
        if target_area == AreaType.ACTIVE and obs and obs.current and obs.current.players[player_idx].active:
            target_card_id = obs.current.players[player_idx].active[0].id
        elif target_area == AreaType.BENCH and obs and obs.current and obs.current.players[player_idx].bench and target_idx is not None:
            bench = obs.current.players[player_idx].bench
            if target_idx < len(bench) and bench[target_idx]:
                target_card_id = bench[target_idx].id
                
        if target_card_id:
            target_card = db.get_card(target_card_id)
            if target_card and target_card.get('type') == energy_type:
                features[15] = 1.0
                
    elif opt.type == OptionType.EVOLVE:
        # Check if active evolution
        if target_area == AreaType.ACTIVE:
            features[15] = 1.0

    return features
