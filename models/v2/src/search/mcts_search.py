import os
import sys
import time
import math
import random

# Resolve imports for local sub-directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.api import Observation, to_observation_class, search_begin, search_step, search_release, search_end, OptionType, SelectContext
from search.determinize import determinize_state
from search.rollout_policy import rollout_agent, get_card_id_from_option, db

class MCTSNode:
    """A node in the IS-MCTS search tree using PUCT."""
    def __init__(self, parent=None, action=None, prior=1.0):
        self.parent = parent
        self.action = action  # Action tuple that led to this node
        self.children = {}    # Maps action tuple -> MCTSNode
        self.visits = 0
        self.wins = 0.0
        self.prior = prior

    def ucb1(self, total_visits, exploration=1.414) -> float:
        win_score = (self.wins / self.visits) if self.visits > 0 else 0.5
        exploration_term = exploration * self.prior * math.sqrt(total_visits + 1) / (1 + self.visits)
        return win_score + exploration_term

def get_action_prior(action_tuple: tuple[int, ...], obs: Observation) -> float:
    if not action_tuple or not obs.select:
        return 1.0
    idx = action_tuple[0]
    if idx >= len(obs.select.option):
        return 1.0
    opt = obs.select.option[idx]
    
    if opt.type == OptionType.ATTACK:
        return 10.0
    elif opt.type == OptionType.EVOLVE:
        return 8.0
    elif opt.type == OptionType.ATTACH:
        return 6.0
    elif opt.type == OptionType.PLAY:
        card_id = get_card_id_from_option(opt, obs)
        card = db.get_card(card_id) if card_id else None
        if card:
            name = card.get('name', '').lower()
            stage = card.get('stage', '')
            is_draw_card = any(keyword in name for keyword in ['draw', 'research', 'determination', 'ariana', 'proton', 'copycat', 'trail', 'helmet'])
            if stage == 'Supporter':
                return 7.0 if is_draw_card else 4.0
            elif stage == 'Item':
                return 5.0 if ('ball' in name or 'transceiver' in name or 'catching' in name) else 3.0
        return 2.0
    elif opt.type == OptionType.ABILITY:
        return 3.0
    elif opt.type == OptionType.RETREAT:
        return 0.5
    elif opt.type == OptionType.END:
        return 0.1
    return 1.0

def get_possible_actions(obs: Observation) -> list[tuple[int, ...]]:
    """Generates valid action options (represented as tuples of indices)."""
    if not obs.select:
        return [()]
        
    num_options = len(obs.select.option)
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    
    # Simple selection actions
    actions = []
    if min_count == 1 and max_count == 1:
        for idx in range(num_options):
            actions.append((idx,))
    else:
        # Fallback to simple default subsets or single options
        for idx in range(num_options):
            actions.append((idx,))
        # Also add a combination of all if min/max allows
        if min_count <= num_options <= max_count:
            actions.append(tuple(range(num_options)))
        # Also add a subset up to min_count
        if min_count > 1 and min_count <= num_options:
            actions.append(tuple(range(min_count)))
            
    # Remove duplicates
    seen = set()
    unique_actions = []
    for act in actions:
        if act not in seen:
            seen.add(act)
            unique_actions.append(act)
            
    return unique_actions

def evaluate_heuristic(obs: Observation, own_idx: int) -> float:
    if not obs.current or len(obs.current.players) < 2:
        return 0.5
    
    own = obs.current.players[own_idx]
    opp = obs.current.players[1 - own_idx]
    
    # Prize difference: range is [-6, 6] (positive is good for us)
    own_prizes = len(own.prize) if own.prize else 0
    opp_prizes = len(opp.prize) if opp.prize else 0
    prize_diff = opp_prizes - own_prizes
    
    # HP difference
    own_hp = 0.0
    if own.active and own.active[0]:
        own_hp += own.active[0].hp
    for b in own.bench:
        if b:
            own_hp += b.hp
            
    opp_hp = 0.0
    if opp.active and opp.active[0]:
        opp_hp += opp.active[0].hp
    for b in opp.bench:
        if b:
            opp_hp += b.hp
            
    hp_diff = own_hp - opp_hp
    
    # Energy difference
    own_energy = 0
    if own.active and own.active[0]:
        own_energy += len(own.active[0].energies) if own.active[0].energies else 0
    for b in own.bench:
        if b and b.energies:
            own_energy += len(b.energies)
            
    opp_energy = 0
    if opp.active and opp.active[0]:
        opp_energy += len(opp.active[0].energies) if opp.active[0].energies else 0
    for b in opp.bench:
        if b and b.energies:
            opp_energy += len(b.energies)
            
    energy_diff = own_energy - opp_energy
    
    # Deck count penalty
    deck_penalty = 0.0
    if own.deckCount < 3:
        deck_penalty -= 2.0
    elif own.deckCount < 6:
        deck_penalty -= 1.0
        
    if opp.deckCount < 3:
        deck_penalty += 2.0
    elif opp.deckCount < 6:
        deck_penalty += 1.0
        
    logit = prize_diff * 1.5 + hp_diff * 0.01 + energy_diff * 0.3 + deck_penalty
    
    x = logit
    if x < -10.0:
        x = -10.0
    elif x > 10.0:
        x = 10.0
    val = 1.0 / (1.0 + math.exp(-x))
    
    # Clamp evaluation between 0.05 and 0.95 so absolute win (1.0) and loss (0.0) are prioritized
    return 0.05 + 0.90 * val

def perform_mcts(obs_dict: dict, own_deck: list[int], time_limit_ms: float = 1200.0) -> list[int]:
    """
    Executes Information Set MCTS over multiple determinizations.
    Returns the best action option list.
    """
    start_time = time.perf_counter()
    obs: Observation = to_observation_class(obs_dict)
    
    if not obs.select or not obs.current:
        return []
        
    possible_actions = get_possible_actions(obs)
    if not possible_actions:
        return [0]
    if len(possible_actions) == 1:
        return list(possible_actions[0])

    # Initialize Python tree root
    root = MCTSNode()
    
    # Run iterations until time limit is exceeded
    iterations = 0
    while (time.perf_counter() - start_time) * 1000.0 < time_limit_ms:
        iterations += 1
        
        # 1. Determinization
        det = determinize_state(obs, own_deck)
        
        # Track spawned search IDs during this iteration for cleanup
        spawned_sids = []
        
        try:
            # Begin simulator search session
            search_state = search_begin(
                obs,
                your_deck=det["your_deck"],
                your_prize=det["your_prize"],
                opponent_deck=det["opponent_deck"],
                opponent_prize=det["opponent_prize"],
                opponent_hand=det["opponent_hand"],
                opponent_active=det["opponent_active"]
            )
            spawned_sids.append(search_state.searchId)
            
            # Selection Phase
            curr_node = root
            curr_search_state = search_state
            visited_path = [curr_node]
            own_idx = obs.current.yourIndex
            
            # Walk down the tree
            while curr_search_state.observation.select and curr_search_state.observation.current.result == -1:
                is_own_turn = (curr_search_state.observation.current.yourIndex == own_idx)
                is_main = (curr_search_state.observation.select.context == SelectContext.MAIN)
                
                if is_own_turn and is_main:
                    # Find available actions in search state
                    node_actions = get_possible_actions(curr_search_state.observation)
                    
                    # Check if we have untried actions at this node
                    untried = [act for act in node_actions if act not in curr_node.children]
                    
                    if untried:
                        # Expansion Phase: Choose untried action with the highest prior
                        chosen_action = max(untried, key=lambda act: get_action_prior(act, curr_search_state.observation))
                        prior_val = get_action_prior(chosen_action, curr_search_state.observation)
                        new_node = MCTSNode(parent=curr_node, action=chosen_action, prior=prior_val)
                        curr_node.children[chosen_action] = new_node
                        curr_node = new_node
                        visited_path.append(curr_node)
                        
                        # Advance search simulator
                        curr_search_state = search_step(curr_search_state.searchId, list(chosen_action))
                        spawned_sids.append(curr_search_state.searchId)
                        break
                    else:
                        # Selection Phase: Choose best child according to UCB1
                        if not curr_node.children:
                            break
                            
                        best_action = None
                        best_ucb = -1.0
                        for act, child in curr_node.children.items():
                            ucb = child.ucb1(curr_node.visits)
                            if ucb > best_ucb:
                                best_ucb = ucb
                                best_action = act
                                
                        if best_action is None:
                            break
                            
                        curr_node = curr_node.children[best_action]
                        visited_path.append(curr_node)
                        curr_search_state = search_step(curr_search_state.searchId, list(best_action))
                        spawned_sids.append(curr_search_state.searchId)
                else:
                    # Opponent's turn OR our own sub-decisions: simulate using rollout_agent without expanding tree nodes
                    action_indices = rollout_agent(curr_search_state.observation)
                    select_info = curr_search_state.observation.select
                    if select_info:
                        action_indices = action_indices[:select_info.maxCount]
                        if len(action_indices) < select_info.minCount:
                            action_indices = list(range(select_info.minCount))
                    curr_search_state = search_step(curr_search_state.searchId, action_indices)
                    spawned_sids.append(curr_search_state.searchId)
            
            # Rollout Phase
            rollout_state = curr_search_state
            rollout_depth = 0
            # Limit rollout depth to avoid infinite loops
            while rollout_state.observation.current and rollout_state.observation.current.result == -1 and rollout_depth < 60:
                rollout_depth += 1
                action_indices = rollout_agent(rollout_state.observation)
                
                # Verify min/max constraints
                select_info = rollout_state.observation.select
                if select_info:
                    action_indices = action_indices[:select_info.maxCount]
                    if len(action_indices) < select_info.minCount:
                        action_indices = list(range(select_info.minCount))
                        
                rollout_state = search_step(rollout_state.searchId, action_indices)
                spawned_sids.append(rollout_state.searchId)
                
            # Backpropagation Phase
            outcome = 0.5  # Draw fallback
            if rollout_state.observation.current:
                if rollout_state.observation.current.result != -1:
                    winner = rollout_state.observation.current.result
                    own_idx = obs.current.yourIndex
                    if winner == own_idx:
                        outcome = 1.0
                    elif winner == 1 - own_idx:
                        outcome = 0.0
                else:
                    outcome = evaluate_heuristic(rollout_state.observation, obs.current.yourIndex)
                    
            for node in visited_path:
                node.visits += 1
                node.wins += outcome
                
        except Exception as e:
            # Handle search/simulation failures gracefully
            pass
        finally:
            # Clean up all search states spawned during this iteration
            for sid in reversed(spawned_sids):
                try:
                    search_release(sid)
                except Exception:
                    pass
                    
            # Complete search session
            try:
                search_end()
            except Exception:
                pass

    # Select best action based on most visits (robust child)
    if not root.children:
        return [0]
        
    best_action = max(root.children.keys(), key=lambda act: root.children[act].visits)
    return list(best_action)
