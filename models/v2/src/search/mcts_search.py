import os
import sys
import time
import math
import random
import json
import numpy as np

# Resolve imports for local sub-directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.api import Observation, to_observation_class, search_begin, search_step, search_release, search_end, OptionType, SelectContext
from search.determinize import determinize_state
from search.rollout_policy import rollout_agent, get_card_id_from_option, db
from search.features import extract_features
from search.option_features import extract_option_features

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))
VALUE_NET_PATH = os.path.join(WEIGHTS_DIR, "value_net_weights.json")
POLICY_NET_PATH = os.path.join(WEIGHTS_DIR, "policy_net_weights.json")

class MLPNetwork:
    def __init__(self, weights_path):
        self.weights = []
        self.biases = []
        self.means = []
        self.stds = []
        self.loaded = False
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "r") as f:
                    data = json.load(f)
                self.means = np.array(data["means"], dtype=np.float32)
                self.stds = np.array(data["stds"], dtype=np.float32)
                self.weights = [np.array(w, dtype=np.float32) for w in data["weights"]]
                self.biases = [np.array(b, dtype=np.float32) for b in data["biases"]]
                self.loaded = True
            except Exception:
                pass

    def predict(self, features: list[float]) -> float:
        if not self.loaded:
            return 0.0
        x = np.array(features, dtype=np.float32)
        x = (x - self.means) / self.stds
        activation = x
        for i in range(len(self.weights) - 1):
            activation = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = np.maximum(0.0, activation)  # ReLU
        output = np.dot(activation, self.weights[-1]) + self.biases[-1]
        return float(output[0])

VALUE_NET = MLPNetwork(VALUE_NET_PATH)
POLICY_NET = MLPNetwork(POLICY_NET_PATH)

class MCTSNode:
    """A node in the IS-MCTS search tree using PUCT."""
    def __init__(self, parent=None, action=None, prior=1.0):
        self.parent = parent
        self.action = action  # Action tuple that led to this node
        self.children = {}    # Maps action tuple -> MCTSNode
        self.visits = 0
        self.wins = 0.0
        self.prior = prior
        self.priors = None    # Will store policy priors once evaluated

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

def get_action_priors_neural(node_actions: list[tuple[int, ...]], obs: Observation) -> dict[tuple[int, ...], float]:
    """Computes search priors for available actions using Policy Network softmax or heuristic fallback."""
    if not POLICY_NET.loaded:
        priors = {}
        for act in node_actions:
            priors[act] = get_action_prior(act, obs)
        s = sum(priors.values())
        if s > 0:
            for act in priors:
                priors[act] /= s
        return priors

    scores = []
    for act in node_actions:
        idx = act[0]
        if obs.select and idx < len(obs.select.option):
            opt = obs.select.option[idx]
            feats = extract_option_features(opt, obs)
            score = POLICY_NET.predict(feats)
            scores.append(score)
        else:
            scores.append(0.0)
            
    max_score = max(scores) if scores else 0.0
    exp_scores = [math.exp(sc - max_score) for sc in scores]
    sum_exp = sum(exp_scores)
    
    priors = {}
    for i, act in enumerate(node_actions):
        priors[act] = exp_scores[i] / sum_exp if sum_exp > 0 else 1.0 / len(node_actions)
    return priors

def get_possible_actions(obs: Observation) -> list[tuple[int, ...]]:
    """Generates valid action options (represented as tuples of indices)."""
    if not obs.select:
        return [()]
        
    num_options = len(obs.select.option)
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    
    actions = []
    if min_count == 1 and max_count == 1:
        for idx in range(num_options):
            actions.append((idx,))
        if not actions:
            for idx in range(num_options):
                actions.append((idx,))
    else:
        for idx in range(num_options):
            actions.append((idx,))
        if min_count <= num_options <= max_count:
            actions.append(tuple(range(num_options)))
        if min_count > 1 and min_count <= num_options:
            actions.append(tuple(range(min_count)))
            
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
    
    own_prizes = len(own.prize) if own.prize else 0
    opp_prizes = len(opp.prize) if opp.prize else 0
    prize_diff = opp_prizes - own_prizes
    
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
    
    return 0.05 + 0.90 * val

def perform_mcts(obs_dict: dict, own_deck: list[int], time_limit_ms: float = 1200.0) -> list[int]:
    start_time = time.perf_counter()
    obs: Observation = to_observation_class(obs_dict)
    
    if not obs.select or not obs.current:
        return []
        
    possible_actions = get_possible_actions(obs)
    if not possible_actions:
        return [0]
    if len(possible_actions) == 1:
        return list(possible_actions[0])

    root = MCTSNode()
    
    iterations = 0
    while (time.perf_counter() - start_time) * 1000.0 < time_limit_ms:
        iterations += 1
        
        det = determinize_state(obs, own_deck)
        spawned_sids = []
        
        try:
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
            
            curr_node = root
            curr_search_state = search_state
            visited_path = [curr_node]
            own_idx = obs.current.yourIndex
            
            while curr_search_state.observation.select and curr_search_state.observation.current.result == -1:
                is_own_turn = (curr_search_state.observation.current.yourIndex == own_idx)
                is_main = (curr_search_state.observation.select.context == SelectContext.MAIN)
                
                if is_own_turn and is_main:
                    node_actions = get_possible_actions(curr_search_state.observation)
                    
                    if curr_node.priors is None:
                        curr_node.priors = get_action_priors_neural(node_actions, curr_search_state.observation)
                        
                    untried = [act for act in node_actions if act not in curr_node.children]
                    
                    if untried:
                        chosen_action = max(untried, key=lambda act: curr_node.priors.get(act, 0.0))
                        prior_val = curr_node.priors.get(chosen_action, 1.0 / len(node_actions))
                        try:
                            curr_search_state = search_step(curr_search_state.searchId, list(chosen_action))
                            spawned_sids.append(curr_search_state.searchId)
                            
                            new_node = MCTSNode(parent=curr_node, action=chosen_action, prior=prior_val)
                            curr_node.children[chosen_action] = new_node
                            curr_node = new_node
                            visited_path.append(curr_node)
                        except Exception:
                            break
                        break
                    else:
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
                            
                        try:
                            curr_search_state = search_step(curr_search_state.searchId, list(best_action))
                            spawned_sids.append(curr_search_state.searchId)
                            curr_node = curr_node.children[best_action]
                            visited_path.append(curr_node)
                        except Exception:
                            break
                else:
                    action_indices = rollout_agent(curr_search_state.observation)
                    select_info = curr_search_state.observation.select
                    if select_info:
                        action_indices = action_indices[:select_info.maxCount]
                        if len(action_indices) < select_info.minCount:
                            action_indices = list(range(select_info.minCount))
                    try:
                        curr_search_state = search_step(curr_search_state.searchId, action_indices)
                        spawned_sids.append(curr_search_state.searchId)
                    except Exception:
                        break
            
            rollout_state = curr_search_state
            
            # Rollout phase: simulate using heuristic rollout policy until player's turn ends or game ends
            rollout_steps = 0
            while (rollout_state.observation.select and 
                   rollout_state.observation.current and 
                   rollout_state.observation.current.result == -1 and 
                   rollout_state.observation.current.yourIndex == own_idx and 
                   rollout_steps < 30):
                
                action_indices = rollout_agent(rollout_state.observation)
                select_info = rollout_state.observation.select
                if select_info:
                    action_indices = action_indices[:select_info.maxCount]
                    if len(action_indices) < select_info.minCount:
                        action_indices = list(range(select_info.minCount))
                
                try:
                    rollout_state = search_step(rollout_state.searchId, action_indices)
                    spawned_sids.append(rollout_state.searchId)
                except Exception:
                    break
                rollout_steps += 1
            
            outcome = 0.5
            if rollout_state.observation.current:
                if rollout_state.observation.current.result != -1:
                    winner = rollout_state.observation.current.result
                    own_idx = obs.current.yourIndex
                    if winner == own_idx:
                        outcome = 1.0
                    elif winner == 1 - own_idx:
                        outcome = 0.0
                else:
                    if VALUE_NET.loaded:
                        feats = extract_features(rollout_state.observation, obs.current.yourIndex)
                        outcome = min(max(VALUE_NET.predict(feats), 0.05), 0.95)
                    else:
                        outcome = evaluate_heuristic(rollout_state.observation, obs.current.yourIndex)
                    
            for node in visited_path:
                node.visits += 1
                node.wins += outcome
                
        except Exception as e:
            pass
        finally:
            for sid in reversed(spawned_sids):
                try:
                    search_release(sid)
                except Exception:
                    pass
            try:
                search_end()
            except Exception:
                pass

    if not root.children:
        return [0]
        
    best_action = max(root.children.keys(), key=lambda act: root.children[act].visits)
    return list(best_action)
