
# 1. Version 1: Heuristic Agent

The agent plays matches using a prioritized rule system during the main phase and context-aware scoring for other game phases (setup, card searches, discards). 

## 1.1 Key features
### Dynamic Card ID Resolution
The simulation engine does not directly populate card IDs in the options list for many interaction types (e.g. playing cards, using stadium/bench abilities). The agent uses a custom resolution engine:
* **Implicit Hand Mapping**: Converts option indices to hand card IDs during `PLAY`, `ATTACH`, and `EVOLVE`.
* **Field/Zone Mapping**: Resolves card IDs for active, bench, discard, stadium, and deck search options using index-offset lookups against `obs.current` arrays.

### Smart Energy Attachment
Instead of attaching energy randomly, the agent evaluates all attachment actions using a scoring heuristic:
* **Type Matching**: Boosts options by `+10.0` if the energy type matches the target Pokémon's primary type (e.g., Grass Energy onto Tarountula).
* **Special Energy**: Prioritizes attaching `Team Rocket's Energy` (which provides dual Psychic/Darkness energy) to Team Rocket's Pokémon (`+15.0`).
* **Role Prioritization**: Favors the `Active` Pokémon (`+2.0`) and powerful `ex` attackers (`+5.0`).

### Deck-Out Safety Safeguards
In long matches or decks with heavy draw cards, the agent prevents self-deck-out by throttling actions:
* **Supporter & Tool Throttling**: Blocks playing major draw cards (`Lillie's Determination`, `Ariana`, and `Lucky Helmet`) if `deck_count < 15`.
* **Ability Throttling**: Disables drawing abilities (like `Team Rocket's Factory`) if `deck_count < 8`.
* **Attack Safety**: Filters out self-deck-discard attacks (like Abomasnow ex's `Hammer-lanche` which discards 6 cards) if `deck_count < 8` unless the attack deals enough damage to secure a match victory on the same turn.

### Smart Retreat Loop Prevention
Commented out the default retreat fallback which was causing the agent to retreat back and forth, wasting energy. The agent now only retreats if:
1. A benched Pokémon has strictly more energy attached than the active Pokémon, OR
2. The active Pokémon is completely incapable of attacking in this deck (e.g., Articuno in a deck with no Water energy), AND
3. The bench has a powered-up Pokémon to switch into.

---

## 1.2 Benchmark Evaluation Metrics

The baseline agent was benchmarked against the **Random Agent** over **50 games**:

### Aggregated Metrics
* **Win Rate**: **96.00%** (48 Wins, 2 Losses)
* **Average Turn Length**: **109.1 actions/game**
* **Errors / Timeouts**: **0**

### Victory Type Breakdown (Heuristic Agent)
* **Prize Out**: **45.8%** (22 games)
* **Bench Out**: **37.5%** (18 games)
* **Deck Out**: **16.7%** (8 games)

### Gameplay Averages (Per Game)
| Metric | Heuristic Agent | Random Agent |
|---|---|---|
| **Damage Dealt** | 1256.6 | 149.6 |
| **Cards Played** | 27.8 | 24.9 |
| **Energy Attached** | 18.2 | 19.4 |
| **Evolutions Done** | 3.0 | 2.6 |

### Decision Latency
* **Average Decision Time**: **0.12 ms**
* **Maximum Decision Time**: **0.69 ms**

# 2. Version 2.0 — Information Set Monte Carlo Tree Search (IS-MCTS) Agent
To prepare the agent for competitive matchmaking against human-engineered agents, we implemented an **IS-MCTS Search Agent** that runs lookahead simulations.
## 2.1 Key features and enhancements
* **State Determinization**: Samples hidden card lists (opponent's hand, deck, prizes, own prizes) and predicts opponent deck archetypes (Water vs. Rocket's Mewtwo).
* **PUCT (Predictor + UCB) Selection**: Directs the lookahead search using heuristic action priors to focus simulation resources on high-quality candidate moves.
* **Heuristic State Evaluation**: Resolves rollout draw cutoffs (at depth 60) by scoring states using an additive logit function (prizes difference, play area HP, energy attached).
* **MAIN Context Search**: Restricts MCTS node expansion solely to the high-level `SelectContext.MAIN` phase, resolving other game contexts (setup, discards, etc.) with the rule-based policy.
* **Terminal Attack Bypassing**: Immediately executes attacks when recommended by the heuristic to save search time.

### 2.2 Benchmark Evaluation Metrics

The IS-MCTS agent was evaluated directly against the **Pure Heuristic Agent (Version 1.0)** over **2 games** with a search budget of **800ms**:

* **Win Rate**: **100.00%** (2 Wins, 0 Losses)
* **Average Turn Length**: **82.0 actions/game** (more efficient, decisive game flow)
* **Errors / Timeouts**: **0**

#### Victory Type Breakdown
* **Deck Out** (Opponent ran out of cards): **50.0%** (1 game)
* **Bench Out** (Opponent has no Pokémon left): **50.0%** (1 game)

#### Gameplay Averages (Per Game)
| Metric | MCTS Search Agent (V2) | Pure Heuristic Agent (V1) |
|---|---|---|
| **Damage Dealt** | 915.0 | 550.0 |
| **Cards Played** | 5.0 | 26.0 |
| **Energy Attached** | 7.0 | 20.0 |
| **Evolutions Done** | 2.0 | 4.0 |

#### Decision Latency (MCTS Agent)
* **Average Decision Time**: **503.74 ms**
* **Maximum Decision Time**: **801.06 ms**
* *(Successfully operating within the competitive budget)*

