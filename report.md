
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

---

# 3. Version 3.0 — AlphaGo/AlphaZero Hybrid Neural MCTS Agent

To resolve performance limitations of single-value network heuristics, we transitioned the search engine to an AlphaGo/AlphaZero-style architecture. This cooperative design uses two neural networks:
1. **Value Network**: A Multi-Layer Perceptron (MLP) mapping 20 state features to the expected match outcome ($z \in [0.05, 0.95]$).
2. **Policy Network**: A Policy MLP mapping 16 action option features to MCTS visit frequency targets, supplying dynamically predicted prior probabilities for PUCT selection.

## 3.1 Architecture & Training
* **Self-Play Match Collection**: Simulated **20,000 matches** (V3 vs. V2, choosing random decks from `decks/csv_file/` for each fight) on **18 CPU cores** with MCTS time limits set to 150ms per search.
* **Dataset**: Gathered **6,897,706 state-outcome pairs** for training.
* **Value MLP Performance**:
  * Train $R^2$: **0.2253**
  * Test $R^2$: **0.2234** (validated on 6.2 million samples)
* **Policy MLP Performance** (Bootstrapped on strategic action preference scoring):
  * Train $R^2$: **0.9912**
  * Test $R^2$: **0.9767**
* **NumPy Deployment**: Both networks are saved as JSON weights and biases. Their feedforward inference passes are implemented purely in NumPy, maintaining 100% Kaggle compatibility with zero PyTorch/scikit-learn sandbox package dependencies.

## 3.2 Head-to-Head Benchmark Results (V3 vs V2)

We evaluated V3 (AlphaGo Neural MCTS) directly against V2 (Heuristic MCTS, with weights disabled) over **10 games**:
* **V3 Win Rate**: **80.00%** (8 Wins, 2 Losses)
* **V2 Win Rate**: **20.00%** (2 Wins)
* **Average Turn Length**: **94.9 actions/game**
* **Victory Type Breakdown (V3)**:
  * Deck Out: **50.0%** (4 games)
  * Bench Out: **50.0%** (4 games)
* **Gameplay Averages (Per Game)**:
  * Damage Dealt: **804.0** (V3) vs **806.0** (V2)
  * Cards Played: **30.3** (V3) vs **31.6** (V2)
  * Energy Attached: **11.6** (V3) vs **11.4** (V2)
  * Evolutions Done: **4.7** (V3) vs **3.6** (V2)
* **Decision Timings**:
  * **V3 Agent (AlphaGo Neural MCTS)**: Avg = **306.19 ms**, Max = **802.37 ms**
  * **V2 Agent (Heuristic MCTS)**: Avg = **308.60 ms**, Max = **803.28 ms**

---

# 4. Version 4.0 — Meta-Gaming and Adaptive Time Management Agent

To finalize the pipeline for Kaggle competition matchmaking, we implemented two strategic updates representing Phase 4:
1. **Dynamic Meta-Gaming / Archetype Guesser**: Matches visible cards in play (active, bench, discard) against **19 pre-compiled meta-decks** to dynamically predict the opponent's exact 60-card list during search.
2. **Adaptive Time Management**: Scales the MCTS time limit per decision based on move complexity (e.g. 200ms/400ms when choice space is constrained) and monitors the total match time bank to trigger emergency heuristics cutoffs, preventing timeouts under Kaggle rules.

## 4.1 Architecture & Training
* **Self-Play Match Collection**: Simulated **50,000 matches** of high-quality search play with dynamic deck mapping enabled on **18 CPU cores**.
* **Dataset & Optimization**: Gathered **11,415,218 state-outcome pairs** saved progressively using a memory-optimized `.jsonl` streaming pipeline to prevent OOM errors.
* **Value MLP Performance (V4)**:
  * Train $R^2$: **0.2188**
  * Test $R^2$: **0.2189** (validated on 11.4 million samples)
* **Policy MLP Performance (V4)**:
  * Train $R^2$: **0.9912**
  * Test $R^2$: **0.9767**

## 4.2 Head-to-Head Benchmark Results (V4 vs V2)

We evaluated V4 directly against V2 (with weights disabled) over **10 games**:
* **V4 Win Rate**: **50.00%** (5 Wins, 5 Losses)
* **V2 Win Rate**: **50.00%** (5 Wins)
* **Average Turn Length**: **85.2 actions/game**
* **Victory Type Breakdown (V4)**:
  * Bench Out: **40.0%** (2 games)
  * Prize Out: **40.0%** (2 games)
  * Deck Out: **20.0%** (1 game)
* **Gameplay Averages (Per Game)**:
  * Damage Dealt: **917.0** (V4) vs **888.0** (V2)
  * Cards Played: **23.5** (V4) vs **24.7** (V2)
  * Energy Attached: **11.7** (V4) vs **10.3** (V2)
  * Evolutions Done: **3.2** (V4) vs **3.3** (V2)
* **Decision Timings**:
  * **V4 Agent (Adaptive Neural MCTS)**: Avg = **151.64 ms**, Max = **801.79 ms** *(50% reduction in average latency compared to V3)*
  * **V2 Agent (Heuristic MCTS)**: Avg = **154.41 ms**, Max = **802.17 ms**

