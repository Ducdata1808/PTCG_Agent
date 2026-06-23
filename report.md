
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
## 1.3 Result on kaggle
- win 0/3 matches with Mewtwo deck
- Score: 359-600

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

## 2.3 Result on Kaggle
- Mewtwo deck: win 3/12 matches, score 434 - 719
- Mega Lucario ex deck: win 7/21 matches, score 473 - 853
---

# 3. Version 3.0 — AlphaGo/AlphaZero Hybrid Neural MCTS Agent

To resolve performance limitations of single-value network heuristics, we transitioned the search engine to an AlphaGo/AlphaZero-style architecture. This cooperative design uses two neural networks:
1. **Value Network**: A Multi-Layer Perceptron (MLP) mapping 20 state features to the expected match outcome ($z \in [0.05, 0.95]$).
2. **Policy Network**: A Policy MLP mapping 16 action option features to MCTS visit frequency targets, supplying dynamically predicted prior probabilities for PUCT selection.

> [!WARNING]
> **CPU-Only Training Requirement:** This project **only supports training and inference on the CPU**. Running training or tree search on a GPU is unsupported and would actually degrade performance due to three main reasons:
> 1. **Tiny Model Architecture & Data Transfer Overhead:** The Value Network (`(64, 32)`) and Policy Network (`(32, 16)`) are extremely lightweight (under 10,000 parameters total). The math calculations take fractions of a millisecond on a CPU. Copying small tensors back and forth between System RAM and GPU VRAM over the PCIe bus introduces latency that is significantly larger than the execution time itself.
> 2. **Kaggle Sandbox Constraints & Compatibility:** The Kaggle competition submission environment runs agents in a CPU-only sandbox with strict library limitations. To ensure 100% compatibility and zero external dependencies (no PyTorch/TensorFlow imports), the trained model weights are saved as JSON, and the feedforward inference is executed via pure NumPy on the CPU inside [main.py](file:///c:/Users/Admin/Documents/viet_code/python/PTCG_Agent/submission/main.py).
> 3. **Sequential Game Logic & MCTS Bottlenecks:** Simulating match state transitions (evaluating card effects, updating HP, shuffling, drawing cards) and MCTS tree traversals are sequential logic tasks. They cannot be parallelized on a GPU. Instead, throughput scaling is achieved by executing games in parallel across multiple CPU cores using Python's multiprocessing pool.

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

We evaluated V3 (AlphaGo Neural MCTS) directly against V2 (Heuristic MCTS, with weights disabled) over **10 games** (Both V3 and V2 use the same submission deck)
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
 
## 3.3 Result on Kaggle
- Mega Lucario ex deck: win 11/29 matches, score 420 - 728
- Mega Lucario ex 2 deck: win 10/ 27, score 483 - 730
---

# 4. Version 4.0 — Meta-Gaming and Adaptive Time Management Agent, Mega Lucario

To finalize the pipeline for Kaggle competition matchmaking, we implemented two strategic updates representing Phase 4:
1. **Dynamic Meta-Gaming / Archetype Guesser**: Matches visible cards in play (active, bench, discard) against **19 pre-compiled meta-decks** to dynamically predict the opponent's exact 60-card list during search.
2. **Adaptive Time Management**: Scales the MCTS time limit per decision based on move complexity (e.g. 200ms/400ms when choice space is constrained) and monitors the total match time bank to trigger emergency heuristics cutoffs, preventing timeouts under Kaggle rules.

## 4.1 Architecture & Training
* **Self-Play Match Collection**: Simulated **50,000 matches** of high-quality search play with dynamic deck mapping enabled on **18 CPU cores**. Unlike previous runs, the 50,000 matches were played using a wide variety of random competitive meta-decks for both players to force the Value and Policy networks to learn generic board representation features.
* **Dataset & Optimization**: Gathered **11,415,218 state-outcome pairs** saved progressively using a memory-optimized `.jsonl` streaming pipeline to prevent OOM errors.
* **Value MLP Performance (V4)**:
  * Train $R^2$: **0.2188**
  * Test $R^2$: **0.2189** (validated on 11.4 million samples)
* **Policy MLP Performance (V4)**:
  * Train $R^2$: **0.9912**
  * Test $R^2$: **0.9767**

### 4.1.1 Analysis of Early Training Issues
Despite the massive scale of the 50,000 matches training, early iterations of the V4 agent showed suboptimal decision-making in specific competitive scenarios (e.g., choosing to attach energy to a basic Riolu and passing while hoarding hand exchange/draw cards). This occurred because:
1. **Misaligned Policy Priors:** The Policy Network's priors heavily favored raw energy attachments and basic play actions, leading the PUCT formula to ignore hand-exchange or draw cards in the MCTS tree.
2. **Missing Rollout Turn-Resolution:** The MCTS rollout phase did not properly resolve full turn boundaries, which made passive actions appear safer to the Value Network than they actually were.
3. **Restricted Action Space:** Heuristic filters originally locked out critical trainer options during node selection.
These issues were fully resolved by fixing the Policy bootstrap targets, restoring turn-resolution rollouts, and opening up the action space.


## 4.2 Benchmark Results (Fixed V4 vs Heuristic Agent V1)

Following core fixes to the Policy Network bootstrap targets (to correctly prioritize Trainer/Supporter cards), restoration of the turn-resolution rollout phase in MCTS, and opening of the action space, the Version 4.0 agent was benchmarked directly against the **Pure Heuristic Agent (Version 1.0)** over **10 games**:

* **Fixed V4 Win Rate**: **70.00%** (7 Wins, 3 Losses)
* **Heuristic V1 Win Rate**: **30.00%** (3 Wins)
* **Errors / Timeouts**: **0** (Stochastic MCTS simulation mismatches handled gracefully)
* **Average Turn Length**: **82.8 actions/game** (highly efficient, decisive game flow)

### Victory Type Breakdown (Fixed V4)
* **Bench Out**: **57.1%** (4 games)
* **Deck Out**: **28.6%** (2 games)
* **Prize Out**: **14.3%** (1 game)

### Gameplay Averages (Per Game)
| Metric | Fixed V4 Agent | Heuristic Agent (V1) |
|---|---|---|
| **Damage Dealt** | 849.0 | 602.0 |
| **Cards Played** | 14.7 | 26.1 |
| **Energy Attached** | 9.5 | 10.9 |
| **Evolutions Done** | 2.1 | 3.0 |

### Decision Latency (Fixed V4)
* **Average Decision Time**: **426.02 ms**
* **Maximum Decision Time**: **841.74 ms**
* *(Successfully operating within Kaggle's competitive budget)*

### Head-to-Head Benchmark Results (Fixed V4 vs V2)

We also evaluated the fixed Version 4.0 agent directly against the **V2 Search Agent (IS-MCTS with heuristics, weights disabled)** over **10 games**:

* **Fixed V4 Win Rate**: **40.00%** (4 Wins, 6 Losses)
* **V2 Win Rate**: **60.00%** (6 Wins)
* **Errors / Timeouts**: **0**
* **Average Turn Length**: **80.9 actions/game**

#### Victory Type Breakdown (Fixed V4)
* **Prize Out**: **75.0%** (3 games)
* **Bench Out**: **25.0%** (1 game)

#### Gameplay Averages (Per Game)
| Metric | Fixed V4 Agent | V2 Search Agent |
|---|---|---|
| **Damage Dealt** | 808.0 | 998.0 |
| **Cards Played** | 21.8 | 15.4 |
| **Energy Attached** | 11.7 | 8.7 |
| **Evolutions Done** | 2.0 | 2.1 |

#### Decision Latency
* **V4 Agent**: Avg = **322.27 ms**, Max = **811.30 ms**
* **V2 Agent**: Avg = **373.66 ms**, Max = **808.36 ms**

## 4.3 Result on Kaggle
- Mega Lucario ex 2 (20000 training matches): win 7/20 matches, score 501 - 674
- Mega Lucario ex 2 (50000 training matches): win 14/31 matches, score 410 - 600

# 5. Version 5.0 — Mega Abomasnow Tuned MCTS Agent

Following the decision to update the default submission deck to the Mega Abomasnow deck list, the Value and Policy networks were retrained on a new self-play dataset of **20,000 matches** (2.6 million state-outcome samples) where V4 strictly used the Mega Abomasnow deck, while the V2 opponent played random decks from the meta-deck database.

## 5.1 Architecture & Training
* **Dataset Size**: **2,637,394 samples**
* **Value MLP Performance (Value Net R²)**:
  * Train $R^2$: **0.4126**
  * Test $R^2$: **0.4086** (significant increase in accuracy due to focusing on a single stable deck layout)
* **Policy MLP Performance**:
  * Train $R^2$: **0.9990**
  * Test $R^2$: **0.9993**

## 5.2 Benchmark Results (V5 vs Heuristic Agent V1)

We evaluated the newly trained V5 agent directly against the **Pure Heuristic Agent (Version 1.0)** over **10 games**:

* **V5 Win Rate**: **90.00%** (9 Wins, 1 Loss)
* **Heuristic V1 Win Rate**: **10.00%** (1 Loss)
* **Errors / Timeouts**: **0**
* **Average Turn Length**: **44.9 actions/game** (reflects Abomasnow's fast, high-damage gameplay)

### Gameplay Averages (Per Game)
| Metric | V5 Agent (Abomasnow) | Heuristic Agent (V1) |
|---|---|---|
| **Damage Dealt** | 1124.0 | 469.0 |
| **Cards Played** | 6.3 | 8.3 |
| **Energy Attached** | 7.2 | 8.3 |
| **Evolutions Done** | 1.3 | 2.4 |

### Decision Latency (V5)
* **Average Decision Time**: **385.43 ms**
* **Maximum Decision Time**: **813.04 ms**

## 5.3 Head-to-Head Benchmark Results (V5 vs V2)

We also ran a head-to-head MCTS vs MCTS evaluation over **10 games**:

* **V5 Win Rate**: **40.00%** (4 Wins, 6 Losses)
* **V2 Win Rate**: **60.00%** (6 Wins)
* **Errors / Timeouts**: **0**
* **Average Turn Length**: **43.4 actions/game**

### Gameplay Averages (Per Game)
| Metric | V5 Agent (Abomasnow) | V2 Search Agent |
|---|---|---|
| **Damage Dealt** | 668.0 | 685.0 |
| **Cards Played** | 6.0 | 7.2 |
| **Energy Attached** | 7.7 | 8.7 |
| **Evolutions Done** | 0.8 | 1.0 |

## 5.4 Result on Kaggle
- Win: 30/66 matches, score 478 - 770, final score 650
---

# 6. Version 6.0 — Alakazam Tuned MCTS Agent

Following the switch to the **Alakazam deck** in the submission folder, the networks were retrained on a fresh dataset of **20,000 matches** (MCTS V4 playing Alakazam vs V2 playing random decks).

## 6.1 Architecture & Training
* **Dataset Size**: **4,482,136 samples**
* **Value MLP Performance**:
  * Train $R^2$: **0.1962**
  * Test $R^2$: **0.1946**
* **Policy MLP Performance**:
  * Train $R^2$: **0.9990**
  * Test $R^2$: **0.9993**

## 6.2 Benchmark Results (Alakazam V6 vs Heuristic V1)
Over **10 games** using the Alakazam deck:
* **Alakazam V6 Win Rate**: **100.00%** (10 Wins, 0 Losses)
* **Heuristic V1 Win Rate**: **0.00%** (0 Wins)
* **Average Turn Length**: **125.0 actions/game**
* **Victory Type Breakdown**:
  * Deck Out: **40.0%** (4 games)
  * Prize Out: **30.0%** (3 games)
  * Bench Out: **30.0%** (3 games)
* **Gameplay Averages**:
  * Damage Dealt: **2236.0** (V6) vs **652.0** (V1)
  * Cards Played: **18.1** (V6) vs **30.4** (V1)
  * Energy Attached: **9.4** (V6) vs **15.4** (V1)
  * Evolutions Done: **7.0** (V6) vs **12.0** (V1)

## 6.3 Benchmark Results (Alakazam V6 vs MCTS V2)
Over **10 games** using the Alakazam deck:
* **Alakazam V6 Win Rate**: **90.00%** (9 Wins, 1 Loss)
* **MCTS V2 Win Rate**: **10.00%** (1 Win)
* **Average Turn Length**: **146.9 actions/game**
* **Victory Type Breakdown**:
  * Prize Out: **66.7%** (6 games)
  * Deck Out: **22.2%** (2 games)
  * Bench Out: **11.1%** (1 game)
* **Gameplay Averages**:
  * Damage Dealt: **3050.0** (V6) vs **1790.0** (V2)
  * Cards Played: **20.8** (V6) vs **32.5** (V2)
  * Energy Attached: **15.3** (V6) vs **15.5** (V2)
  * Evolutions Done: **11.1** (V6) vs **11.6** (V2)
* **Decision Latency**:
  * **V6 Agent**: Avg = **254.99 ms**, Max = **848.46 ms**
  * **V2 Agent**: Avg = **271.30 ms**, Max = **812.47 ms**

## 6.4 Notice: Ineffectiveness of Hardcoded Deck-Specific Heuristics
A key finding during Version 6 development was that **manually coding deck-specific heuristic rules is unnecessary and often counterproductive**. The Neural MCTS search naturally learns optimal card-hoarding and board-state value estimations without hand-crafted rules.

### Case Study: Capping Bench Size for Alakazam
Because Alakazam's attack scales with hand size, we attempted to hardcode an Alakazam-specific heuristic to conserve hand cards:
* **Rule:** Capped the bench size to a maximum of 2 basic Pokémon and blocked playing non-essential item cards.
* **Intended Goal:** Keep cards in hand to deal maximum damage.
* **Actual Result:** The agent's win rate against MCTS V2 **dropped from 90% down to 30%**.

### Why it failed:
1. **Draw Engine Deprivation:** Capping the bench prevented the agent from setting up essential draw engines such as **Dudunsparce**, **Genesect**, and **Fezandipiti ex**. Without these benched engines drawing cards, the agent's hand size actually ended up smaller in the mid-to-late game.
2. **State Distribution Shift:** Forcing artificial constraints on the agent's actions moved the board state outside the distribution that the Value Network was trained on, causing incorrect lookahead evaluations.
3. **Conclusion:** Allowing the MCTS search and Neural Networks to naturally balance board setup and hand conservation is far superior to hardcoded heuristic strategies.

## 6.5 Training Volume and Diminishing Returns (20k vs. 50k Matches)
To test if scaling up training yields higher performance, we retrained the Alakazam MCTS V4 networks on an expanded dataset of **50,000 matches** (generating **11,094,822 samples** over **7.5 hours** of CPU run time). 

The results showed that **increasing the training matches beyond 20,000 is unnecessary**:
* **20,000 Matches:** Achieved a **90% win rate** against both Heuristic V1 and MCTS V2.
* **50,000 Matches:** Achieved a **60% win rate** against Heuristic V1 and a **50% win rate** against MCTS V2.

### Why 50,000 matches did not improve target performance:
1. **Generalization Trade-off:** The 50,000-match pipeline trains against a diverse set of random meta-decks. While this makes the model highly robust against the entire pool of 19 competitive meta-decks, it dilutes the deck-specific optimization for the Alakazam mirror matchup (which is used in local benchmarks).
2. **Diminishing Returns on Value Net R²:** The Value Network R² remained stable at **0.1934** (compared to 0.1946 at 20k matches), showing that additional training volume did not significantly improve the neural network's capacity to estimate states.
3. **Conclusion:** A budget of **20,000 matches** represents the optimal sweet spot for deck-specific tuning, providing maximum targeted performance with significantly shorter training times (~3 hours vs ~7.5 hours).

## 6.6 Result on Kaggle
20000 training matches: win 22/47 matches, score 518 - 773, final score 646
50000 training mtaches: win 11/25 matches, score 487 - 678, final score 526


# 7. Version 7.0 — Hop Trevenant Tuned MCTS Agent

Following the switch to the **Hop Trevenant deck** in the submission folder, the networks were retrained on a fresh dataset of **20,000 matches** (MCTS V4 playing Hop Trevenant vs V2 playing random decks).

## 7.1 Architecture & Training
* **Dataset Size**: **4,444,830 samples**
* **Value MLP Performance**:
  * Train $R^2$: **0.2160**
  * Test $R^2$: **0.2139**
* **Policy MLP Performance**:
  * Train $R^2$: **0.9990**
  * Test $R^2$: **0.9993**

## 7.2 Benchmark Results (Hop Trevenant V7 vs Heuristic V1)
Over **10 games** using the Hop Trevenant deck:
* **Hop Trevenant V7 Win Rate**: **80.00%** (8 Wins, 2 Losses)
* **Heuristic V1 Win Rate**: **20.00%** (2 Losses)
* **Average Turn Length**: **111.6 actions/game**
* **Victory Type Breakdown**:
  * Prize Out: **87.5%** (7 games)
  * Bench Out: **12.5%** (1 game)
* **Gameplay Averages**:
  * Damage Dealt: **1273.0** (V7) vs **1104.0** (V1)
  * Cards Played: **21.8** (V7) vs **36.4** (V1)
  * Energy Attached: **11.7** (V7) vs **15.1** (V1)
  * Evolutions Done: **4.6** (V7) vs **5.4** (V1)

## 7.3 Benchmark Results (Hop Trevenant V7 vs MCTS V2)
Over **10 games** using the Hop Trevenant deck:
* **Hop Trevenant V7 Win Rate**: **50.00%** (5 Wins, 5 Losses)
* **MCTS V2 Win Rate**: **50.00%** (5 Wins)
* **Average Turn Length**: **113.7 actions/game**
* **Victory Type Breakdown**:
  * Prize Out: **80.0%** (4 games)
  * Bench Out: **20.0%** (1 game)
* **Gameplay Averages**:
  * Damage Dealt: **1251.0** (V7) vs **1192.0** (V2)
  * Cards Played: **26.9** (V7) vs **23.7** (V2)
  * Energy Attached: **15.1** (V7) vs **15.1** (V2)
  * Evolutions Done: **3.7** (V7) vs **5.2** (V2)
* **Decision Latency**:
  * **V7 Agent**: Avg = **297.59 ms**, Max = **815.04 ms**
  * **V2 Agent**: Avg = **299.84 ms**, Max = **867.65 ms**

 ## 7.4 Result on Kaggle
 
