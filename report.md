# PTCG Heuristic Agent Version 1 — Performance Report

This report summarizes the design, features, and evaluation metrics of the baseline **Rule-Based Heuristic Agent** (Version 1.0) developed for the PTCG AI Battle Challenge.

---

## 1. Key Features & Design Architecture

The agent plays matches using a prioritized rule system during the main phase and context-aware scoring for other game phases (setup, card searches, discards). 

### 1.1 Dynamic Card ID Resolution
The simulation engine does not directly populate card IDs in the options list for many interaction types (e.g. playing cards, using stadium/bench abilities). The agent uses a custom resolution engine:
* **Implicit Hand Mapping**: Converts option indices to hand card IDs during `PLAY`, `ATTACH`, and `EVOLVE`.
* **Field/Zone Mapping**: Resolves card IDs for active, bench, discard, stadium, and deck search options using index-offset lookups against `obs.current` arrays.

### 1.2 Smart Energy Attachment
Instead of attaching energy randomly, the agent evaluates all attachment actions using a scoring heuristic:
* **Type Matching**: Boosts options by `+10.0` if the energy type matches the target Pokémon's primary type (e.g., Grass Energy onto Tarountula).
* **Special Energy**: Prioritizes attaching `Team Rocket's Energy` (which provides dual Psychic/Darkness energy) to Team Rocket's Pokémon (`+15.0`).
* **Role Prioritization**: Favors the `Active` Pokémon (`+2.0`) and powerful `ex` attackers (`+5.0`).

### 1.3 Deck-Out Safety Safeguards
In long matches or decks with heavy draw cards, the agent prevents self-deck-out by throttling actions:
* **Supporter & Tool Throttling**: Blocks playing major draw cards (`Lillie's Determination`, `Ariana`, and `Lucky Helmet`) if `deck_count < 15`.
* **Ability Throttling**: Disables drawing abilities (like `Team Rocket's Factory`) if `deck_count < 8`.
* **Attack Safety**: Filters out self-deck-discard attacks (like Abomasnow ex's `Hammer-lanche` which discards 6 cards) if `deck_count < 8` unless the attack deals enough damage to secure a match victory on the same turn.

### 1.4 Smart Retreat Loop Prevention
Commented out the default retreat fallback which was causing the agent to retreat back and forth, wasting energy. The agent now only retreats if:
1. A benched Pokémon has strictly more energy attached than the active Pokémon, OR
2. The active Pokémon is completely incapable of attacking in this deck (e.g., Articuno in a deck with no Water energy), AND
3. The bench has a powered-up Pokémon to switch into.

---

## 2. Benchmark Evaluation Metrics

The agent was benchmarked against the **Random Agent** over **50 games** using the default evaluation harness:

### 2.1 Aggregated Metrics
* **Win Rate**: **96.00%** (48 Wins, 2 Losses)
* **Average Turn Length**: **109.1 actions/game**
* **Errors / Timeouts**: **0**

### 2.2 Victory Type Breakdown (Heuristic Agent)
* **Prize Out** (Took all prize cards): **45.8%** (22 games)
* **Bench Out** (Opponent has no active/bench Pokémon left): **37.5%** (18 games)
* **Deck Out** (Opponent starts turn with 0 cards in deck): **16.7%** (8 games)

### 2.3 Gameplay Averages (Per Game)
| Metric | Heuristic Agent | Random Agent |
|---|---|---|
| **Damage Dealt** | 1256.6 | 149.6 |
| **Cards Played** | 27.8 | 24.9 |
| **Energy Attached** | 18.2 | 19.4 |
| **Evolutions Done** | 3.0 | 2.6 |

### 2.4 Decision Latency
* **Average Decision Time**: **0.12 ms**
* **Maximum Decision Time**: **0.69 ms**
* *(Both well within the competition limits of 10,000ms/turn)*

---

## 3. Findings & Next Steps

* **Randomness Ceiling**: A 96.00% win rate is the practical maximum for a rule-based agent against a random opponent due to unavoidable RNG factors (e.g., drawing no basic Pokémon in the opening hand and benched out via mulligans, or unfortunate prize card locks).
* **Search Integration**: To progress further and prepare for competitive match-making, the next phase will integrate the C++ search API (`search_begin` / `search_step`) to perform Monte Carlo Tree Search (MCTS) lookahead.
