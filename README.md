# PTCG_Agent
Build an AI Training Agent to play the Pokémon Trading Card Game.

## Local Evaluation & Testing

You can evaluate your agent's decision-making locally by simulating multiple games against a benchmark opponent (e.g. a random agent).

### Running the Evaluation Benchmark
To run the integrated evaluation benchmark:

```bash
python scripts/evaluate.py
```

This tool simulates 10 games, alternating first/second turn order, and outputs:
1. **Win Rate**: The percentage of matches won.
2. **Victory Type Breakdown**: How many matches were won by taking all prize cards (Prize Out) vs. depleting the opponent's deck (Deck Out) or bench (Bench Out).
3. **Average Game Metrics**: Comparison of average damage dealt, cards played, energy attached, and evolutions completed per game.
4. **Decision Timings**: Average and maximum decision latencies to verify timing performance constraints.

---

## Deck Validation Tests

Whenever you add a new deck CSV file in the `decks/` folder, you can run the dynamic deck validation tests to ensure the deck matches the game's submission rules.

### How it Works
The validation test suite checks every `.csv` file in the `decks/` folder (as well as `data/sample_submission/deck.csv`) against the following constraints:
1. Must contain **exactly 60 cards**.
2. Every Card ID must exist in the card database (`EN_Card_Data.csv`).
3. Must contain **at least 1 Basic Pokémon**.
4. No card (by name) is duplicated more than **4 times** (excluding Basic Energy).
5. Must contain at most **1 ACE SPEC** card.

### Running the Tests
To run the deck validation tests, execute the following command from the project root directory:

```bash
python -m unittest tests/test_decks.py
```

### Running All Tests
To run all tests (including database lookup tests):

```bash
python -m unittest discover tests/
```

---

## Debugging & Tracing Tools

For deep-diving into the agent's gameplay behavior and fixing edge cases, the following debugging tools are available in the `scripts/` folder:

### 1. Game Tracing (`scripts/trace_game.py`)
To watch a single full game play out step-by-step with raw logs and option choices:
```bash
python scripts/trace_game.py
```

### 2. Loss Trace Finder (`scripts/find_losing_trace.py`)
To automatically run up to 50 simulated games, find the first game the heuristic agent loses, and print the last 30 turns of the decision history leading to the defeat:
```bash
python scripts/find_losing_trace.py
```

---

## Performance Reports & Walkthroughs

The design, features, and benchmark metrics for both versions of the agent are documented in:
* [report.md](file:///c:/Users/Admin/Documents/viet_code/python/PTCG_Agent/report.md): Specifications and 50-game benchmark metrics for the baseline **Rule-Based Heuristic Agent (V1.0)**.
* [walkthrough.md](file:///C:/Users/Admin/.gemini/antigravity-ide/brain/737e6ed2-d9c7-4cde-b81e-e27fd373d5ac/walkthrough.md): Comprehensive implementation details and 10-game benchmark metrics for the **Phase 2 IS-MCTS Search Agent** (featuring PUCT node selection, state determinization, and heuristic state evaluations).


