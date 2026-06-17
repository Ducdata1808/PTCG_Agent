# PTCG_Agent
Build an AI Training Agent to play the Pokémon Trading Card Game.

## Project Setup

Install dependencies for local development and analysis:

```bash
pip install -r requirements.txt
```

---

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
