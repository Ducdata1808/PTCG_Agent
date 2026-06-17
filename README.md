# PTCG_Agent
Build an AI Training Agent to play the Pokémon Trading Card Game.

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
