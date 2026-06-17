Dataset Description
Episode Replay
You can access episode replays for your Submissions from the Submissions tab, or you can download them via the CLI (or MCP) read more here: https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md

You can download replay files from other teams from the Leaderboard, and we will enable a daily episode export of the top rated episodes (to help BC/RL/IL). This will be posted in the competition forums.

Dataset Description
This competition provides card metadata and reference materials for the Pokémon Trading Card Game environment used in the simulator.

The dataset contains card identifiers, card names, expansion information, and reference images that correspond to the cards available in the competition environment. These files are provided to help participants understand the card pool and map card IDs used in the simulator to their corresponding card details.

Two versions of the card data are provided: an English version and a Japanese version. The content is identical except for the language of the card names and descriptions.

Files
The dataset includes the following files:

Card\_ID\_List\_EN.pdf - A reference document listing all cards available in the competition environment. Each entry includes the card ID, card name, expansion set, collection number, and an image of the card.

Card\_ID\_List\_JP.pdf - The Japanese version of the card reference document. The structure is identical to the English version, but card names and descriptions are written in Japanese.

EN Card Data.csv - A structured dataset containing metadata for each card in English.

JP Card Data.csv - The Japanese version of the card metadata dataset.

CSV File Structure
Both CSV files share the same schema and contain structured metadata for all cards used in the competition environment.

Card ID
Unique identifier for each card used by the simulator.

Card Name
The name of the card.

Expansion
The expansion set the card belongs to.

Collection No.
The card's collection number within the expansion.

Stage (Pokémon) / Type (Energy and Trainer)
Indicates the Pokémon evolution stage (Basic, Stage 1, Stage 2) or the card type for Energy and Trainer cards.

Rule
Special rule text associated with the card, if applicable.

Category
The category of the card (e.g., Pokémon, Trainer, Energy).

Previous stage
The previous evolution stage required for this Pokémon card.

HP
Hit Points of the Pokémon.

Type
Pokémon type (e.g., Grass, Fire, Water).

Weakness
Type weakness of the Pokémon.

Resistance (Type)
Type resistance of the Pokémon.

Retreat
Retreat cost required to switch the Pokémon.

Move Name
Name of the attack or move.

Cost
Energy cost required to use the move.

Damage
Damage dealt by the move.

Effect Explanation
Description of the move effect or additional rule text.