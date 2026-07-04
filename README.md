# PTCG Agent Development Suite

Build and train a state-of-the-art AI agent (utilizing Information Set Monte Carlo Tree Search and custom-tuned Value/Policy MLP networks) to play the Pokémon Trading Card Game.

About competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview

Or https://github.com/Ducdata1808/PTCG_Agent/tree/main/about_competition


---

## 📁 Repository File Structure

```text
PTCG_Agent/
├── data/                      # External datasets and card images
│   └── card_images/           # High-resolution PNG files for deck visualization
├── decks/                     # Meta-deck CSV layouts
│   └── csv_file/              # Competitive archetype deck files (e.g. Alakazam.csv, Abomasnow.csv)
├── static/                    # Dashboard static assets
│   └── style.css              # Custom premium Glassmorphism layout styling
├── templates/                 # Dashboard web templates
│   └── index.html             # Main battle simulator frontend dashboard
├── submission/                # Final agent code & weights targeted for Kaggle submission
│   ├── main.py                # Main submission agent entrypoint
│   ├── deck.csv               # Currently active deck config
│   ├── EN_Card_Data.csv       # Card database (SDK format)
│   └── src/                   # Agent source code (core, search, MCTS, utils)
│       └── search/
│           ├── value_net_weights.json   # Value Network weight configuration
│           └── policy_net_weights.json  # Policy Network weight configuration
├── models/                    # Saved models directory
├── scripts/                   # Pipelines & debugging tools
│   ├── collect_data.py        # Self-play data collection engine
│   ├── train_value_net.py     # Value/Policy network training loop
│   ├── evaluate.py            # Local agent evaluation benchmark (V4 vs V1)
│   └── evaluate_v4_vs_v2.py   # Head-to-head MCTS benchmark (V4 vs V2)
├── tests/                     # Verification test suites
├── app.py                     # Flask web server for local dashboard simulation
├── manage_agent.py            # Unified pipeline manager CLI
├── requirements.txt           # Python package dependencies
├── report.md                  # Development research findings and version summaries
└── README.md                  # Project documentation & setup guide
```

---

## 🚀 Quick Start Pipeline

We provide a unified management script `manage_agent.py` in the root folder to handle all configuration, training pipelines, evaluations, and packaging.

### Step 0: Create virtual environment and download dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```

### Step 1: Select & Validate Your Deck
Choose your active training/submission deck from the meta-decks stored in `decks/csv_file/`:
```bash
python manage_agent.py select-deck Alakazam
```
*This copies the selected deck layout into `submission/deck.csv`.*

Next, validate the selected deck to ensure it complies with the competition constraints (60 cards, max 4 duplicates, basic Pokémon requirements, etc.):
```bash
python -m unittest tests/test_decks.py
```
Meta decks: https://ptcg-kaggle-meta.vercel.app/2026-06-21

### Step 2: Collect Data & Train Networks
Run the self-play match data collection and train the Value & Policy networks:
```bash
# Example: Run 20,000 self-play training games
python manage_agent.py train 20000
```
> [!IMPORTANT]
> **CPU Utilization Note:** Data collection leverages Python's multiprocessing pool to execute games in parallel. This step will fully utilize all available CPU cores to maximize performance speed.

> [!TIP]
> **Training Volume Recommendation:** Experiments show that **20,000 matches** represents the optimal sweet spot for deck-specific tuning. Increasing the training matches to 50,000 matches is unnecessary and does not improve targeted win rates (due to generalization trade-offs) while tripling the training time (~7.5 hours vs ~3 hours). See the [report.md](report.md) Section 6.5 for the full case study.

> [!WARNING]
> **CPU-Only Training Requirement:** This project **only supports training and inference on the CPU**. Running training or tree search on a GPU is unsupported and would actually degrade performance (more details: Section 3 on https://github.com/Ducdata1808/PTCG_Agent/edit/main/report.md).

This pipeline command automatically:
1. Simulates $N$ matches (agent vs. random decks) and writes outcomes to `data/self_play_data.jsonl`.
2. Fits the neural network MLPs on the generated samples.
3. Syncs the trained weights into `models/v2/src/search/` and `submission/src/search/`.
4. Packages the final assets into a submission-ready `submission.tar.gz` archive.

### Step 3: Evaluate Performance
Benchmark your trained agent (V4) against the Pure Heuristic baseline (V1) and the previous MCTS engine (V2) over 10 games:
```bash
python manage_agent.py evaluate
```
*This runs both the V4 vs. Heuristic and V4 vs. V2 local simulated match sets sequentially and displays the win rates and metrics.*

### Step 4: Create Submission File
To submit your agent to Kaggle, you need a compressed tarball containing the code, weights, and deck in the `submission/` directory.

* If you ran **Step 2**, the pipeline has **already packaged `submission.tar.gz` for you**.
* If you want to manually rebuild the package at any time (e.g. after modifying the active deck or code without retraining), run the following command from the project root:

```bash
tar -czf submission.tar.gz -C submission .
```


---

## 🖥️ Web Simulator Dashboard

We provide an interactive local web portal to configure and test matchups between V4 agents using different meta-decks.

### Running the Simulator Server
To start the dashboard locally, run the Flask server:
```bash
python app.py
```
Then, open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 🛠️ Debugging & Tracing Tools

For deep-diving into the agent's gameplay behavior and inspecting decision logs:

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

## 📖 AI Agent Development Progression
For detailed research findings, performance metrics, and win-rate logs across all versions, see the [report.md](report.md).
