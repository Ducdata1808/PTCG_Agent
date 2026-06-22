import os
import sys
import json
from flask import Flask, jsonify, request, render_template, send_from_directory

# Configure system path to import agent
workspace_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, workspace_dir)
sys.path.insert(0, os.path.join(workspace_dir, "submission"))

from main import agent as v4_agent

app = Flask(__name__)

# Directory of meta-decks
DECKS_DIR = os.path.join(workspace_dir, "decks", "csv_file")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/decks")
def get_decks():
    """List all available meta-decks."""
    try:
        decks = []
        if os.path.exists(DECKS_DIR):
            for file_name in os.listdir(DECKS_DIR):
                if file_name.endswith(".csv"):
                    decks.append(file_name[:-4])
        return jsonify({"success": True, "decks": sorted(decks)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/static/card_images/<path:filename>")
def card_image(filename):
    """Serve card images from the data/card_images directory."""
    return send_from_directory(os.path.join(workspace_dir, "data", "card_images"), filename)

@app.route("/api/deck-details/<deck_name>")
def get_deck_details(deck_name):
    """Get aggregated card metadata for a specific deck."""
    try:
        deck_path = os.path.join(DECKS_DIR, f"{deck_name}.csv")
        if not os.path.exists(deck_path):
            return jsonify({"success": False, "error": "Deck not found."}), 404
        
        from core.card_database import CardDatabase
        db = CardDatabase(os.path.join(workspace_dir, "data", "EN_Card_Data.csv"))
        
        card_ids = []
        with open(deck_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    card_ids.append(int(line))
                    
        from collections import Counter
        id_counts = Counter(card_ids)
        
        details = []
        for cid, count in id_counts.items():
            card = db.get_card(cid)
            if card:
                details.append({
                    "id": cid,
                    "name": card["name"],
                    "stage": card["stage"],
                    "type": card["type"],
                    "count": count
                })
        
        # Sort by stage (Pokémon vs Items vs Energy) then name
        details.sort(key=lambda x: (x["stage"], x["name"]))
        
        return jsonify({"success": True, "cards": details})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Simulate a match between two V4 agents with selected decks."""
    data = request.json or {}
    deck1_name = data.get("deck1")
    deck2_name = data.get("deck2")

    if not deck1_name or not deck2_name:
        return jsonify({"success": False, "error": "Both deck1 and deck2 are required."}), 400

    import random
    available_decks = []
    if os.path.exists(DECKS_DIR):
        for file_name in os.listdir(DECKS_DIR):
            if file_name.endswith(".csv"):
                available_decks.append(file_name[:-4])

    if not available_decks:
        return jsonify({"success": False, "error": "No available decks found."}), 400

    if deck1_name == "_random_":
        deck1_name = random.choice(available_decks)
    if deck2_name == "_random_":
        deck2_name = random.choice(available_decks)

    deck1_path = os.path.join(DECKS_DIR, f"{deck1_name}.csv")
    deck2_path = os.path.join(DECKS_DIR, f"{deck2_name}.csv")

    if not os.path.exists(deck1_path) or not os.path.exists(deck2_path):
        return jsonify({"success": False, "error": "One or both selected decks do not exist."}), 400

    try:
        from kaggle_environments import make

        # Set environment variables for the agents to read the correct decks
        os.environ["AGENT0_DECK_PATH"] = os.path.abspath(deck1_path)
        os.environ["AGENT1_DECK_PATH"] = os.path.abspath(deck2_path)
        os.environ["MCTS_TIME_LIMIT_MS"] = "150.0"

        # Initialize the Pokémon TCG Environment
        env = make("cabt")
        
        # Run simulation using V4 agent on both sides
        env.run([v4_agent, v4_agent])

        # Extract the visualization steps payload
        vis_payload = env.steps[0][0]["visualize"]

        return jsonify({
            "success": True,
            "visualize": vis_payload
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
