import os
import sys
import json
from kaggle_environments import make

# Add workspace root and submission to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)
sys.path.insert(0, os.path.join(workspace_dir, "submission"))

# Import V4 and V2 wrappers
from main import agent as v4_agent
from models.v2 import agent as v2_agent

def main():
    # Set deck environment variables to ensure both agents load the submission/deck.csv
    deck_path = os.path.abspath(os.path.join(workspace_dir, "submission", "deck.csv"))
    os.environ["AGENT_DECK_PATH"] = deck_path
    os.environ["AGENT0_DECK_PATH"] = deck_path
    os.environ["AGENT1_DECK_PATH"] = deck_path
    
    print(f"Using deck: {deck_path}")
    print("Initializing kaggle_environments 'cabt'...")
    env = make("cabt")
    
    print("Running simulation (V4 vs V2)...")
    env.run([v4_agent, v2_agent])
    
    # Output the visualization json
    output_path = os.path.join(workspace_dir, "vis.json")
    print(f"Saving visualization to {output_path}...")
    with open(output_path, "w") as file:
        json.dump(env.steps[0][0]["visualize"], file)
        
    print("Done! You can now open visualizer.html in your browser and upload vis.json to replay the match.")

if __name__ == "__main__":
    main()
