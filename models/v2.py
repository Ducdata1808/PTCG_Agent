import sys
import os

V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2")

def agent(obs_dict) -> list[int]:
    """Wrapper function to execute the Version 2 (IS-MCTS) agent with clean path injection."""
    orig_path = list(sys.path)
    # Inject V2 paths at the front
    sys.path.insert(0, V2_DIR)
    sys.path.insert(0, os.path.join(V2_DIR, "src"))
    try:
        from main import agent as v2_agent
        return v2_agent(obs_dict)
    finally:
        # Restore original path list to prevent collision
        sys.path = orig_path
