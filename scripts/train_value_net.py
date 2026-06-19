import json
import os
import sys
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Add root, submission, and submission/src to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
submission_dir = os.path.join(root_dir, "submission")
sys.path.insert(0, root_dir)
sys.path.insert(0, submission_dir)
sys.path.insert(0, os.path.join(submission_dir, "src"))

from search.option_features import extract_option_features
from search.features import extract_features

def main():
    data_path = "data/self_play_data.jsonl"
    if not os.path.exists(data_path):
        if os.path.exists("data/self_play_data.json"):
            data_path = "data/self_play_data.json"
        else:
            print(f"Error: data/self_play_data.jsonl not found. Run scripts/collect_data.py first.")
            return
        
    print("Loading self-play dataset...")
    if data_path.endswith(".jsonl"):
        # Count lines first to pre-allocate
        num_samples = 0
        with open(data_path, "r") as f:
            for _ in f:
                num_samples += 1
                
        print(f"Dataset contains {num_samples} samples. Pre-allocating memory...")
        # Resolve number of features dynamically from first line
        num_features = 42  # Default fallback
        with open(data_path, "r") as f:
            first_line = f.readline()
            if first_line:
                first_obj = json.loads(first_line)
                num_features = len(first_obj["features"])
                
        X_val = np.zeros((num_samples, num_features), dtype=np.float32)
        y_val = np.zeros(num_samples, dtype=np.float32)
        
        with open(data_path, "r") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                X_val[idx] = obj["features"]
                y_val[idx] = obj["label"]
    else:
        with open(data_path, "r") as f:
            samples = json.load(f)
        X_val = np.array([s["features"] for s in samples], dtype=np.float32)
        y_val = np.array([s["label"] for s in samples], dtype=np.float32)
    
    # Compute feature statistics for manual standardization
    means_val = np.mean(X_val, axis=0)
    stds_val = np.std(X_val, axis=0)
    stds_val[stds_val == 0.0] = 1.0  # Avoid division by zero
    
    X_val_scaled = (X_val - means_val) / stds_val
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_val_scaled, y_val, test_size=0.1, random_state=42)
    
    print(f"Training Value MLPRegressor on {X_train_val.shape[0]} samples...")
    mlp_val = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )
    mlp_val.fit(X_train_val, y_train_val)
    print(f"Value Net Train R^2: {mlp_val.score(X_train_val, y_train_val):.4f}, Test R^2: {mlp_val.score(X_test_val, y_test_val):.4f}")
    
    # Export Value Network weights
    weights_val = [w.tolist() for w in mlp_val.coefs_]
    biases_val = [b.tolist() for b in mlp_val.intercepts_]
    output_weights_val = {
        "means": means_val.tolist(),
        "stds": stds_val.tolist(),
        "weights": weights_val,
        "biases": biases_val
    }
    
    out_dir = "submission/src/search"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "value_net_weights.json"), "w") as f:
        json.dump(output_weights_val, f)
        
    # 2. Bootstrap Policy Network
    # We will generate option features and train the Policy Network to assign higher scores to actions that align with strategic targets.
    # To do this, we synthesize a dataset of option features:
    # - OptionType.PLAY/ATTACH/EVOLVE/ATTACK/ABILITY get higher preference values when they align with the strategic goals.
    # Let's generate a synthetic bootstrap dataset of option features (e.g. 50,000 samples)
    print("Generating bootstrap dataset for Policy Network...")
    X_pol = []
    y_pol = []
    
    # Generate positive and negative samples for option types
    for option_type in range(7):
        for hp_norm in [0.0, 0.3, 0.6, 0.9]:
            for energy_norm in [0.0, 0.2, 0.4, 0.6, 0.8]:
                for match_type in [0.0, 1.0]:
                    # Build feature vector of size 16
                    feat = [0.0] * 16
                    feat[option_type] = 1.0 # option type
                    feat[10] = 1.0 # Is Pokémon
                    feat[11] = hp_norm
                    feat[14] = energy_norm
                    feat[15] = match_type
                    
                    # Preference target: higher HP, higher energy, and matching type are preferred
                    # Especially attacks (type 4), evolutions (type 2), and energy attachments (type 1)
                    score = 0.1
                    if option_type == 4: # Attack
                        score = 0.9
                    elif option_type == 2: # Evolve
                        score = 0.8 + 0.1 * hp_norm
                    elif option_type == 1: # Attach
                        score = 0.5 + 0.4 * match_type + 0.1 * energy_norm
                    elif option_type == 0: # Play basic
                        score = 0.4 + 0.2 * hp_norm
                    elif option_type == 3: # Ability
                        score = 0.3
                    
                    X_pol.append(feat)
                    y_pol.append(score)
                    
    X_pol = np.array(X_pol, dtype=np.float32)
    y_pol = np.array(y_pol, dtype=np.float32)
    
    means_pol = np.mean(X_pol, axis=0)
    stds_pol = np.std(X_pol, axis=0)
    stds_pol[stds_pol == 0.0] = 1.0
    
    X_pol_scaled = (X_pol - means_pol) / stds_pol
    X_train_pol, X_test_pol, y_train_pol, y_test_pol = train_test_split(X_pol_scaled, y_pol, test_size=0.1, random_state=42)
    
    print(f"Training Policy MLPRegressor on {X_train_pol.shape[0]} samples...")
    mlp_pol = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp_pol.fit(X_train_pol, y_train_pol)
    print(f"Policy Net Train R^2: {mlp_pol.score(X_train_pol, y_train_pol):.4f}, Test R^2: {mlp_pol.score(X_test_pol, y_test_pol):.4f}")
    
    # Export Policy Network weights
    weights_pol = [w.tolist() for w in mlp_pol.coefs_]
    biases_pol = [b.tolist() for b in mlp_pol.intercepts_]
    output_weights_pol = {
        "means": means_pol.tolist(),
        "stds": stds_pol.tolist(),
        "weights": weights_pol,
        "biases": biases_pol
    }
    
    with open(os.path.join(out_dir, "policy_net_weights.json"), "w") as f:
        json.dump(output_weights_pol, f)
        
    print("Successfully trained both Value and Policy Networks!")

if __name__ == "__main__":
    main()
