import csv
import os

class CardDatabase:
    """In-memory database for looking up Pokémon TCG card metadata by Card ID."""
    
    def __init__(self, csv_path=None):
        if csv_path is None:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                current_dir = "/kaggle_simulations/agent"
                if not os.path.exists(current_dir):
                    current_dir = os.getcwd()
            
            candidates = [
                os.path.join(current_dir, "EN_Card_Data.csv"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), "EN_Card_Data.csv"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))), "data", "EN_Card_Data.csv"),
                "/kaggle_simulations/agent/EN_Card_Data.csv",
                os.path.join(os.getcwd(), "data", "EN_Card_Data.csv"),
                os.path.join(os.getcwd(), "EN_Card_Data.csv"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path is None:
                csv_path = "EN_Card_Data.csv"  # Fallback to current working directory default
            
        self.cards = {}
        self.load_database(csv_path)

    def load_database(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Card database file not found at {csv_path}")
            
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    card_id = int(row['Card ID'])
                    # Group multiple moves/attacks for the same card
                    if card_id not in self.cards:
                        self.cards[card_id] = {
                            'id': card_id,
                            'name': row['Card Name'],
                            'expansion': row['Expansion'],
                            'collection_no': row['Collection No.'],
                            'stage': row['Stage (Pokémon)/Type (Energy and Trainer)'],
                            'rule': row['Rule'],
                            'category': row['Category'],
                            'previous_stage': row['Previous stage'],
                            'hp': int(row['HP']) if row['HP'].isdigit() else None,
                            'type': row['Type'],
                            'weakness': row['Weakness'],
                            'resistance': row['Resistance (Type)'],
                            'retreat': int(row['Retreat']) if row['Retreat'].isdigit() else 0,
                            'moves': []
                        }
                    
                    # Add move if it exists
                    if row['Move Name'] and row['Move Name'] != 'n/a':
                        self.cards[card_id]['moves'].append({
                            'name': row['Move Name'],
                            'cost': row['Cost'],
                            'damage': row['Damage'],
                            'effect': row['Effect Explanation']
                        })
                except Exception as e:
                    # Silently skip malformed header rows or errors
                    continue

    def get_card(self, card_id):
        """Retrieve metadata for a specific Card ID. Returns None if not found."""
        return self.cards.get(card_id, None)

    def search_by_name(self, name):
        """Find cards matching a specific name (case-insensitive substring match)."""
        name_lower = name.lower()
        return [card for card in self.cards.values() if name_lower in card['name'].lower()]
