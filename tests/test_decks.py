import glob
import os
import sys
import unittest

# Ensure the root src folder is in the path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.core.card_database import CardDatabase

class TestDecksValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize database pointing to EN_Card_Data.csv
        csv_path = os.path.join(base_dir, "data", "EN_Card_Data.csv")
        cls.db = CardDatabase(csv_path)

    def test_all_decks_in_folder(self):
        # Find all .csv files in the decks/ directory
        decks_dir = os.path.join(base_dir, "decks")
        deck_files = glob.glob(os.path.join(decks_dir, "*.csv"))
        
        # Also check sample_submission/deck.csv
        sample_deck = os.path.join(base_dir, "data", "sample_submission", "deck.csv")
        if os.path.exists(sample_deck):
            deck_files.append(sample_deck)

        self.assertGreater(len(deck_files), 0, "No deck .csv files found to validate.")

        for deck_path in deck_files:
            deck_name = os.path.basename(deck_path)
            with self.subTest(deck=deck_name):
                self.validate_single_deck(deck_path)

    def validate_single_deck(self, deck_path):
        # 1. Read deck file
        card_ids = []
        with open(deck_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    card_ids.append(int(line))

        # 2. Check total size is exactly 60
        self.assertEqual(len(card_ids), 60, f"Deck {deck_path} must contain exactly 60 cards, got {len(card_ids)}")

        # 3. Check every card ID is valid and retrieve metadata
        deck_cards = []
        for card_id in card_ids:
            card = self.db.get_card(card_id)
            self.assertIsNotNone(card, f"Card ID {card_id} in deck {deck_path} does not exist in the card database.")
            deck_cards.append(card)

        # 4. Check for at least 1 Basic Pokémon
        has_basic_pokemon = False
        for card in deck_cards:
            # Stage 'Basic Pokémon' or checks stage string
            if card['category'] == 'Pokémon' and 'Basic' in card['stage']:
                has_basic_pokemon = True
                break
            # Also handle Japanese/Custom categories just in case
            if "Basic Pokémon" in card['stage'] or "Basic" in card['stage'] and "Pokémon" in card['category']:
                has_basic_pokemon = True
                break
            # Special case for Trainer's Pokémon (Team Rocket)
            if "Pokémon" in card['category'] and "Basic" in card['stage']:
                has_basic_pokemon = True
                break

        self.assertTrue(has_basic_pokemon, f"Deck {deck_path} must contain at least 1 Basic Pokémon.")

        # 5. Check 4-copy card rule (ignoring Basic Energy)
        card_counts = {}
        for card in deck_cards:
            name = card['name']
            is_basic_energy = "Basic" in card['stage'] and "Energy" in card['stage']
            
            if not is_basic_energy:
                card_counts[name] = card_counts.get(name, 0) + 1

        for name, count in card_counts.items():
            self.assertLessEqual(count, 4, f"Card '{name}' has {count} copies in deck {deck_path} (maximum allowed is 4).")
            
        # 6. Check 1-copy ACE SPEC rule
        ace_spec_count = 0
        for card in deck_cards:
            if "ACE SPEC" in card['rule']:
                ace_spec_count += 1
                
        self.assertLessEqual(ace_spec_count, 1, f"Deck {deck_path} has {ace_spec_count} ACE SPEC cards (maximum allowed is 1).")

if __name__ == '__main__':
    unittest.main()
