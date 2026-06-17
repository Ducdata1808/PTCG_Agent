import sys
import os

submission_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(submission_dir)

from card_database import CardDatabase

db = CardDatabase()
card = db.get_card(1227)
if card:
    print(f"Loaded card 1227 successfully: {card['name']}")
else:
    print("Failed to load card 1227!")
