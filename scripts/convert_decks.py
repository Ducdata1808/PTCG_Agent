import os
import csv
import re

csv_path = "data/EN_Card_Data.csv"
raw_dir = "decks/raw_text"
out_dir = "decks"

# Load card database
cards = {}
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row:
            continue
        card_id = int(row[0])
        name = row[1]
        cards[name] = card_id

def normalize(s):
    s = s.lower()
    s = s.replace("'", "").replace("’", "").replace("`", "")
    s = s.replace("-", " ")
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def map_special_names(name):
    name_lower = name.lower()
    if 'grass energy' in name_lower:
        return 'Basic {G} Energy'
    elif 'fire energy' in name_lower:
        return 'Basic {R} Energy'
    elif 'water energy' in name_lower:
        return 'Basic {W} Energy'
    elif 'lightning energy' in name_lower:
        return 'Basic {L} Energy'
    elif 'psychic energy' in name_lower:
        return 'Basic {P} Energy'
    elif 'fighting energy' in name_lower:
        return 'Basic {F} Energy'
    elif 'darkness energy' in name_lower:
        return 'Basic {D} Energy'
    elif 'metal energy' in name_lower:
        return 'Basic {M} Energy'
    return name

normalized_cards = {normalize(name): (name, card_id) for name, card_id in cards.items()}

# Priority card IDs for replacements
# 1227: Lillie's Determination
# 1102: Dusk Ball
# 1123: Switch
# 1152: Poké Pad
# 1097: Night Stretcher
SPECIAL_RED_PRIORITY = [1227, 1102, 1123, 1152, 1097]
OTHER_PRIORITY = [1102, 1227, 1123, 1152, 1097]

# Find all txt files in raw_dir
txt_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]

for filename in txt_files:
    file_path = os.path.join(raw_dir, filename)
    print(f"\nProcessing {filename}...")
    
    deck_lines = []
    total_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Skip header lines if any
            if line.startswith("Pokémon:") or line.startswith("Trainer:") or line.startswith("Energy:"):
                continue
                
            match = re.match(r'^(\d+)\s+(.+)$', line)
            if not match:
                continue
                
            count = int(match.group(1))
            raw_name = match.group(2).strip()
            
            # Remove expansion suffix if present (e.g. "TWM 128" or "MEG 119")
            clean_name = re.sub(r'\s+[A-Z]{3,4}\s+\d+$', '', raw_name)
            clean_name = map_special_names(clean_name)
            
            deck_lines.append((count, clean_name, line_num, line))
            total_count += count
            
    if total_count != 60:
        print(f"  [Error] Deck count is not 60 (got {total_count}) in {filename}")
        continue

    # First Pass: Match all standard cards, identify replacements
    matched_ids = []
    card_counts = {}  # Keep track of counts of non-basic energy card IDs
    pending_replacements = [] # list of (count, type) where type is 'special_red' or 'other'
    missing_unresolved = []

    for count, name, line_num, orig_line in deck_lines:
        name_lower = name.lower()
        if 'special red card' in name_lower:
            pending_replacements.append((count, 'special_red'))
            continue
        elif any(x in name_lower for x in ['pokemon center lady', 'pokémon center lady', "az's tranquility", 'transformation tome']):
            pending_replacements.append((count, 'other'))
            continue

        norm_name = normalize(name)
        
        # Check standard lookup
        found_card_id = None
        if norm_name in normalized_cards:
            _, found_card_id = normalized_cards[norm_name]
        else:
            # Try substring match
            for c_norm, (real_name, card_id) in normalized_cards.items():
                if norm_name == c_norm or norm_name in c_norm or c_norm in norm_name:
                    found_card_id = card_id
                    break

        if found_card_id is not None:
            # Check if it's basic energy (ignore 4-copy rule for basic energy)
            is_basic_energy = False
            for real_name, cid in cards.items():
                if cid == found_card_id:
                    if "Basic" in real_name and "Energy" in real_name:
                        is_basic_energy = True
                    break
            
            matched_ids.extend([found_card_id] * count)
            if not is_basic_energy:
                card_counts[found_card_id] = card_counts.get(found_card_id, 0) + count
        else:
            # If not found in DB at all, treat as 'other' replacement
            pending_replacements.append((count, 'other'))

    # Second Pass: Resolve replacements dynamically
    for count, rep_type in pending_replacements:
        priority_list = SPECIAL_RED_PRIORITY if rep_type == 'special_red' else OTHER_PRIORITY
        for _ in range(count):
            resolved = False
            for target_id in priority_list:
                if card_counts.get(target_id, 0) < 4:
                    card_counts[target_id] = card_counts.get(target_id, 0) + 1
                    matched_ids.append(target_id)
                    resolved = True
                    break
            if not resolved:
                print(f"  [Error] Could not resolve replacement for type {rep_type} in {filename} without violating copy limits!")

    # Write to csv
    base_name = os.path.splitext(filename)[0]
    out_path = os.path.join(out_dir, f"{base_name}.csv")
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for card_id in matched_ids:
            out_f.write(f"{card_id}\n")
    print(f"  [Success] Wrote 60 card IDs to {out_path}")
