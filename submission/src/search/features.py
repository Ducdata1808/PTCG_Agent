from cg.api import Observation

def extract_features(obs: Observation, own_idx: int) -> list[float]:
    """
    Extracts a fixed-length feature vector from a game state Observation for the MLP value network.
    Vector length: 20 features.
    """
    if not obs or not obs.current or len(obs.current.players) < 2:
        return [0.0] * 20

    own = obs.current.players[own_idx]
    opp = obs.current.players[1 - own_idx]

    # 1. Prizes remaining (lower is better)
    own_prizes = len(own.prize) if own.prize else 0
    opp_prizes = len(opp.prize) if opp.prize else 0
    prize_diff = opp_prizes - own_prizes

    # 2. Active Pokemon HP
    own_active_hp = 0.0
    if own.active and own.active[0]:
        own_active_hp = float(own.active[0].hp)
        
    opp_active_hp = 0.0
    if opp.active and opp.active[0]:
        opp_active_hp = float(opp.active[0].hp)

    # 3. Bench Pokemon HP
    own_bench_hp = 0.0
    for b in own.bench:
        if b:
            own_bench_hp += float(b.hp)
            
    opp_bench_hp = 0.0
    for b in opp.bench:
        if b:
            opp_bench_hp += float(b.hp)

    # 4. Energy attached to Active
    own_active_energy = 0.0
    if own.active and own.active[0] and own.active[0].energies:
        own_active_energy = float(len(own.active[0].energies))
        
    opp_active_energy = 0.0
    if opp.active and opp.active[0] and opp.active[0].energies:
        opp_active_energy = float(len(opp.active[0].energies))

    # 5. Energy attached to Bench
    own_bench_energy = 0.0
    for b in own.bench:
        if b and b.energies:
            own_bench_energy += float(len(b.energies))
            
    opp_bench_energy = 0.0
    for b in opp.bench:
        if b and b.energies:
            opp_bench_energy += float(len(b.energies))

    # 6. Hand count
    own_hand = float(own.handCount)
    opp_hand = float(opp.handCount)

    # 7. Deck count
    own_deck = float(own.deckCount)
    opp_deck = float(opp.deckCount)

    # 8. Discard count
    own_discard = float(len(own.discard)) if own.discard else 0.0
    opp_discard = float(len(opp.discard)) if opp.discard else 0.0

    # 9. Bench count
    own_bench_count = float(len(own.bench)) if own.bench else 0.0
    opp_bench_count = float(len(opp.bench)) if opp.bench else 0.0

    # 10. Status conditions for active (1.0 if any, 0.0 otherwise)
    own_status = 0.0
    if own.poisoned or own.burned or own.asleep or own.paralyzed or own.confused:
        own_status = 1.0
            
    opp_status = 0.0
    if opp.poisoned or opp.burned or opp.asleep or opp.paralyzed or opp.confused:
        opp_status = 1.0

    return [
        float(own_prizes),
        float(opp_prizes),
        float(prize_diff),
        own_active_hp,
        opp_active_hp,
        own_bench_hp,
        opp_bench_hp,
        own_active_energy,
        opp_active_energy,
        own_bench_energy,
        opp_bench_energy,
        own_bench_count,
        opp_bench_count,
        own_status,
        opp_status,
        float(obs.current.turn),
        min(own_hand, 5.0),
        min(opp_hand, 5.0),
        min(own_discard, 8.0),
        min(opp_discard, 8.0)
    ]
