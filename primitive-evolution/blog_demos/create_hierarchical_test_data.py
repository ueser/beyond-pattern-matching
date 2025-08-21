#!/usr/bin/env python3
"""
Create synthetic hierarchical test data with clear multi-level motif structure
"""

import json
import random
from typing import List, Dict

def create_hierarchical_test_data():
    """
    Create synthetic winners.json with clear L1->L2->L3->L4->L5 hierarchical motifs
    """
    
    # Define the hierarchical motif structure we want to discover
    
    # L1 motifs (primitive combinations) - meaningful patterns
    l1_motifs = {
        'INC2': '++',        # increment twice
        'DEC2': '--',        # decrement twice  
        'RIGHT2': '>>',      # move right twice
        'LEFT2': '<<',       # move left twice
        'INC3': '+++',       # increment three times
        'DEC3': '---',       # decrement three times
        'MOVE_RIGHT': '+>',  # increment then move right
        'MOVE_LEFT': '-<',   # decrement then move left
        'RESET': '+-',       # increment then decrement (no-op, but pattern)
        'FLIP': '-+',        # decrement then increment (no-op, but pattern)
    }
    
    # L2 motifs (L1 combinations) - should be discovered as L1 motif combinations
    l2_motifs = {
        'DOUBLE_INC': ['INC2', 'INC2'],           # ++ then ++ = ++++
        'DOUBLE_DEC': ['DEC2', 'DEC2'],           # -- then -- = ----
        'INC_MOVE': ['INC2', 'RIGHT2'],           # ++ then >> = ++>>
        'DEC_MOVE': ['DEC2', 'LEFT2'],            # -- then << = --<<
        'BIG_INC': ['INC3', 'INC2'],              # +++ then ++ = +++++
        'BIG_DEC': ['DEC3', 'DEC2'],              # --- then -- = -----
        'MOVE_PATTERN': ['MOVE_RIGHT', 'MOVE_LEFT'], # +> then -< = +>-<
        'RESET_CYCLE': ['RESET', 'FLIP'],         # +- then -+ = +--+
    }
    
    # L3 motifs (L1+L2 combinations)
    l3_motifs = {
        'MEGA_INC': ['DOUBLE_INC', 'INC3'],       # (++++) then +++ = +++++++
        'COMPLEX_MOVE': ['INC_MOVE', 'DEC_MOVE'], # (++>>) then (--<<) = ++>>--<<
        'PATTERN_REPEAT': ['MOVE_PATTERN', 'RESET_CYCLE'], # (+>-<) then (+--+) = +>-<+--+
    }
    
    # L4 motifs (L1+L2+L3 combinations)
    l4_motifs = {
        'SUPER_PATTERN': ['MEGA_INC', 'COMPLEX_MOVE'],  # (+++++++)(++>>--<<) 
        'ULTRA_CYCLE': ['PATTERN_REPEAT', 'DOUBLE_INC'], # (+>-<+--+)(++++)
    }
    
    # L5 motifs (L1+L2+L3+L4 combinations)
    l5_motifs = {
        'ULTIMATE': ['SUPER_PATTERN', 'ULTRA_CYCLE'],  # Combines everything
    }
    
    def expand_motif(motif_id: str, level: int = 1) -> str:
        """Recursively expand a motif to its primitive representation"""
        if level == 1 and motif_id in l1_motifs:
            return l1_motifs[motif_id]
        elif level == 2 and motif_id in l2_motifs:
            parts = []
            for part in l2_motifs[motif_id]:
                parts.append(expand_motif(part, level=1))
            return ''.join(parts)
        elif level == 3 and motif_id in l3_motifs:
            parts = []
            for part in l3_motifs[motif_id]:
                if part in l1_motifs:
                    parts.append(expand_motif(part, level=1))
                elif part in l2_motifs:
                    parts.append(expand_motif(part, level=2))
            return ''.join(parts)
        elif level == 4 and motif_id in l4_motifs:
            parts = []
            for part in l4_motifs[motif_id]:
                if part in l1_motifs:
                    parts.append(expand_motif(part, level=1))
                elif part in l2_motifs:
                    parts.append(expand_motif(part, level=2))
                elif part in l3_motifs:
                    parts.append(expand_motif(part, level=3))
            return ''.join(parts)
        elif level == 5 and motif_id in l5_motifs:
            parts = []
            for part in l5_motifs[motif_id]:
                if part in l1_motifs:
                    parts.append(expand_motif(part, level=1))
                elif part in l2_motifs:
                    parts.append(expand_motif(part, level=2))
                elif part in l3_motifs:
                    parts.append(expand_motif(part, level=3))
                elif part in l4_motifs:
                    parts.append(expand_motif(part, level=4))
            return ''.join(parts)
        else:
            return motif_id  # fallback
    
    # Generate synthetic programs with clear hierarchical structure
    programs = []
    
    # Generate programs with L1 motifs (high frequency)
    for motif_id in l1_motifs:
        pattern = expand_motif(motif_id, level=1)
        for i in range(15):  # 15 copies of each L1 motif
            program = f",{pattern}."
            programs.append(program)
    
    # Generate programs with L2 motifs (medium frequency)  
    for motif_id in l2_motifs:
        pattern = expand_motif(motif_id, level=2)
        for i in range(8):  # 8 copies of each L2 motif
            program = f",{pattern}."
            programs.append(program)
    
    # Generate programs with L3 motifs (lower frequency)
    for motif_id in l3_motifs:
        pattern = expand_motif(motif_id, level=3)
        for i in range(5):  # 5 copies of each L3 motif
            program = f",{pattern}."
            programs.append(program)
    
    # Generate programs with L4 motifs (low frequency)
    for motif_id in l4_motifs:
        pattern = expand_motif(motif_id, level=4)
        for i in range(3):  # 3 copies of each L4 motif
            program = f",{pattern}."
            programs.append(program)
    
    # Generate programs with L5 motifs (very low frequency)
    for motif_id in l5_motifs:
        pattern = expand_motif(motif_id, level=5)
        for i in range(2):  # 2 copies of each L5 motif
            program = f",{pattern}."
            programs.append(program)
    
    # Add some composite programs (combinations of different level motifs)
    composite_patterns = [
        f"{expand_motif('INC2', 1)}{expand_motif('DOUBLE_INC', 2)}",  # L1 + L2
        f"{expand_motif('MOVE_PATTERN', 2)}{expand_motif('INC3', 1)}", # L2 + L1
        f"{expand_motif('MEGA_INC', 3)}{expand_motif('DEC2', 1)}",    # L3 + L1
    ]
    
    for pattern in composite_patterns:
        for i in range(4):  # 4 copies of each composite
            program = f",{pattern}."
            programs.append(program)
    
    # Shuffle the programs
    random.seed(42)
    random.shuffle(programs)
    
    # Create winners.json structure
    winners_data = {
        "hierarchical_task_1": {
            "solutions": programs[:len(programs)//4]
        },
        "hierarchical_task_2": {
            "solutions": programs[len(programs)//4:len(programs)//2]
        },
        "hierarchical_task_3": {
            "solutions": programs[len(programs)//2:3*len(programs)//4]
        },
        "hierarchical_task_4": {
            "solutions": programs[3*len(programs)//4:]
        }
    }
    
    # Save to file
    output_path = "hierarchical_test_winners.json"
    with open(output_path, 'w') as f:
        json.dump(winners_data, f, indent=2)
    
    print(f"Created hierarchical test data: {output_path}")
    print(f"Total programs: {len(programs)}")
    print(f"Expected motifs:")
    print(f"  L1: {len(l1_motifs)} motifs")
    print(f"  L2: {len(l2_motifs)} motifs") 
    print(f"  L3: {len(l3_motifs)} motifs")
    print(f"  L4: {len(l4_motifs)} motifs")
    print(f"  L5: {len(l5_motifs)} motifs")
    
    # Show some example programs
    print(f"\nExample programs:")
    for i, program in enumerate(programs[:10]):
        print(f"  {i+1}. {program}")
    
    # Show expected expansions
    print(f"\nExpected motif expansions:")
    print(f"  L1 'INC2' (++): '{expand_motif('INC2', 1)}'")
    print(f"  L2 'DOUBLE_INC' (INC2+INC2): '{expand_motif('DOUBLE_INC', 2)}'")
    print(f"  L3 'MEGA_INC' (DOUBLE_INC+INC3): '{expand_motif('MEGA_INC', 3)}'")
    print(f"  L4 'SUPER_PATTERN': '{expand_motif('SUPER_PATTERN', 4)}'")
    print(f"  L5 'ULTIMATE': '{expand_motif('ULTIMATE', 5)}'")
    
    return output_path

if __name__ == "__main__":
    create_hierarchical_test_data()
