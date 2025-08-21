"""
Hierarchical Motif Mining Pipeline

This module implements the main pipeline for discovering hierarchical motifs:
G0 --evolve--> winners --mine--> L1 motifs --update--> G1
G1 --evolve--> winners --mine--> L2 motifs --update--> G2
G2 --evolve--> winners --mine--> L3 motifs --update--> G3
...
Stop when MDL gain < ε or no new motifs discovered

The multi-level loop:
1. Symbolize corpus with motifs up to current max level K
2. Mine candidate motifs as contiguous sequences of symbols
3. Accept candidates as level K+1 if they pass acyclicity/support/MDL criteria
4. Add accepted motifs to grammar and renormalize
5. Stop when no new motifs added or MDL gains < threshold
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from datetime import datetime

# Add blog_demos to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.miner_l1 import mine_l1
from core.miner_l2 import symbolize_corpus, count_ngrams, mdl_gain_symbol, is_acyclic
from core.grammar_updater import (
    add_l1_motifs, save_grammar, get_grammar_stats, remove_unused_motifs,
    next_motif_index, normalize_body_rules, validate_grammar_structure
)


def create_level_k_minus_1_grammar(grammar: Dict[str, Any], target_level: int) -> Dict[str, Any]:
    """
    Create a grammar that only contains motifs with level < target_level.

    This implements the requirement: "When mining motifs for level-k from the winners,
    always use the level-{k-1} representation of the winner genome."

    Args:
        grammar: Full grammar containing motifs of various levels
        target_level: The level we want to mine (k)

    Returns:
        Grammar containing only primitives and motifs with level < target_level
    """
    level_k_minus_1_grammar = {}

    # Copy all non-motif rules (Program, Body, etc.)
    for nt, rules in grammar.items():
        if not (nt.startswith('Motif_') or nt.startswith('L')):
            level_k_minus_1_grammar[nt] = rules
        elif nt.startswith('L') and 'M' in nt:
            # New format: extract level from LkMx name
            try:
                level_part = nt.split('M')[0][1:]  # Remove 'L' and get number before 'M'
                motif_level = int(level_part)
                if motif_level < target_level:
                    level_k_minus_1_grammar[nt] = rules
            except (ValueError, IndexError):
                # Not a valid LkMx format, skip
                pass
        elif nt.startswith('Motif_'):
            # Old format: check level in metadata
            if isinstance(rules, dict) and 'level' in rules:
                motif_level = rules['level']
                if motif_level < target_level:
                    level_k_minus_1_grammar[nt] = rules
            else:
                # No level info, assume level 1 (legacy)
                if 1 < target_level:
                    level_k_minus_1_grammar[nt] = rules

    motif_count = len([k for k in level_k_minus_1_grammar.keys() if k.startswith('L') or k.startswith('Motif_')])
    print(f"[L{target_level} Miner] Created level-{target_level-1} grammar with {motif_count} motifs")

    # Debug: show what motifs were included
    if motif_count > 0:
        motifs = [k for k in level_k_minus_1_grammar.keys() if k.startswith('L') or k.startswith('Motif_')]
        print(f"[L{target_level} Miner] Level-{target_level-1} motifs: {motifs[:5]}{'...' if len(motifs) > 5 else ''}")
    else:
        print(f"[L{target_level} Miner] DEBUG: Full grammar has {len([k for k in grammar.keys() if k.startswith('L') or k.startswith('Motif_')])} motifs")
        all_motifs = [(k, grammar[k].get('level', 'no-level') if isinstance(grammar[k], dict) else 'not-dict') for k in grammar.keys() if k.startswith('L') or k.startswith('Motif_')]
        print(f"[L{target_level} Miner] DEBUG: All motifs: {all_motifs[:5]}{'...' if len(all_motifs) > 5 else ''}")

    return level_k_minus_1_grammar


def mine_lk_motifs(winners_data: Dict[str, Any], grammar: Dict[str, Any], level: int,
                   min_support: int = 3, mdl_lambda: float = 1.0, mdl_thresh: float = 2.0,
                   min_n: int = 2, max_n: int = 5) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """
    Mine level-K motifs from winners using symbol-level analysis.

    This is the generalized version that works for any level K ≥ 2.
    For level K, it mines compositions over {primitives ∪ motifs with level ≤ K-1}.

    Args:
        winners_data: Dictionary with task solutions
        grammar: Current grammar containing motifs up to level K-1
        level: Target level for new motifs (K)
        min_support: Minimum support count for motifs
        mdl_lambda: Lambda parameter for MDL calculation
        mdl_thresh: Minimum MDL gain threshold
        min_n: Minimum n-gram length
        max_n: Maximum n-gram length

    Returns:
        Tuple of (motif_sequences, motif_info_list)
        - motif_sequences: List of motif symbol sequences that passed all criteria
        - motif_info_list: List of dicts with support, MDL, and pattern info
    """
    print(f"[L{level} Miner] Mining level-{level} motifs...")

    # Extract all programs from winners
    programs = []
    for _, task_data in winners_data.items():
        if isinstance(task_data, dict) and 'solutions' in task_data:
            programs.extend(task_data['solutions'])
        elif isinstance(task_data, dict) and 'programs' in task_data:
            programs.extend(task_data['programs'])
        elif isinstance(task_data, list):
            programs.extend(task_data)

    if not programs:
        print(f"[L{level} Miner] No programs found in winners data")
        return [], []

    print(f"[L{level} Miner] Processing {len(programs)} programs")

    # Create level-(k-1) representation: use only motifs with level < k
    level_k_minus_1_grammar = create_level_k_minus_1_grammar(grammar, level)

    # Symbolize programs using level-(k-1) grammar
    # Convert programs to the format expected by symbolize_corpus
    winners_data_formatted = {"temp_task": {"solutions": programs}}
    symbol_sequences = symbolize_corpus(winners_data_formatted, level_k_minus_1_grammar)

    if not symbol_sequences:
        print(f"[L{level} Miner] No symbol sequences generated")
        return [], []

    avg_len = sum(len(seq) for seq in symbol_sequences) / len(symbol_sequences)
    print(f"[L{level} Miner] Generated {len(symbol_sequences)} symbol sequences (avg_len={avg_len:.1f})")

    # Mine frequent n-grams
    print(f"[L{level} Miner] Mining n-grams (n={min_n}-{max_n}, min_support={min_support})...")
    frequent_ngrams = count_ngrams(symbol_sequences, min_n, max_n, min_support)

    print(f"[L{level} Miner] Found {len(frequent_ngrams)} frequent n-grams")

    # Filter by MDL gain and acyclicity
    motifs = []
    motif_info = []

    # Count support for each n-gram
    ngram_counter = Counter()
    for seq in symbol_sequences:
        for i in range(len(seq)):
            for n in range(min_n, min(max_n + 1, len(seq) - i + 1)):
                ngram = tuple(seq[i:i+n])
                ngram_counter[ngram] += 1

    for ngram in frequent_ngrams:
        support = ngram_counter[tuple(ngram)]

        # Check acyclicity: expansion should only contain primitives or motifs with level < K
        if not is_acyclic_for_level(grammar, ngram, level):
            continue

        # Check MDL gain first - no point checking primitives if MDL is too low
        mdl_gain = mdl_gain_symbol(ngram, symbol_sequences, mdl_lambda)

        if mdl_gain < mdl_thresh:
            continue  # Silently reject low-quality patterns

        # For L2+: Reject primitive-only patterns that pass MDL (they should be L1 motifs)
        if level >= 2:
            primitives = {'+', '-', '>', '<', '[', ']'}
            if all(token in primitives for token in ngram):
                print(f"[L{level} Miner] Rejected primitive pattern: {ngram} (MDL={mdl_gain:.2f}, should be L1 motif)")
                continue

        # Pattern passes all checks
            motifs.append(ngram)
            motif_info.append({
                'pattern': ngram,
                'support': support,
                'mdl_gain': mdl_gain,
                'level': level
            })
            print(f"[L{level} Miner] Accepted motif: {ngram} (support={support}, MDL={mdl_gain:.2f})")

    print(f"[L{level} Miner] Final L{level} motifs: {len(motifs)}")
    return motifs, motif_info


def is_acyclic_for_level(grammar: Dict[str, Any], ngram: List[str], target_level: int) -> bool:
    """
    Check if adding this ngram as a level-K motif would maintain acyclicity.

    For level K motifs, the expansion should only contain:
    - Primitives (level 0)
    - Motifs with level ≤ K-1

    Args:
        grammar: Current grammar
        ngram: Proposed motif expansion
        target_level: Level for the new motif

    Returns:
        True if acyclic, False otherwise
    """
    TOKENS = set("><+-[]")

    for symbol in ngram:
        if symbol in TOKENS:
            # Primitive tokens are always OK
            continue
        elif (symbol.startswith('Motif_') or symbol.startswith('L')) and symbol in grammar:
            # Check motif level (support both old Motif_X and new LkMx formats)
            motif_level = None

            if symbol.startswith('Motif_'):
                # Old format: get level from object metadata
                motif_obj = grammar[symbol]
                if isinstance(motif_obj, dict) and 'level' in motif_obj:
                    motif_level = motif_obj['level']
            elif symbol.startswith('L') and 'M' in symbol:
                # New format: extract level from name (e.g., L2M5 -> level 2)
                try:
                    level_part = symbol.split('M')[0][1:]  # Remove 'L' and get number before 'M'
                    motif_level = int(level_part)
                except (ValueError, IndexError):
                    pass  # Not a valid LkMx format

            if motif_level is not None and motif_level >= target_level:
                # Would create cycle: level K motif referencing level ≥ K motif
                return False
            # If no level specified, assume it's OK (legacy motifs)
        else:
            # Unknown symbol - could be problematic
            print(f"[Acyclicity] Warning: unknown symbol '{symbol}' in ngram {ngram}")
            return False

    return True


def add_lk_motifs(grammar: Dict[str, Any], motifs: List[List[str]],
                  motif_weight: float = 0.1, level: int = 2) -> Dict[str, Any]:
    """
    Add level-K motifs to grammar with acyclicity checking.

    This is the generalized version of add_l2_motifs that works for any level K ≥ 2.

    Args:
        grammar: Base grammar to extend
        motifs: List of motif symbol lists
        motif_weight: Weight to assign each motif in Body rules
        level: Motif level (for metadata)

    Returns:
        Updated grammar
    """
    import copy

    grammar = copy.deepcopy(grammar)
    validate_grammar_structure(grammar)

    # Add motifs one by one, checking acyclicity
    added_motifs = []
    for motif_symbols in motifs:
        if not motif_symbols:
            continue

        # Generate motif name using new LkMx format
        motif_index = next_motif_index(grammar, level)
        motif_name = f"L{level}M{motif_index}"

        # Double-check acyclicity before adding
        if not is_acyclic_for_level(grammar, motif_symbols, level):
            print(f"[Grammar] Skipping motif {motif_symbols} - would violate level constraints")
            continue

        # Safe to add - update the grammar
        grammar[motif_name] = {
            'rules': [[motif_symbols, 1.0]],
            'level': level
        }

        # Add reference to Body rules
        grammar['Body']['rules'].append([[motif_name], motif_weight])

        added_motifs.append(motif_name)

    # Normalize Body weights
    normalize_body_rules(grammar)

    print(f"[Grammar] Added {len(added_motifs)} L{level} motifs: {added_motifs}")
    return grammar


class HierarchicalMiner:
    """
    Main class for hierarchical motif mining.
    
    Implements the pipeline:
    1. Start with base grammar G0
    2. Mine L1 motifs from winners, create G1
    3. Mine L2 motifs from winners using G1, create G2
    4. Continue until no significant improvement
    """
    
    def __init__(self, base_grammar_path: str, base_output_dir: str = "hierarchical_evolution",
                 l1_config: Optional[Dict[str, Any]] = None,
                 lk_config: Optional[Dict[str, Any]] = None,
                 max_levels: int = 10):
        """
        Initialize the hierarchical miner.

        Args:
            base_grammar_path: Path to initial grammar (G0)
            base_output_dir: Base directory for hierarchical evolution experiments
            l1_config: Configuration for L1 mining
            lk_config: Configuration for L2+ mining (applies to all levels ≥ 2)
            max_levels: Maximum number of levels to discover
        """
        self.base_grammar_path = base_grammar_path
        self.base_output_dir = base_output_dir
        self.max_levels = max_levels

        # Create session directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        # Default configurations
        self.l1_config = {
            'step_limit': 1000,
            'min_support': 3,
            'mdl_lambda': 1.5,
            'mdl_thresh': 3.0,
            'max_linear_len': 6,
            'min_len': 3,
            'motif_weight': 0.1
        }
        if l1_config:
            self.l1_config.update(l1_config)

        # Configuration for all levels K ≥ 2 (L2, L3, L4, ...)
        self.lk_config = {
            'min_support': 3,
            'mdl_lambda': 1.0,
            'mdl_thresh': 2.0,
            'min_n': 2,
            'max_n': 5,
            'motif_weight': 0.1
        }
        if lk_config:
            self.lk_config.update(lk_config)

        # Pipeline state
        self.current_grammar = None
        self.iteration = 0
        self.history = []
        self.current_round_dir = None
    
    def run_pipeline(self, winners_data: Dict[str, Any], max_iterations: int = 5) -> Dict[str, Any]:
        """
        Run the complete hierarchical mining pipeline.
        
        Args:
            winners_data: Dictionary with task solutions
            max_iterations: Maximum number of iterations
            
        Returns:
            Final enriched grammar
        """
        print("Starting hierarchical motif mining pipeline...")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Session directory: {self.session_dir}")

        # Load base grammar
        with open(self.base_grammar_path, 'r') as f:
            self.current_grammar = json.load(f)

        print(f"[Pipeline] Loaded base grammar: {self.base_grammar_path}")
        stats = get_grammar_stats(self.current_grammar)
        print(f"  - Motifs: {stats['motif_count']} (levels: {dict(stats['motifs_by_level'])})")

        # Run iterations
        for iteration in range(1, max_iterations + 1):
            self.iteration = iteration

            # Create round directory
            self.current_round_dir = os.path.join(self.session_dir, f"round_{iteration}")
            os.makedirs(self.current_round_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"Round directory: {self.current_round_dir}")
            print(f"{'='*60}")

            iteration_start_stats = get_grammar_stats(self.current_grammar)
            total_discovered = 0
            level_motifs_discovered = {}
            all_motif_info = {}
            
            # Phase 1: Mine L1 motifs
            print("\n[L1 Phase] Mining token-level motifs...")
            l1_motifs = mine_l1(
                winners_data=winners_data,
                step_limit=self.l1_config['step_limit'],
                min_support=self.l1_config['min_support'],
                mdl_lambda=self.l1_config['mdl_lambda'],
                mdl_thresh=self.l1_config['mdl_thresh'],
                max_linear_len=self.l1_config['max_linear_len'],
                min_len=self.l1_config['min_len']
            )

            if l1_motifs:
                self.current_grammar = add_l1_motifs(
                    grammar=self.current_grammar,
                    motifs=l1_motifs,
                    motif_weight=self.l1_config['motif_weight'],
                    level=1
                )
                total_discovered += len(l1_motifs)
                level_motifs_discovered[1] = len(l1_motifs)

                # Save L1 motifs info (simplified since L1 doesn't return detailed info)
                l1_motif_info = []
                for motif in l1_motifs:
                    l1_motif_info.append({
                        'pattern': list(motif),
                        'support': 'N/A',  # L1 miner doesn't return support info
                        'mdl_gain': 'N/A',  # L1 miner doesn't return MDL info
                        'level': 1
                    })
                all_motif_info[1] = l1_motif_info

                print(f"[L1 Phase] Added {len(l1_motifs)} motifs")
            else:
                print("[L1 Phase] No L1 motifs discovered")
                level_motifs_discovered[1] = 0
                all_motif_info[1] = []

            # Always save L1 grammar and motifs (even if empty)
            l1_grammar_path = os.path.join(self.current_round_dir, "grammar_L1.json")
            save_grammar(self.current_grammar, l1_grammar_path)

            l1_motifs_path = os.path.join(self.current_round_dir, "motifs_L1.json")
            with open(l1_motifs_path, 'w') as f:
                json.dump(all_motif_info[1], f, indent=2)

            print(f"[L1 Phase] Saved grammar: {l1_grammar_path}")
            print(f"[L1 Phase] Saved motifs: {l1_motifs_path}")
            
            # Phase 2+: Mine L2, L3, L4, ... motifs (symbol-level)
            # Keep mining higher levels until no new motifs found
            level = 2

            while level <= self.max_levels:
                print(f"\n[L{level} Phase] Mining level-{level} motifs...")

                lk_motifs, lk_motif_info = mine_lk_motifs(
                    winners_data=winners_data,
                    grammar=self.current_grammar,
                    level=level,
                    min_support=self.lk_config['min_support'],
                    mdl_lambda=self.lk_config['mdl_lambda'],
                    mdl_thresh=self.lk_config['mdl_thresh'],
                    min_n=self.lk_config['min_n'],
                    max_n=self.lk_config['max_n']
                )

                if lk_motifs:
                    self.current_grammar = add_lk_motifs(
                        grammar=self.current_grammar,
                        motifs=lk_motifs,
                        motif_weight=self.lk_config['motif_weight'],
                        level=level
                    )
                    total_discovered += len(lk_motifs)
                    level_motifs_discovered[level] = len(lk_motifs)
                    all_motif_info[level] = lk_motif_info

                    print(f"[L{level} Phase] Added {len(lk_motifs)} motifs")
                else:
                    print(f"[L{level} Phase] No L{level} motifs discovered - stopping level discovery")
                    level_motifs_discovered[level] = 0
                    all_motif_info[level] = lk_motif_info  # Empty list

                # Always save Lk grammar and motifs (even if empty)
                lk_grammar_path = os.path.join(self.current_round_dir, f"grammar_L{level}.json")
                save_grammar(self.current_grammar, lk_grammar_path)

                lk_motifs_path = os.path.join(self.current_round_dir, f"motifs_L{level}.json")
                with open(lk_motifs_path, 'w') as f:
                    json.dump(all_motif_info[level], f, indent=2)

                print(f"[L{level} Phase] Saved grammar: {lk_grammar_path}")
                print(f"[L{level} Phase] Saved motifs: {lk_motifs_path}")

                if lk_motifs:
                    level += 1  # Continue to next level
                else:
                    break  # No more motifs at this level, stop
            
            # Clean up unused motifs
            self.current_grammar = remove_unused_motifs(self.current_grammar)

            # Save final iteration grammar
            final_iteration_grammar_path = os.path.join(self.current_round_dir, "grammar_final.json")
            save_grammar(self.current_grammar, final_iteration_grammar_path)
            print(f"[Pipeline] Saved final iteration grammar: {final_iteration_grammar_path}")

            # Save all motifs summary for this iteration
            all_motifs_path = os.path.join(self.current_round_dir, "motifs_all.json")
            with open(all_motifs_path, 'w') as f:
                json.dump(all_motif_info, f, indent=2)
            print(f"[Pipeline] Saved all motifs summary: {all_motifs_path}")

            # Track history
            iteration_end_stats = get_grammar_stats(self.current_grammar)
            history_entry = {
                'iteration': iteration,
                'l1_motifs': len(l1_motifs) if l1_motifs else 0,
                'total_motifs_before': iteration_start_stats['motif_count'],
                'total_motifs_after': iteration_end_stats['motif_count'],
                'motifs_added': iteration_end_stats['motif_count'] - iteration_start_stats['motif_count'],
                'max_level_reached': level - 1,
                'level_motifs': level_motifs_discovered.copy(),
                'round_dir': self.current_round_dir
            }
            self.history.append(history_entry)

            print(f"\n[Iteration {iteration} Summary]")
            print(f"  L1 motifs: {len(l1_motifs) if l1_motifs else 0}")
            for lvl, count in level_motifs_discovered.items():
                if lvl > 1:  # Skip L1 since we already printed it
                    print(f"  L{lvl} motifs: {count}")
            print(f"  Max level reached: L{level - 1}")
            print(f"  Total motifs: {iteration_start_stats['motif_count']} → {iteration_end_stats['motif_count']}")

            # Check stopping condition
            if total_discovered == 0:
                print(f"\n[Pipeline] Stopping: no motifs discovered in iteration {iteration}")
                break
        
        # Save session summary
        session_summary_path = os.path.join(self.session_dir, "session_summary.json")
        with open(session_summary_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[Pipeline] Saved session summary: {session_summary_path}")

        # Print summary
        self.print_summary()

        return self.current_grammar
    
    def print_summary(self) -> None:
        """Print a summary of the mining pipeline."""
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")

        # Calculate totals by level
        total_l1 = sum(h['l1_motifs'] for h in self.history)
        level_totals = {}
        max_level_overall = 1

        for h in self.history:
            max_level_overall = max(max_level_overall, h.get('max_level_reached', 1))
            for level, count in h.get('level_motifs', {}).items():
                level_totals[level] = level_totals.get(level, 0) + count

        print(f"Iterations completed: {len(self.history)}")
        print(f"Maximum level reached: L{max_level_overall}")
        print(f"Total L1 motifs mined: {total_l1}")
        for level in sorted(level_totals.keys()):
            print(f"Total L{level} motifs mined: {level_totals[level]}")

        if self.current_grammar:
            final_stats = get_grammar_stats(self.current_grammar)
            print("Final grammar statistics:")
            print(f"  - Total non-terminals: {final_stats['total_nonterminals']}")
            print(f"  - Total motifs: {final_stats['motif_count']}")
            print(f"  - Motifs by level: {dict(final_stats['motifs_by_level'])}")
            print(f"  - Body rules: {final_stats['body_rules']}")
            print(f"  - Is acyclic: {final_stats['is_acyclic']}")

        print("\nIteration details:")
        for h in self.history:
            level_str = f"L1={h['l1_motifs']}"
            for level, count in h.get('level_motifs', {}).items():
                level_str += f", L{level}={count}"
            print(f"  Iter {h['iteration']}: {level_str}, max_level=L{h.get('max_level_reached', 1)}, "
                  f"total={h['total_motifs_before']}→{h['total_motifs_after']} "
                  f"(+{h['motifs_added']})")


def main():
    """CLI interface for the hierarchical miner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical motif mining pipeline")
    parser.add_argument("--winners", required=True, help="Path to winners JSON file")
    parser.add_argument("--base-grammar", required=True, help="Path to base grammar")
    parser.add_argument("--output-dir", required=True, help="Output directory for grammars")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    
    # L1 configuration
    parser.add_argument("--l1-min-support", type=int, default=3)
    parser.add_argument("--l1-mdl-thresh", type=float, default=3.0)
    parser.add_argument("--l1-motif-weight", type=float, default=0.1)

    # LK (L2+) configuration
    parser.add_argument("--lk-min-support", type=int, default=3)
    parser.add_argument("--lk-mdl-thresh", type=float, default=2.0)
    parser.add_argument("--lk-motif-weight", type=float, default=0.1)
    parser.add_argument("--max-levels", type=int, default=10, help="Maximum levels to discover")

    args = parser.parse_args()

    # Load winners data
    with open(args.winners, 'r') as f:
        winners_data = json.load(f)

    print(f"Loaded winners from: {args.winners}")
    print(f"Tasks: {len(winners_data)}")
    total_solutions = sum(len(task_data.get('solutions', []))
                         for task_data in winners_data.values())
    print(f"Total solutions: {total_solutions}")

    # Configure miners
    l1_config = {
        'min_support': args.l1_min_support,
        'mdl_thresh': args.l1_mdl_thresh,
        'motif_weight': args.l1_motif_weight
    }

    lk_config = {
        'min_support': args.lk_min_support,
        'mdl_thresh': args.lk_mdl_thresh,
        'motif_weight': args.lk_motif_weight
    }

    # Create and run pipeline
    miner = HierarchicalMiner(
        base_grammar_path=args.base_grammar,
        base_output_dir=args.output_dir,
        l1_config=l1_config,
        lk_config=lk_config,
        max_levels=args.max_levels
    )

    miner.run_pipeline(winners_data, max_iterations=args.max_iterations)
    
    print("\nPipeline completed successfully!")
    print(f"Final grammar available in: {args.output_dir}")


if __name__ == "__main__":
    main()