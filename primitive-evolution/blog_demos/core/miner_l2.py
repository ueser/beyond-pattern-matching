"""
L2 (Symbol-Level) Motif Miner for Hierarchical Compositions

This module mines motifs at the symbol level, finding frequent compositions 
of already-known motifs and tokens. It enforces acyclicity to prevent infinite recursion.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import sys
import os

# Add blog_demos to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

TOKENS = set("><+-[]")


def flatten_motif_to_tokens(grammar: Dict[str, Any], nt: str, memo: Dict[str, List[str]], 
                          stack: Optional[Set[str]] = None) -> List[str]:
    """
    Flatten a motif to its constituent tokens, handling cycles gracefully.
    Returns empty list if cycles are detected.
    """
    if nt in memo:
        return memo[nt]
    
    if stack is None:
        stack = set()
        
    if nt in stack:  # Cycle detected
        memo[nt] = []
        return []
    
    obj = grammar.get(nt)
    if not isinstance(obj, dict) or 'rules' not in obj:
        # Terminal symbol - return as-is if token, empty otherwise
        result = [nt] if nt in TOKENS else []
        memo[nt] = result
        return result
    
    rules = obj['rules']
    if not (isinstance(rules, list) and len(rules) >= 1 and 
            isinstance(rules[0], list) and len(rules[0]) == 2):
        memo[nt] = []
        return []
    
    prod, _weight = rules[0]
    stack.add(nt)
    
    tokens = []
    for sym in prod:
        # Support both old Motif_X and new LkMx formats
        if isinstance(sym, str) and (sym.startswith('Motif_') or (sym.startswith('L') and 'M' in sym)):
            tokens.extend(flatten_motif_to_tokens(grammar, sym, memo, stack))
        elif isinstance(sym, str) and sym in TOKENS:
            tokens.append(sym)
        # Skip other symbols
    
    stack.remove(nt)
    memo[nt] = tokens
    return tokens


def build_motif_key_map(grammar: Dict[str, Any]) -> Dict[str, str]:
    """Build mapping from token sequences to motif names for longest-match replacement."""
    key_map: Dict[str, str] = {}
    memo: Dict[str, List[str]] = {}
    
    for nt, obj in grammar.items():
        # Support both old Motif_X and new LkMx formats
        if not (isinstance(nt, str) and (nt.startswith('Motif_') or (nt.startswith('L') and 'M' in nt))):
            continue
        if not (isinstance(obj, dict) and 'rules' in obj):
            continue

        tokens = flatten_motif_to_tokens(grammar, nt, memo)
        if tokens and all(t in TOKENS for t in tokens):
            key_map[''.join(tokens)] = nt
    
    return key_map


def motifize_body(body: str, key_map: Dict[str, str]) -> List[str]:
    """
    Replace longest-matching token sequences with motif names.
    Returns list of symbols (mix of motif names and individual tokens).
    """
    # Sort keys by length (longest first) for greedy longest-match
    ordered_keys = sorted(key_map.keys(), key=len, reverse=True)
    
    symbols = []
    i = 0
    
    while i < len(body):
        matched = False
        for key in ordered_keys:
            if body.startswith(key, i):
                symbols.append(key_map[key])
                i += len(key)
                matched = True
                break
        
        if not matched:
            symbols.append(body[i])
            i += 1
    
    return symbols


def symbolize_corpus(winners_data: Dict[str, Any], grammar: Dict[str, Any]) -> List[List[str]]:
    """Convert winner programs to symbol-level sequences using current grammar."""
    # Build key mapping for token->motif replacement
    key_map = build_motif_key_map(grammar)
    
    symbol_sequences = []
    
    for task_name, task_data in winners_data.items():
        solutions = task_data.get('solutions', [])
        
        for prog in solutions:
            # Strip I/O wrapping if present
            body = prog
            if len(prog) >= 2 and prog[0] == ',' and prog[-1] == '.':
                body = prog[1:-1]
            
            # Strip no-ops
            body = strip_noops(body)
            if not body:
                continue
                
            # Convert to symbol sequence
            symbols = motifize_body(body, key_map)
            if symbols:
                symbol_sequences.append(symbols)
    
    return symbol_sequences


def strip_noops(s: str) -> str:
    """Remove local no-ops until stable."""
    prev = None
    while s != prev:
        prev = s
        s = s.replace("+-", "").replace("-+", "")
        s = s.replace("<>", "").replace("><", "")
        s = s.replace("[]", "")
    return s


def count_ngrams(symbol_sequences: List[List[str]], min_n: int = 2, max_n: int = 5, 
                min_support: int = 3) -> List[List[str]]:
    """Count frequent n-grams in symbol sequences."""
    ngram_counter = Counter()
    
    for seq in symbol_sequences:
        for n in range(min_n, min(max_n + 1, len(seq) + 1)):
            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i:i+n])
                ngram_counter[ngram] += 1
    
    # Return n-grams with sufficient support as lists
    frequent_ngrams = []
    for ngram, count in ngram_counter.items():
        if count >= min_support:
            frequent_ngrams.append(list(ngram))
    
    return frequent_ngrams


def is_acyclic(grammar: Dict[str, Any], new_motif_expansion: List[str], 
              new_motif_name: Optional[str] = None) -> bool:
    """
    Check if adding a new motif with given expansion would create cycles.
    Uses topological ordering approach.
    """
    # Build dependency graph: motif -> set of motifs it depends on
    deps = {}
    
    # Add existing dependencies
    for nt, obj in grammar.items():
        if not (isinstance(nt, str) and nt.startswith('Motif_')):
            continue
        if not (isinstance(obj, dict) and 'rules' in obj):
            continue
            
        rules = obj['rules']
        if not (isinstance(rules, list) and len(rules) >= 1):
            continue
            
        prod, _ = rules[0]
        motif_refs = [s for s in prod if isinstance(s, str) and s.startswith('Motif_')]
        if motif_refs:
            deps[nt] = set(motif_refs)
    
    # Add the proposed new motif
    if new_motif_name:
        motif_refs = [s for s in new_motif_expansion if isinstance(s, str) and s.startswith('Motif_')]
        if motif_refs:
            deps[new_motif_name] = set(motif_refs)
    
    # Check for cycles using DFS
    visited = set()
    rec_stack = set()
    
    def has_cycle(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False
            
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in deps.get(node, []):
            if has_cycle(neighbor):
                return True
                
        rec_stack.remove(node)
        return False
    
    # Check all nodes
    for node in deps:
        if node not in visited:
            if has_cycle(node):
                return False
    
    return True


def mdl_gain_symbol(ngram: List[str], symbol_sequences: List[List[str]], 
                   mdl_lambda: float = 1.0) -> float:
    """
    Calculate MDL gain for a symbol-level motif.
    
    Args:
        ngram: The n-gram to evaluate as a potential motif
        symbol_sequences: Corpus as symbol sequences  
        mdl_lambda: Cost weight for grammar expansion
        
    Returns:
        MDL gain (higher is better)
    """
    ngram_len = len(ngram)
    if ngram_len <= 1:
        return 0.0
    
    # Count occurrences
    total_occurrences = 0
    for seq in symbol_sequences:
        i = 0
        while i <= len(seq) - ngram_len:
            if seq[i:i+ngram_len] == ngram:
                total_occurrences += 1
                i += ngram_len  # Non-overlapping count
            else:
                i += 1
    
    if total_occurrences == 0:
        return 0.0
    
    # Compression benefit: replace N symbols with 1
    compression_benefit = total_occurrences * (ngram_len - 1)
    
    # Grammar cost: number of symbols in the expansion
    grammar_cost = ngram_len
    
    mdl_gain = compression_benefit - mdl_lambda * grammar_cost
    return mdl_gain


def mine_l2(winners_data: Dict[str, Any], grammar: Dict[str, Any], min_support: int = 3,
           mdl_lambda: float = 1.0, mdl_thresh: float = 2.0, min_n: int = 2, 
           max_n: int = 5) -> List[List[str]]:
    """
    Mine L2 (symbol-level) motifs from symbolized corpus.
    
    Args:
        winners_data: Dictionary with task solutions
        grammar: Current grammar (including L1 motifs)
        min_support: Minimum support for n-grams
        mdl_lambda: Lambda parameter for MDL calculation  
        mdl_thresh: Minimum MDL gain threshold
        min_n: Minimum n-gram size
        max_n: Maximum n-gram size
        
    Returns:
        List of motifs as symbol lists
    """
    print(f"[L2 Miner] Converting corpus to symbol sequences...")
    
    # Convert corpus to symbol-level sequences
    symbol_sequences = symbolize_corpus(winners_data, grammar)
    
    if not symbol_sequences:
        print("[L2 Miner] No symbol sequences generated")
        return []
    
    avg_len = sum(len(seq) for seq in symbol_sequences) / len(symbol_sequences)
    print(f"[L2 Miner] Generated {len(symbol_sequences)} symbol sequences (avg_len={avg_len:.1f})")
    
    # Mine frequent n-grams
    print(f"[L2 Miner] Mining n-grams (n={min_n}-{max_n}, min_support={min_support})...")
    frequent_ngrams = count_ngrams(symbol_sequences, min_n, max_n, min_support)
    
    print(f"[L2 Miner] Found {len(frequent_ngrams)} frequent n-grams")
    
    # Filter by MDL gain and acyclicity
    motifs = []
    for ngram in frequent_ngrams:
        # Check acyclicity
        if not is_acyclic(grammar, ngram):
            continue
            
        # Check MDL gain
        mdl_gain = mdl_gain_symbol(ngram, symbol_sequences, mdl_lambda)
        if mdl_gain >= mdl_thresh:
            motifs.append(ngram)
            print(f"[L2 Miner] Accepted motif: {ngram} (MDL={mdl_gain:.2f})")
    
    # Sort by MDL gain (highest first) for stable ordering
    motifs.sort(key=lambda m: -mdl_gain_symbol(m, symbol_sequences, mdl_lambda))
    
    print(f"[L2 Miner] Final L2 motifs: {len(motifs)}")
    return motifs


if __name__ == "__main__":
    # Test the L2 miner
    import argparse
    
    parser = argparse.ArgumentParser(description="Test L2 motif miner")
    parser.add_argument("--winners", required=True, help="Path to winners JSON file")
    parser.add_argument("--grammar", required=True, help="Path to grammar JSON file")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--mdl-thresh", type=float, default=2.0)
    args = parser.parse_args()
    
    with open(args.winners, 'r') as f:
        winners = json.load(f)
    
    with open(args.grammar, 'r') as f:
        grammar = json.load(f)
    
    motifs = mine_l2(winners, grammar, min_support=args.min_support, 
                    mdl_thresh=args.mdl_thresh)
    
    print(f"\nDiscovered L2 motifs:")
    for i, motif in enumerate(motifs, 1):
        print(f"{i:2d}. {motif}")