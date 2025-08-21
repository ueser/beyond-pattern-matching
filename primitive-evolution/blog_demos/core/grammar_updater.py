"""
Grammar Update System with Acyclicity Enforcement

This module handles updating PCFG grammars by adding new motifs while maintaining
acyclicity and proper probability normalization.
"""

import json
from typing import Dict, List, Tuple, Any, Optional, Set
import copy


TOKENS = set("><+-[]")


def next_motif_index(grammar: Dict[str, Any], level: int) -> int:
    """Find the next available motif index for a specific level."""
    max_index = 0
    prefix = f'L{level}M'
    for nt in grammar:
        if isinstance(nt, str) and nt.startswith(prefix):
            try:
                index = int(nt[len(prefix):])
                max_index = max(max_index, index)
            except (ValueError, IndexError):
                continue
    return max_index + 1


def validate_grammar_structure(grammar: Dict[str, Any]) -> None:
    """Validate that grammar has the expected structure."""
    required_nts = ['Program', 'Head', 'Body', 'Tail', 'Tok']
    
    for nt in required_nts:
        if nt not in grammar:
            raise ValueError(f"Missing required non-terminal: {nt}")
        
        obj = grammar[nt]
        if not isinstance(obj, dict) or 'rules' not in obj:
            raise ValueError(f"Invalid structure for {nt}: missing 'rules'")
        
        rules = obj['rules']
        if not isinstance(rules, list):
            raise ValueError(f"Invalid rules for {nt}: must be list")
        
        for rule in rules:
            if not (isinstance(rule, list) and len(rule) == 2):
                raise ValueError(f"Invalid rule format in {nt}: {rule}")
            prod, weight = rule
            if not isinstance(prod, list):
                raise ValueError(f"Invalid production in {nt}: {prod}")
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Invalid weight in {nt}: {weight}")


def normalize_body_rules(grammar: Dict[str, Any]) -> None:
    """Normalize the weights in Body rules to sum to 1.0."""
    if 'Body' not in grammar:
        raise ValueError("Grammar missing Body non-terminal")
    
    body_rules = grammar['Body']['rules']
    total_weight = sum(rule[1] for rule in body_rules)
    
    if total_weight <= 0:
        raise ValueError("Total weight of Body rules is non-positive")
    
    # Normalize weights
    for rule in body_rules:
        rule[1] = rule[1] / total_weight


def check_acyclicity(grammar: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if the grammar is acyclic by detecting cycles in the dependency graph.
    
    Returns:
        tuple: (is_acyclic, list_of_cycles)
    """
    # Build dependency graph: nt -> set of non-terminals it references
    deps: Dict[str, Set[str]] = {}
    
    for nt, obj in grammar.items():
        if not (isinstance(obj, dict) and 'rules' in obj):
            continue
            
        rules = obj['rules']
        if not rules:
            continue
            
        # For simplicity, only consider the first rule (main expansion)
        prod, _ = rules[0]
        referenced_nts = set()
        
        for symbol in prod:
            if isinstance(symbol, str) and symbol in grammar:
                referenced_nts.add(symbol)
        
        if referenced_nts:
            deps[nt] = referenced_nts
    
    # Detect cycles using DFS
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node, path):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return True
            
        if node in visited:
            return False
            
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in deps.get(node, []):
            if dfs(neighbor, path + [neighbor]):
                rec_stack.remove(node)
                return True
                
        rec_stack.remove(node)
        return False
    
    # Check all nodes
    for node in deps:
        if node not in visited:
            dfs(node, [node])
    
    is_acyclic = len(cycles) == 0
    return is_acyclic, cycles


def add_l1_motifs(grammar: Dict[str, Any], motifs: List[str], 
                 motif_weight: float = 0.1, level: int = 1) -> Dict[str, Any]:
    """
    Add L1 (token-level) motifs to grammar.
    
    Args:
        grammar: Base grammar to extend
        motifs: List of motif token strings
        motif_weight: Weight to assign each motif in Body rules
        level: Motif level (for metadata)
        
    Returns:
        Updated grammar
    """
    grammar = copy.deepcopy(grammar)
    validate_grammar_structure(grammar)
    
    # Add motif definitions
    added_motifs = []
    for motif_tokens in motifs:
        if not motif_tokens:
            continue

        # Generate motif name using new LkMx format
        motif_index = next_motif_index(grammar, level)
        motif_name = f"L{level}M{motif_index}"
        
        # Convert token string to list of individual characters
        expansion = list(motif_tokens)
        
        # Add motif definition
        grammar[motif_name] = {
            'rules': [[expansion, 1.0]],
            'level': level
        }
        
        # Add reference to Body rules
        grammar['Body']['rules'].append([[motif_name], motif_weight])
        
        added_motifs.append(motif_name)
    
    # Normalize Body weights
    normalize_body_rules(grammar)
    
    print(f"[Grammar] Added {len(added_motifs)} L1 motifs: {added_motifs}")
    return grammar


def add_l2_motifs(grammar: Dict[str, Any], motifs: List[List[str]], 
                 motif_weight: float = 0.1, level: int = 2) -> Dict[str, Any]:
    """
    Add L2 (symbol-level) motifs to grammar with acyclicity checking.
    
    Args:
        grammar: Base grammar to extend
        motifs: List of motif symbol lists
        motif_weight: Weight to assign each motif in Body rules
        level: Motif level (for metadata)
        
    Returns:
        Updated grammar
    """
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
        
        # Create temporary grammar with this motif added
        temp_grammar = copy.deepcopy(grammar)
        temp_grammar[motif_name] = {
            'rules': [[motif_symbols, 1.0]],
            'level': level
        }
        
        # Check if adding this motif would create cycles among motifs only
        # Import the L2 is_acyclic function that only checks motif-to-motif dependencies
        from core.miner_l2 import is_acyclic as l2_is_acyclic
        if not l2_is_acyclic(grammar, motif_symbols, motif_name):
            print(f"[Grammar] Skipping motif {motif_symbols} - would create cycles among motifs")
            continue
        
        # Safe to add - update the actual grammar
        grammar[motif_name] = {
            'rules': [[motif_symbols, 1.0]],
            'level': level
        }
        
        # Add reference to Body rules
        grammar['Body']['rules'].append([[motif_name], motif_weight])
        
        added_motifs.append(motif_name)

    
    # Normalize Body weights
    normalize_body_rules(grammar)
    
    print(f"[Grammar] Added {len(added_motifs)} L2 motifs: {added_motifs}")
    return grammar


def remove_unused_motifs(grammar: Dict[str, Any]) -> Dict[str, Any]:
    """Remove motifs that are not referenced anywhere."""
    grammar = copy.deepcopy(grammar)
    
    # Find all referenced non-terminals
    referenced = set()
    
    for nt, obj in grammar.items():
        if not (isinstance(obj, dict) and 'rules' in obj):
            continue
            
        for rule in obj['rules']:
            if len(rule) >= 2:
                prod, _ = rule
                for symbol in prod:
                    if isinstance(symbol, str) and symbol in grammar:
                        referenced.add(symbol)
    
    # Remove unreferenced motifs
    to_remove = []
    for nt in grammar:
        if (isinstance(nt, str) and nt.startswith('Motif_') and 
            nt not in referenced):
            to_remove.append(nt)
    
    for nt in to_remove:
        del grammar[nt]
    
    if to_remove:
        print(f"[Grammar] Removed {len(to_remove)} unused motifs: {to_remove}")
    
    return grammar


def get_grammar_stats(grammar: Dict[str, Any]) -> Dict[str, Any]:
    """Get statistics about the grammar."""
    stats = {
        'total_nonterminals': len(grammar),
        'motif_count': 0,
        'motifs_by_level': {},
        'max_motif_level': 0,
        'body_rules': 0,
        'is_acyclic': True,
        'cycles': []
    }
    
    # Count motifs by level (support both old Motif_X and new LkMx formats)
    for nt, obj in grammar.items():
        if isinstance(nt, str) and (nt.startswith('Motif_') or nt.startswith('L')):
            # Check if it's a motif (either old format Motif_X or new format LkMx)
            is_motif = False
            level = 1  # default level

            if nt.startswith('Motif_'):
                # Old format: get level from object metadata
                is_motif = True
                level = obj.get('level', 1) if isinstance(obj, dict) else 1
            elif nt.startswith('L') and 'M' in nt:
                # New format: extract level from name (e.g., L2M5 -> level 2)
                try:
                    level_part = nt.split('M')[0][1:]  # Remove 'L' and get number before 'M'
                    level = int(level_part)
                    is_motif = True
                except (ValueError, IndexError):
                    pass  # Not a valid LkMx format

            if is_motif:
                stats['motif_count'] += 1
                stats['motifs_by_level'][level] = stats['motifs_by_level'].get(level, 0) + 1
                stats['max_motif_level'] = max(stats['max_motif_level'], level)
    
    # Count Body rules
    if 'Body' in grammar and isinstance(grammar['Body'], dict):
        rules = grammar['Body'].get('rules', [])
        stats['body_rules'] = len(rules)
    
    # Check acyclicity
    is_acyclic, cycles = check_acyclicity(grammar)
    stats['is_acyclic'] = is_acyclic
    stats['cycles'] = cycles
    
    return stats


def save_grammar(grammar: Dict[str, Any], filepath: str) -> None:
    """Save grammar to JSON file with validation."""
    # Validate before saving
    validate_grammar_structure(grammar)
    is_acyclic, cycles = check_acyclicity(grammar)
    
    if not is_acyclic:
        print(f"WARNING: Grammar has cycles: {cycles}")
    
    # Save with nice formatting
    with open(filepath, 'w') as f:
        json.dump(grammar, f, indent=2)
    
    # Print summary
    stats = get_grammar_stats(grammar)
    print(f"[Grammar] Saved to {filepath}")
    print(f"  - Total non-terminals: {stats['total_nonterminals']}")
    print(f"  - Motifs: {stats['motif_count']} (levels: {dict(stats['motifs_by_level'])})")
    print(f"  - Body rules: {stats['body_rules']}")
    print(f"  - Acyclic: {stats['is_acyclic']}")


if __name__ == "__main__":
    # Test the grammar updater
    import argparse
    
    parser = argparse.ArgumentParser(description="Test grammar updater")
    parser.add_argument("--grammar", required=True, help="Path to base grammar")
    parser.add_argument("--test", choices=['l1', 'l2', 'stats'], default='stats',
                       help="Test to run")
    args = parser.parse_args()
    
    with open(args.grammar, 'r') as f:
        grammar = json.load(f)
    
    if args.test == 'stats':
        stats = get_grammar_stats(grammar)
        print("Grammar statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    elif args.test == 'l1':
        # Test adding some L1 motifs
        test_motifs = ["[>+<-]", "++", "<<"]
        updated = add_l1_motifs(grammar, test_motifs)
        print("Updated grammar with L1 motifs")
        
    elif args.test == 'l2':
        # Test adding some L2 motifs  
        test_motifs = [["Motif_1", ">"], ["<", "Motif_2"]]
        updated = add_l2_motifs(grammar, test_motifs)
        print("Updated grammar with L2 motifs")