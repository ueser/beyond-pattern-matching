"""
L1 (Token-Level) Motif Miner with Behavioral Signatures

This module mines motifs at the token level, using behavioral validation
to ensure discovered patterns have consistent effects across different inputs.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import sys
import os

# Add blog_demos to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from brainfuck import BrainfuckInterpreter

TOKENS = set("><+-[]")
LINEAR_TOKENS = set("><+-")
TEST_X = [0, 1, 2, 3, 5, 8]
WINDOW = 5  # number of cells to inspect (centered at origin)


def strip_noops(s: str) -> str:
    """Remove local no-ops until stable: '+-/-+', '<>/><', and empty loop '[]'."""
    prev = None
    while s != prev:
        prev = s
        s = s.replace("+-", "").replace("-+", "")
        s = s.replace("<>", "").replace("><", "")
        s = s.replace("[]", "")
    return s


def is_balanced(s: str) -> bool:
    """Check if brackets are balanced."""
    depth = 0
    for c in s:
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def slice_candidates(programs: List[str], max_linear_len: int = 6) -> List[str]:
    """Extract candidate motifs from programs using syntactic slicing."""
    candidates = []

    for prog in programs:
        # Strip I/O wrapping if present
        body = prog
        if len(prog) >= 2 and prog[0] == ',' and prog[-1] == '.':
            body = prog[1:-1]

        # Always strip no-ops - we don't want to mine meaningless patterns
        body = strip_noops(body)
        
        if not body:
            continue
            
        # Find all balanced loops with context
        i = 0
        while i < len(body):
            if body[i] == '[':
                # Find matching bracket
                depth = 1
                j = i + 1
                while j < len(body) and depth > 0:
                    if body[j] == '[':
                        depth += 1
                    elif body[j] == ']':
                        depth -= 1
                    j += 1
                
                if depth == 0:  # Found balanced loop
                    loop = body[i:j]
                    candidates.append(loop)
                    
                    # Loop + following context (>, <)
                    if j < len(body) and body[j] in '><':
                        candidates.append(loop + body[j])
                    
                    # Preceding context + loop
                    if i > 0 and body[i-1] in '><':
                        candidates.append(body[i-1] + loop)
                    
                    # Adjacent loops
                    if j < len(body) and body[j] == '[':
                        k = j + 1
                        depth2 = 1
                        while k < len(body) and depth2 > 0:
                            if body[k] == '[':
                                depth2 += 1
                            elif body[k] == ']':
                                depth2 -= 1
                            k += 1
                        if depth2 == 0:
                            candidates.append(body[i:k])
                    
                    i = j
                else:
                    i += 1
            else:
                # Extract linear segments and all their subsequences
                start = i
                while i < len(body) and body[i] in LINEAR_TOKENS:
                    i += 1

                # Extract all subsequences of the linear segment
                if start < i:
                    segment = body[start:i]
                    # Add all subsequences of length 2 to max_linear_len
                    for length in range(2, min(len(segment) + 1, max_linear_len + 1)):
                        for pos in range(len(segment) - length + 1):
                            subseq = segment[pos:pos + length]
                            candidates.append(subseq)

                if start == i:
                    i += 1
    
    return candidates


def run_harness(chunk: str, step_limit: int = 1000) -> Optional[Tuple[bool, int, Dict[int, Tuple[int, int]]]]:
    """
    Run chunk on small inputs, return behavioral signature or None on failure.
    
    Returns:
        tuple: (drain_entry?, delta_p, coeffs_by_offset) where coeffs_by_offset 
               maps relative cell offset j to (a,b) fitted in c_j' ~= a*x + b
    """
    # Disallow I/O in chunks
    if any(c in chunk for c in ',.!'):
        return None
    if not is_balanced(chunk):
        return None

    xs = TEST_X
    outs_by_x: List[Tuple[int, List[int]]] = []  # (final_ptr, cells)

    for x in xs:
        itp = BrainfuckInterpreter(memory_size=64)
        itp.memory[0] = x
        # Add poison values to neighbors for better detection
        if len(itp.memory) > 1:
            itp.memory[1] = 11  # right neighbor
        if len(itp.memory) > 63:
            itp.memory[63] = 7  # left neighbor (wrapping)
            
        try:
            itp.run_step(chunk, max_steps=step_limit, preserve_memory=False)
        except Exception:
            return None
        if itp.hit_step_limit:
            return None
        
        final_ptr = itp.pointer
        # Gather a window of cells centered at origin (0)
        k = WINDOW // 2
        cells: List[int] = []
        for j in range(-k, k + 1):
            idx = j % len(itp.memory)
            cells.append(itp.memory[idx])
        outs_by_x.append((final_ptr, cells))

    # Pointer delta must be constant across all inputs
    deltas = [fp for fp, _ in outs_by_x]
    if any(d != deltas[0] for d in deltas):
        return None
    delta_p = deltas[0]

    # Check if entry cell drains (becomes 0 for all x)
    k = WINDOW // 2
    drains = all(cells[k] == 0 for _, cells in outs_by_x)

    # Fit linear coefficients per offset j: cell[j] = a*x + b
    coeffs: Dict[int, Tuple[int, int]] = {}
    for j in range(-k, k + 1):
        ys = [cells[k + j] for _, cells in outs_by_x]
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        
        if denom == 0:
            a = 0.0
            b = round(sy / n) if n > 0 else 0
        else:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
        coeffs[j] = (int(round(a)), int(round(b)))

    return (bool(drains), int(delta_p), coeffs)


def cluster_by_signature(candidates_with_sigs: List[Tuple[str, Tuple[bool, int, Dict[int, Tuple[int, int]]]]]) -> Dict[Tuple[Any, ...], List[str]]:
    """Group candidates by their behavioral signature."""
    groups = defaultdict(list)
    
    for chunk, sig in candidates_with_sigs:
        if sig is None:
            continue
        drains, dp, coeffs = sig
        # Create a hashable signature key
        coeff_items = tuple(sorted(coeffs.items()))
        sig_key = (drains, dp, coeff_items)
        groups[sig_key].append(chunk)
    
    return dict(groups)


def select_shortest_high_support(chunks: List[str], min_support: int, counter: Counter) -> Optional[str]:
    """Select the shortest chunk with highest support from a group."""
    valid_chunks = [c for c in chunks if counter[c] >= min_support]
    if not valid_chunks:
        return None
    
    # Sort by length (shorter first), then by support (higher first)
    valid_chunks.sort(key=lambda c: (len(c), -counter[c]))
    return valid_chunks[0]


def mdl_gain_token(chunk: str, programs: List[str], mdl_lambda: float = 1.5) -> float:
    """
    Calculate MDL gain for a token-level motif.
    
    MDL gain = (compression_benefit) - λ * (grammar_cost)
    where:
    - compression_benefit = original_length - compressed_length
    - grammar_cost = length of motif expansion
    """
    # Count occurrences and calculate compression
    total_original_len = sum(len(prog) for prog in programs)
    chunk_len = len(chunk)
    
    # Count non-overlapping occurrences
    total_occurrences = 0
    for prog in programs:
        body = prog
        if len(prog) >= 2 and prog[0] == ',' and prog[-1] == '.':
            body = prog[1:-1]
        body = strip_noops(body)
        
        # Count non-overlapping matches
        i = 0
        while i <= len(body) - chunk_len:
            if body[i:i+chunk_len] == chunk:
                total_occurrences += 1
                i += chunk_len
            else:
                i += 1
    
    if total_occurrences == 0:
        return 0.0
    
    # Calculate compression benefit: each occurrence saves (chunk_len - 1) characters
    # (we replace N characters with 1 motif symbol)
    compression_benefit = total_occurrences * (chunk_len - 1)
    
    # Grammar cost: adding the motif definition
    grammar_cost = chunk_len
    
    mdl_gain = compression_benefit - mdl_lambda * grammar_cost
    return mdl_gain


def mine_l1(winners_data: Dict[str, Any], step_limit: int = 1000, min_support: int = 3,
           mdl_lambda: float = 1.5, mdl_thresh: float = 3.0, max_linear_len: int = 6,
           min_len: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Mine L1 (token-level) motifs from winners using behavioral validation and MDL scoring.
    
    Args:
        winners_data: Dictionary with task solutions
        step_limit: Maximum steps for behavioral harness
        min_support: Minimum support count for motifs  
        mdl_lambda: Lambda parameter for MDL calculation
        mdl_thresh: Minimum MDL gain threshold
        max_linear_len: Maximum length for linear segments
        min_len: Minimum motif length
        
    Returns:
        List of discovered motifs as token strings
    """
    # Collect all winner programs
    programs = []
    for task_name, task_data in winners_data.items():
        solutions = task_data.get('solutions', [])
        programs.extend(solutions)
    
    print(f"[L1 Miner] Processing {len(programs)} winner programs")
    
    # Slice candidates
    candidates = slice_candidates(programs, max_linear_len)
    candidates = [c for c in candidates if len(c) >= min_len]
    
    print(f"[L1 Miner] Extracted {len(candidates)} candidates (min_len={min_len})")
    
    # Count candidate frequencies
    counter = Counter(candidates)
    
    # Evaluate candidates with behavioral harness
    candidates_with_sigs = []
    for chunk in counter:
        sig = run_harness(chunk, step_limit)
        if sig is not None:
            candidates_with_sigs.append((chunk, sig))
    
    print(f"[L1 Miner] {len(candidates_with_sigs)} candidates passed behavioral validation")
    
    # Cluster by behavioral signature
    groups = cluster_by_signature(candidates_with_sigs)
    
    print(f"[L1 Miner] Clustered into {len(groups)} behavioral groups")
    
    # Select representatives and filter by MDL
    motifs = []
    motif_info = []
    for _, chunks in groups.items():
        # Select best representative from this behavioral group
        rep = select_shortest_high_support(chunks, min_support, counter)
        if rep is None:
            continue

        # Check MDL gain
        mdl_gain = mdl_gain_token(rep, programs, mdl_lambda)
        if mdl_gain >= mdl_thresh:
            motifs.append(rep)
            motif_info.append({
                'pattern': list(rep),
                'support': counter[rep],
                'mdl_gain': mdl_gain,
                'level': 1
            })
            print(f"[L1 Miner] Accepted motif: '{rep}' (support={counter[rep]}, MDL={mdl_gain:.2f})")

    print(f"[L1 Miner] Final L1 motifs: {len(motifs)}")
    return motifs, motif_info


if __name__ == "__main__":
    # Test the L1 miner
    import argparse
    
    parser = argparse.ArgumentParser(description="Test L1 motif miner")
    parser.add_argument("--winners", required=True, help="Path to winners JSON file")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--mdl-thresh", type=float, default=3.0)
    args = parser.parse_args()
    
    with open(args.winners, 'r') as f:
        winners = json.load(f)
    
    motifs = mine_l1(winners, min_support=args.min_support, mdl_thresh=args.mdl_thresh)
    
    print("\nDiscovered L1 motifs:")
    for i, motif in enumerate(motifs, 1):
        print(f"{i:2d}. '{motif}'")