"""
    ## **2) Mine** **motifs** **(candidate building blocks) from winners**



**Syntactic slicing:**

- Extract all **balanced loops** [...] (with nesting) and small **loop+context** ([...]>', '<' + [...], '[...]' '[...]').

- Also extract short **linear segments** between loops (≤ L tokens) — these help form non-loop rules (like +-runs, pointer shifts).




**Canonicalization:**

- Strip local no-ops (+- / -+ / <> / >< pairs).

- Normalize **pointer origin** (treat loop-entry cell as 0).

- Run-length encode +/- and </> into counts: e.g., [+++]> → (+^3)(>^1).




**Behavioral signature (tiny harness):**

- For each candidate chunk C:

    - probe inputs x ∈ {0,1,2,3,5,8} in a small tape window (e.g., 5 cells),

    - record: drains_entry?, Δp, and **cell-wise linear effect coefficients** (fit c_j' = a_j x + b_j with small residual).


- This yields a **behavioral key**: (drain=True, Δp=0, coeffs={(+1): (1,0)}) etc.




**Cluster/deduplicate:**

Group candidates with the same behavioral key (within rounding). Keep the **shortest** and the **most frequent** representative.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Make blog_demos a module root for imports like core.* and tasks.*
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from brainfuck import BrainfuckInterpreter

TOKENS = set("><+-[]")
LINEAR_TOKENS = set("><+-")
TEST_X = [0, 1, 2, 3, 5, 8]
WINDOW = 5  # number of cells to inspect (centered at origin index 0)


# -----------------------------
# Utilities
# -----------------------------

def strip_noops(s: str) -> str:
    """Remove local no-ops until stable: '+-/-+', '<>/><', and empty loop '[]'.
    Note: '[]' is a no-op only when the entry cell is zero; otherwise it would loop.
    For motif mining we remove it as a degenerate chunk; loops that actually act
    (e.g., '[<]' or '[>]') are NOT removed here because they change pointer state.
    """
    prev = None
    while s != prev:
        prev = s
        s = s.replace("+-", "").replace("-+", "")
        s = s.replace("<>", "").replace("><", "")
        s = s.replace("[]", "")
    return s


def is_balanced(s: str) -> bool:
    st = []
    for c in s:
        if c == '[':
            st.append(c)
        elif c == ']':
            if not st:
                return False
            st.pop()
    return not st


def bracket_pairs(code: str) -> Dict[int, int]:
    stack = []
    pairs: Dict[int, int] = {}
    for i, c in enumerate(code):
        if c == '[':
            stack.append(i)
        elif c == ']':
            if not stack:
                continue
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    return pairs


def slice_candidates_from_code(code: str, max_linear_len: int = 6) -> List[str]:
    """Deprecated: use slice_candidates_from_symbols instead."""
    return []

    # Loops and loop contexts
# -----------------------------
# Motif rewrite helpers (from 04_motif_rewrite)
# -----------------------------

def flatten_motif_to_tokens(grammar: Dict[str, Any], nt: str, _memo: Dict[str, List[str]], _stack: Optional[set] = None) -> List[str]:
    if nt in _memo:
        return _memo[nt]
    if _stack is None:
        _stack = set()
    if nt in _stack:
        _memo[nt] = []
        return []
    obj = grammar.get(nt)
    if not isinstance(obj, dict) or 'rules' not in obj:
        return [nt]
    rules = obj['rules']
    if not (isinstance(rules, list) and len(rules) >= 1 and isinstance(rules[0], list) and len(rules[0]) == 2):
        return [nt]
    prod, _w = rules[0]
    _stack.add(nt)
    out: List[str] = []
    for sym in prod:
        if isinstance(sym, str) and sym.startswith('Motif_'):
            out.extend(flatten_motif_to_tokens(grammar, sym, _memo, _stack))
        elif isinstance(sym, str):
            for ch in sym:
                if ch in TOKENS:
                    out.append(ch)
    _stack.remove(nt)
    _memo[nt] = out
    return out


def build_motif_key_map(grammar: Dict[str, Any]) -> Dict[str, str]:
    key_map: Dict[str, str] = {}
    memo: Dict[str, List[str]] = {}
    total_nt = 0
    skipped_cycles = 0
    for nt, obj in grammar.items():
        if not (isinstance(nt, str) and nt.startswith('Motif_')):
            continue
        total_nt += 1
        if not (isinstance(obj, dict) and 'rules' in obj):
            continue
        toks = flatten_motif_to_tokens(grammar, nt, memo)
        if toks and all(t in TOKENS for t in toks):
            key_map[''.join(toks)] = nt
        else:
            skipped_cycles += 1
    print(f"[build_motif_key_map] motifs in grammar: {total_nt}; keys built: {len(key_map)}; skipped(cycle/invalid): {skipped_cycles}")
    return key_map


def motifize_body(body: str, key_map: Dict[str, str]) -> List[str]:
    ordered = sorted(key_map.keys(), key=len, reverse=True)
    out: List[str] = []
    i = 0
    n = len(body)
    while i < n:
        matched = False
        for k in ordered:
            if body.startswith(k, i):
                out.append(key_map[k])
                i += len(k)
                matched = True
                break
        if not matched:
            out.append(body[i])
            i += 1
    return out


def expand_symbols_to_tokens(seq: List[str], grammar: Dict[str, Any]) -> str:
    memo: Dict[str, List[str]] = {}
    out: List[str] = []
    for s in seq:
        if isinstance(s, str) and s.startswith('Motif_'):
            out.extend(flatten_motif_to_tokens(grammar, s, memo))
        else:
            # explode any multi-char terminal into tokens
            if isinstance(s, str) and len(s) > 1:
                for ch in s:
                    if ch in TOKENS:
                        out.append(ch)
            else:
                out.append(s)
    return ''.join(out)

# -----------------------------
# Symbol-level candidate slicing (supports motif tokens)
# -----------------------------

def _segments_symbols(symbols: List[str]) -> List[List[str]]:
    segs: List[List[str]] = []
    i, n = 0, len(symbols)
    while i < n:
        if symbols[i] == '[':
            depth, j = 1, i + 1
            while j < n and depth:
                if symbols[j] == '[':
                    depth += 1
                elif symbols[j] == ']':
                    depth -= 1
                j += 1
            segs.append(symbols[i:j])
            i = j
        else:
            j = i
            while j < n and symbols[j] != '[':
                j += 1
            segs.append(symbols[i:j])
            i = j
    return [s for s in segs if s]


def slice_candidates_from_symbols(symbols: List[str], max_linear_len: int = 6) -> List[List[str]]:
    cands: List[List[str]] = []
    segs = _segments_symbols(symbols)
    n = len(symbols)

    # build index map from segment starts to allow context slicing
    starts = []
    pos = 0
    for seg in segs:
        starts.append(pos)
        pos += len(seg)

    for idx, seg in enumerate(segs):
        start = starts[idx]
        if seg and seg[0] == '[' and seg[-1] == ']':
            # pure loop segment
            cands.append(seg)
            # loop followed by '>' or '<'
            after = start + len(seg)
            if after < n and symbols[after] in ('>', '<'):
                cands.append(seg + [symbols[after]])
            # '<' or '>' before loop
            if start - 1 >= 0 and symbols[start - 1] in ('<', '>'):
                cands.append([symbols[start - 1]] + seg)
            # adjacent loops
            if idx + 1 < len(segs) and segs[idx + 1] and segs[idx + 1][0] == '[':
                cands.append(seg + segs[idx + 1])
        else:
            # linear segment
            if 0 < len(seg) <= max_linear_len:
                cands.append(seg)

    return cands


def run_harness(chunk: str, step_limit: int = 1000) -> Optional[Tuple[bool, int, Dict[int, Tuple[int, int]]]]:
    """Run chunk on small inputs, return (drain_entry?, delta_p, coeffs_by_offset) or None on failure.
    coeffs_by_offset maps relative cell offset j to (a,b) fitted in c_j' ~= a*x + b.
    """
    # disallow I/O in chunks
    if any(c in chunk for c in ',.!'):
        return None
    if not is_balanced(chunk):
        return None

    xs = TEST_X
    outs_by_x: List[Tuple[int, List[int]]] = []  # (final_ptr, cells)

    for x in xs:
        itp = BrainfuckInterpreter(memory_size=64)
        itp.memory[0] = x
        try:
            itp.run_step(chunk, max_steps=step_limit, preserve_memory=False)
        except Exception:
            return None
        if itp.hit_step_limit:
            return None
        final_ptr = itp.pointer
        # gather a window of cells [-k..+k] mapped to indices 0..len-1 circularly
        k = WINDOW // 2
        cells: List[int] = []
        for j in range(-k, k + 1):
            idx = (j) % len(itp.memory)
            cells.append(itp.memory[idx])
        outs_by_x.append((final_ptr, cells))

    # pointer delta must be constant across xs
    deltas = [fp for fp, _ in outs_by_x]
    if any(d != deltas[0] for d in deltas):
        return None
    delta_p = deltas[0]

    # drains entry if origin cell becomes 0 for all x
    k = WINDOW // 2
    drains = all(cells[k] == 0 for _, cells in outs_by_x)

    # Fit linear coeffs per offset j
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
            b = round(sy / n)
        else:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
        coeffs[j] = (int(round(a)), int(round(b)))

    return (bool(drains), int(delta_p), coeffs)


def mine_motifs_from_winners(winners_path: str, max_linear_len: int = 6, step_limit: int = 1000,
                             min_support: int = 2, min_len: int = 3, rewrite_grammar_path: Optional[str] = None,
                             mdl_lambda: float = 1.0, mdl_tau: float = 2.0) -> Tuple[Dict[Tuple[Any, ...], str], Counter]:
    print(f"[mine] loading winners from: {winners_path}")
    with open(winners_path, 'r') as f:
        winners = json.load(f)
    print(f"[mine] tasks in winners: {len(winners)}")

    grammar = None
    if rewrite_grammar_path and os.path.exists(rewrite_grammar_path):
        try:
            with open(rewrite_grammar_path, 'r') as gf:
                grammar = json.load(gf)
            print(f"[mine] loaded rewrite grammar: {rewrite_grammar_path} (NTs={sum(isinstance(v, dict) and 'rules' in v for v in grammar.values())})")
        except Exception as e:
            print(f"[mine] failed to load rewrite grammar {rewrite_grammar_path}: {e}")
            grammar = None
    else:
        print("[mine] rewrite grammar not provided or not found; mining on primitives only")

    key_map = build_motif_key_map(grammar) if grammar else {}

    # Gather all winner programs across tasks
    programs: List[str] = []
    for tn, obj in winners.items():
        sols = obj.get('solutions') or []
        programs.extend(sols)
    print(f"[mine] total winner programs: {len(programs)}")

    # Build symbol-level sequences per program
    symbol_sequences: List[List[str]] = []
    for prog in programs:
        body = prog
        if len(prog) >= 2 and prog[0] == ',' and prog[-1] == '.':
            body = prog[1:-1]
        raw = strip_noops(body)
        seq = motifize_body(raw, key_map) if key_map else list(raw)
        symbol_sequences.append(seq)
    if symbol_sequences:
        lens = [len(s) for s in symbol_sequences]
        print(f"[mine] rewrite complete: {len(symbol_sequences)} sequences; avg_len={sum(lens)/len(lens):.2f}; sample={symbol_sequences[0][:20]}")

    # Slice candidates from symbol sequences and count
    cand_counter: Counter = Counter()
    motifful = 0
    for seq in symbol_sequences:
        for c in slice_candidates_from_symbols(seq, max_linear_len=max_linear_len):
            if len(c) >= min_len:
                key = ' '.join(c)
                cand_counter[key] += 1
                if 'Motif_' in key:
                    motifful += 1
    total_cands = len(cand_counter)
    print(f"[mine] candidates counted: {total_cands} (min_len={min_len}); with Motif_*: {motifful}")

    # Evaluate candidates: flatten symbols to tokens and run harness
    by_sig: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    eval_total = 0
    kept_total = 0
    for cand_key, cnt in cand_counter.items():
        cand_symbols = cand_key.split(' ')
        flat = expand_symbols_to_tokens(cand_symbols, grammar) if grammar else ''.join(cand_symbols)
        eval_total += 1
        sig = run_harness(flat, step_limit=step_limit)
        if sig is None:
            continue
        drains, dp, coeffs = sig
        coeff_items = tuple(sorted(coeffs.items()))
        sig_key = (drains, dp, coeff_items)
        by_sig[sig_key].append(cand_key)
        kept_total += 1
    print(f"[mine] evaluated candidates: {eval_total}; kept (valid sig): {kept_total}; groups: {len(by_sig)}")

    # Pick representative per signature: shortest (in symbols), then most frequent
    # MDL/coverage: estimate savings by replacing occurrences of candidate with a single symbol
    def mdl_gain_for_candidate(cand_syms: List[str]) -> int:
        # coverage in symbol space (how many times this exact key appears across sequences)
        key = ' '.join(cand_syms)
        support = cand_counter.get(key, 0)
        if support <= 0:
            return -10**9
        L = len(cand_syms)
        saved = support * (L - 1)  # each occurrence: L tokens -> 1 symbol, saves L-1
        rule_size = L  # simple proxy: rule length in symbols
        gain = int(round(saved - mdl_lambda * rule_size))
        return gain

    # Build representatives with both support and MDL gain thresholds
    rep_by_sig: Dict[Tuple[Any, ...], str] = {}
    reps_with_motifs = 0
    for sig_key, chunks in by_sig.items():
        chunks.sort(key=lambda s: (len(s.split(' ')), -cand_counter[s]))
        for choice in chunks:
            if cand_counter[choice] < min_support:
                continue
            syms = choice.split(' ')
            gain = mdl_gain_for_candidate(syms)
            if gain >= mdl_tau:
                rep_by_sig[sig_key] = choice
                if any(tok.startswith('Motif_') for tok in syms):
                    reps_with_motifs += 1
                break
    print(f"[mine] representatives selected: {len(rep_by_sig)} (min_support={min_support}, mdl_tau={mdl_tau}); with Motif_* reps: {reps_with_motifs}")

    return rep_by_sig, cand_counter


def expand_grammar(base_grammar_path: str, out_path: str, motifs: List[str], motif_weight: float = 0.1,
                   rewrite_grammar_path: Optional[str] = None, min_final_symbols: int = 3) -> None:
    print(f"[expand] base: {base_grammar_path}; out: {out_path}; motifs_in: {len(motifs)}")
    with open(base_grammar_path, 'r') as f:
        grammar = json.load(f)
    # Optionally load a rewrite grammar that defines previously discovered motifs
    rewrite_grammar = None
    if rewrite_grammar_path is None:
        # try default enriched_h_cfg.json alongside repo root
        root = os.path.dirname(os.path.dirname(__file__))
        candidate = os.path.join(root, 'motifs', 'enriched_h_cfg.json')
        if os.path.exists(candidate):
            rewrite_grammar_path = candidate
    if rewrite_grammar_path and os.path.exists(rewrite_grammar_path):
        try:
            with open(rewrite_grammar_path, 'r') as rf:
                rewrite_grammar = json.load(rf)
            print(f"[expand] loaded prior enriched grammar: {rewrite_grammar_path}")
        except Exception as e:
            print(f"[expand] failed to load prior enriched grammar: {e}")
            rewrite_grammar = None

    # Ensure schema {NT: {"rules": [ [ [symbols], weight], ... ] }}
    def has_rules(obj):
        return isinstance(obj, dict) and 'rules' in obj

    for nt in list(grammar.keys()):
        if not has_rules(grammar[nt]):
            # normalize compact form if present
            rules = grammar[nt]
            grammar[nt] = { 'rules': rules }

    # Build maps of existing motifs
    # 1) token-string -> NT (for decomposing raw token runs)
    existing_token_map: Dict[str, str] = {}
    # 2) symbol-sequence (space-joined) -> NT (to avoid duplicates when expansion includes Motif_* symbols)
    existing_symbol_map: Dict[str, str] = {}

    motif_index_max = 0
    # incorporate existing motifs from base grammar
    count_existing = 0
    for nt, obj in grammar.items():
        if isinstance(nt, str) and nt.startswith('Motif_') and has_rules(obj):
            try:
                num = int(nt.split('_', 1)[1])
                motif_index_max = max(motif_index_max, num)
            except Exception:
                pass
            rules = obj['rules']
            if isinstance(rules, list) and len(rules) == 1 and isinstance(rules[0], list) and len(rules[0]) == 2:
                prod, w = rules[0]
                if w == 1.0 and isinstance(prod, list):
                    sym_key = ' '.join(prod)
                    existing_symbol_map[sym_key] = nt
                    count_existing += 1
                    if all(isinstance(sym, str) and len(sym) == 1 and sym in TOKENS for sym in prod):
                        existing_token_map[''.join(prod)] = nt
    print(f"[expand] base motifs indexed: {count_existing}; next_id>{motif_index_max}")
    # incorporate motifs from rewrite grammar (previously discovered), to avoid clashes and enable reuse
    if rewrite_grammar and isinstance(rewrite_grammar, dict):
        count_prior = 0
        for nt, obj in rewrite_grammar.items():
            if isinstance(nt, str) and nt.startswith('Motif_') and isinstance(obj, dict) and 'rules' in obj:
                rules = obj['rules']
                if isinstance(rules, list) and len(rules) == 1 and isinstance(rules[0], list) and len(rules[0]) == 2:
                    prod, w = rules[0]
                    if w == 1.0 and isinstance(prod, list):
                        sym_key = ' '.join(prod)
                        existing_symbol_map.setdefault(sym_key, nt)
                        count_prior += 1
                        if all(isinstance(sym, str) and len(sym) == 1 and sym in TOKENS for sym in prod):
                            existing_token_map.setdefault(''.join(prod), nt)
                try:
                    num = int(nt.split('_', 1)[1])
                    motif_index_max = max(motif_index_max, num)
                except Exception:
                    pass
        print(f"[expand] prior motifs indexed: {count_prior}; next_id>{motif_index_max}")

    # Helper: decompose a raw token string using known token-level motifs (longest-first)
    def decompose_tokens(s: str) -> List[str]:
        if not existing_token_map:
            return list(s)
        keys = sorted(existing_token_map.keys(), key=len, reverse=True)
        out: List[str] = []
        i = 0
        while i < len(s):
            matched = False
            for k in keys:
                if s.startswith(k, i):
                    out.append(existing_token_map[k])
                    i += len(k)
                    matched = True
                    break
            if not matched:
                out.append(s[i])
                i += 1
        return out

    # Convert an input motif spec (either raw tokens or space-separated symbols) into an expansion list
    def parse_motif_spec(s: str) -> List[str]:
        if 'Motif_' in s or ' ' in s:
            syms = [t for t in s.split(' ') if t]
            out: List[str] = []
            buf_tokens: List[str] = []
            def flush_buf():
                nonlocal out, buf_tokens
                if buf_tokens:
                    out.extend(decompose_tokens(''.join(buf_tokens)))
                    buf_tokens = []
            for t in syms:
                if isinstance(t, str) and t.startswith('Motif_'):
                    flush_buf(); out.append(t)
                elif len(t) == 1 and t in TOKENS:
                    buf_tokens.append(t)
                else:
                    # unknown -> flush and skip
                    flush_buf()
            flush_buf()
            return out
        else:
            return decompose_tokens(s)

    # Deduplicate and add motifs (prefer longer first for stable naming)
    seen_new: set[str] = set()
    added_nts: List[str] = []
    next_idx = motif_index_max + 1 if motif_index_max > 0 else 1

    # Determine target level: 1 if no rewrite grammar (L1), else 2 (L2)
    target_level = 1 if not rewrite_grammar else 2
    allowed_level_motifs: set[str] = set()
    if rewrite_grammar and isinstance(rewrite_grammar, dict):
        for k, v in rewrite_grammar.items():
            if isinstance(k, str) and k.startswith('Motif_') and isinstance(v, dict) and 'rules' in v:
                allowed_level_motifs.add(k)

    def enforce_level(expansion: List[str]) -> List[str]:
        if not rewrite_grammar:
            return expansion  # L1: tokens-only expansions already ensured by parse/decompose
        out: List[str] = []
        memo: Dict[str, List[str]] = {}
        for sym in expansion:
            if isinstance(sym, str) and sym.startswith('Motif_'):
                if sym in allowed_level_motifs:
                    out.append(sym)
                else:
                    # inline to tokens using rewrite_grammar to avoid introducing L2/L2 cycles
                    toks = flatten_motif_to_tokens(rewrite_grammar, sym, memo)
                    out.extend(toks)
            else:
                out.append(sym)
        return out

    reused_count = 0
    new_count = 0
    for s in sorted(motifs, key=len, reverse=True):
        exp = parse_motif_spec(s)
        exp = enforce_level(exp)
        # Enforce minimum final symbol length to avoid trivial wrappers like [Motif_X]
        if len(exp) < min_final_symbols:
            continue
        sym_key = ' '.join(exp)
        if sym_key in existing_symbol_map:
            nt = existing_symbol_map[sym_key]
            added_nts.append(nt)
            reused_count += 1
            continue
        if sym_key in seen_new:
            continue
        seen_new.add(sym_key)
        nt = f"Motif_{next_idx}"
        while nt in grammar:
            next_idx += 1
            nt = f"Motif_{next_idx}"
        grammar[nt] = { 'level': target_level, 'rules': [ [ exp, 1.0 ] ] }
        existing_symbol_map[sym_key] = nt
        # update token map if pure tokens
        if all(len(x) == 1 and x in TOKENS for x in exp):
            existing_token_map[''.join(exp)] = nt
        added_nts.append(nt)
        new_count += 1
        next_idx += 1
    print(f"[expand] motifs reused: {reused_count}; new: {new_count}; total_added_refs: {len(added_nts)}; level={target_level}")

    # Hook motifs into Body uniquely
    if 'Body' not in grammar or 'rules' not in grammar['Body']:
        raise RuntimeError("Base grammar missing Body.rules")
    body_rules = grammar['Body']['rules']
    existing_refs = set()
    for rule in body_rules:
        if isinstance(rule, list) and len(rule) == 2 and isinstance(rule[0], list) and len(rule[0]) == 1:
            existing_refs.add(rule[0][0])

    # Append motifs to Body uniquely
    for nt in added_nts:
        if nt not in existing_refs:
            body_rules.append([ [nt], float(motif_weight) ])
    # Renormalize Body.rules to sum to 1.0 after edits
    total_w = 0.0
    for rule in body_rules:
        if isinstance(rule, list) and len(rule) == 2 and isinstance(rule[1], (int, float)):
            total_w += float(rule[1])
    if total_w > 0:
        for rule in body_rules:
            if isinstance(rule, list) and len(rule) == 2 and isinstance(rule[1], (int, float)):
                rule[1] = float(rule[1]) / total_w
    else:
        # Fallback: assign uniform weights
        k = len(body_rules)
        if k:
            w = 1.0 / k
            for i in range(k):
                if isinstance(body_rules[i], list) and len(body_rules[i]) == 2:
                    body_rules[i][1] = w


    # Resolve undefined motif references by copying from rewrite_grammar, or inlining tokens
    def collect_undefined_refs(g: Dict[str, Any]) -> set:
        refs = set()
        for name, obj in g.items():
            if not (isinstance(obj, dict) and 'rules' in obj):
                continue
            rules = obj['rules']
            if not rules: continue
            prod, _ = rules[0]
            for s in prod:
                if isinstance(s, str) and s.startswith('Motif_') and s not in g:
                    refs.add(s)
        return refs

    copied = 0
    inlined = 0
    if rewrite_grammar:
        # Iteratively copy definitions from rewrite grammar when available
        while True:
            undefined = collect_undefined_refs(grammar)
            if not undefined:
                break
            progressed = False
            for ref in list(undefined):
                obj = rewrite_grammar.get(ref) if isinstance(rewrite_grammar, dict) else None
                if isinstance(obj, dict) and 'rules' in obj:
                    grammar[ref] = obj  # copy definition verbatim
                    copied += 1
                    progressed = True
            if not progressed:
                break
        # Inline any remaining undefined using flattened tokens
        if undefined:
            memo: Dict[str, List[str]] = {}
            inline_map: Dict[str, List[str]] = {}
            for ref in list(undefined):
                toks = flatten_motif_to_tokens(rewrite_grammar, ref, memo)
                if toks:
                    inline_map[ref] = toks
            if inline_map:
                for name, obj in list(grammar.items()):
                    if not (isinstance(obj, dict) and 'rules' in obj):
                        continue
                    prod, w = obj['rules'][0]
                    new_prod: List[str] = []
                    for s in prod:
                        if isinstance(s, str) and s in inline_map:
                            new_prod.extend(inline_map[s])
                            inlined += 1
                        else:
                            new_prod.append(s)
                    obj['rules'][0] = [new_prod, w]
    undefined_final = collect_undefined_refs(grammar)
    print(f"[expand] undefined motif refs resolved: copied={copied}, inlined={inlined}, remaining={len(undefined_final)}")

    # Write out the enriched grammar
    with open(out_path, 'w') as f:
        json.dump(grammar, f, indent=2)

    # Post-write validation summary
    def validate(pth: str):
        g = json.load(open(pth, 'r'))
        TOK = set(">+<-[]")
        # nested count
        nested = 0
        total_m = 0
        for nt, obj in g.items():
            if not (isinstance(nt, str) and nt.startswith('Motif_')):
                continue
            if not (isinstance(obj, dict) and 'rules' in obj):
                continue
            total_m += 1
            prod, _ = obj['rules'][0]
            if any(isinstance(s, str) and s.startswith('Motif_') for s in prod):
                nested += 1
        # cycles and non-terminating
        graph = {}
        for nt, obj in g.items():
            if not (isinstance(obj, dict) and 'rules' in obj):
                continue
            prod, _ = obj['rules'][0]
            refs = [s for s in prod if isinstance(s, str) and s.startswith('Motif_')]
            if refs:
                graph.setdefault(nt, set()).update(refs)
        visited, stack = set(), set()
        def dfs(u):
            if u in stack: return True
            if u in visited: return False
            visited.add(u); stack.add(u)
            for v in graph.get(u, []):
                if dfs(v): return True
            stack.remove(u)
            return False
        cycles = [nt for nt in graph if dfs(nt)]
        # flatten
        memo = {}
        def flatten(nt):
            if nt in memo: return memo[nt]
            obj = g.get(nt)
            if not (isinstance(obj, dict) and 'rules' in obj): return [nt]
            prod, _ = obj['rules'][0]
            out = []
            for s in prod:
                if isinstance(s, str) and s.startswith('Motif_'):
                    out.extend(flatten(s))
                elif isinstance(s, str):
                    for ch in s:
                        if ch in TOK:
                            out.append(ch)
            memo[nt] = out
            return out
        nonterm = []
        for nt in graph:
            toks = flatten(nt)
            if not all(t in TOK for t in toks):
                nonterm.append(nt)
        print(f"[validate] motifs: {total_m}; nested: {nested}; cycles: {len(cycles)}; non-terminating: {len(nonterm)}")


def main():
    ap = argparse.ArgumentParser(description="Mine motifs from winners and expand grammar")
    ap.add_argument('--winners', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'runs', 'cfg_best.json'))
    ap.add_argument('--base-grammar', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'motifs', 'minimal_cfg.json'))
    ap.add_argument('--out-grammar', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'motifs', 'enriched_cfg.json'))
    ap.add_argument('--rewrite-grammar', default=None, help='Path to grammar used for rewrite; separate from --out-grammar')
    ap.add_argument('--max-linear-len', type=int, default=6)
    ap.add_argument('--step-limit', type=int, default=500)
    ap.add_argument('--min-support', type=int, default=2)
    ap.add_argument('--top-k', type=int, default=10, help='limit number of motifs added')
    ap.add_argument('--motif-weight', type=float, default=0.1)
    ap.add_argument('--min-len', type=int, default=3)
    ap.add_argument('--save-motifs', default=None, help='Optional path to save discovered motifs and supports as JSON')
    ap.add_argument('--save-motifized-winners', default=None, help='Optional path to save winners rewritten with motifs as JSON')
    args = ap.parse_args()

    # Auto-derive rewrite grammar from winners if not provided: <winners_base_without__best>.json
    # Prefer motifs/<base>.json; fallback to winners dir.
    rewrite_path = args.rewrite_grammar
    if not rewrite_path:
        w_dir = os.path.dirname(args.winners)
        w_base = os.path.splitext(os.path.basename(args.winners))[0]
        base = w_base[:-5] if w_base.endswith('_best') else w_base
        root = os.path.dirname(os.path.dirname(__file__))
        candidate_motifs = os.path.join(root, 'motifs', f'{base}.json')
        selected = candidate_motifs if os.path.exists(candidate_motifs) else None
        if selected:
            # Validate that selected file looks like a grammar (has at least one nonterminal with rules)
            try:
                g = json.load(open(selected, 'r'))
                looks_like_grammar = any(isinstance(v, dict) and 'rules' in v for v in g.values()) if isinstance(g, dict) else False
            except Exception:
                looks_like_grammar = False
            if looks_like_grammar:
                rewrite_snapshot = os.path.join(root, 'motifs', f'{base}_rewrite_cfg.json')
                try:
                    with open(rewrite_snapshot, 'w') as wf:
                        json.dump(g, wf, indent=2)
                    rewrite_path = rewrite_snapshot
                    print(f"[main] auto rewrite: base='{base}'; selected='{selected}'; snapshot='{rewrite_snapshot}'")
                except Exception as e:
                    print(f"[main] auto rewrite failed to snapshot from {selected}: {e}")
                    rewrite_path = selected
            else:
                print(f"[main] auto rewrite skipped; candidate does not look like a grammar: {selected}")
        else:
            print(f"[main] no auto rewrite candidate found for base='{base}'")

    rep_by_sig, counter = mine_motifs_from_winners(
        winners_path=args.winners,
        max_linear_len=args.max_linear_len,
        step_limit=args.step_limit,
        min_support=args.min_support,
        min_len=args.min_len,
        rewrite_grammar_path=rewrite_path,
    )


    # Sort motif reps by (support desc, length asc) and take top-k
    reps = list(rep_by_sig.values())
    reps.sort(key=lambda s: (-counter[s], len(s)))
    if args.top_k > 0:
        reps = reps[:args.top_k]

    print(f"Discovered {len(reps)} motifs (after support filter). Top examples:")
    for s in reps[:10]:
        print(f"  [{counter[s]}x] {s}")

    expand_grammar(args.base_grammar, args.out_grammar, reps, motif_weight=args.motif_weight,
                   rewrite_grammar_path=rewrite_path, min_final_symbols=args.min_len)
    print(f"Enriched grammar written to {args.out_grammar}")

    # After expand, optionally save motifized winners using the enriched grammar
    if args.save_motifized_winners:
        try:
            with open(args.winners, 'r') as wf:
                winners_doc = json.load(wf)
            motif_map = build_motif_key_map(json.load(open(args.out_grammar)))
            rewritten: Dict[str, Any] = {}
            for tn, obj in winners_doc.items():
                sols = obj.get('solutions') or []
                rewritten[tn] = {
                    'solutions_symbolic': [
                        motifize_body(
                            strip_noops(s[1:-1] if (len(s) >= 2 and s[0]==',' and s[-1]=='.') else s),
                            motif_map
                        ) for s in sols
                    ],
                    'solutions': sols,
                }
            with open(args.save_motifized_winners, 'w') as rw:
                json.dump(rewritten, rw, indent=2)
            print(f"[main] saved motifized winners to {args.save_motifized_winners}")
        except Exception as e:
            print(f"[main] failed to save motifized winners: {e}")


if __name__ == '__main__':
    main()
