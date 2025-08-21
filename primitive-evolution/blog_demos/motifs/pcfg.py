import random
import json
from typing import Dict, Any, List, Tuple

from core import evo

# -----------------------------
# Grammar (PCFG) utilities
# -----------------------------
TOKENS = [">", "<", "+", "-", "[", "]"]


def default_grammar() -> Dict[str, Any]:
    return {
        "Program": {"rules": [(["Head", "Body", "Tail"], 1.0)]},
        "Head":    {"rules": [([","], 1.0)]},
        "Tail":    {"rules": [(["."], 1.0)]},

        "Body":    {"rules": [(["Body", "Body"], 0.20), (["Loop"], 0.50), (["Atom"], 0.30)]},

        "Loop":    {"rules": [(["[", "LoopBody", "]"], 1.0)]},

        # recursive: allows any-length loop body
        "LoopBody":{"rules": [(["Atom"], 0.40), (["Atom","LoopBody"], 0.35), (["LoopBody","Atom"], 0.25)]},

        "Atom":    {"rules": [(["Shift"], 0.40), (["Inc"], 0.25), (["Dec"], 0.25), (["Loop"], 0.10)]},

        "Shift":   {"rules": [([">"], 0.5), (["<"], 0.5)]},
        "Inc":     {"rules": [(["+"], 1.0)]},
        "Dec":     {"rules": [(["-"], 1.0)]}
    }


def _weighted_choice(rules: List[Tuple[List[str], float]]) -> List[str]:
    r = random.random() * sum(w for _, w in rules)
    s = 0.0
    for prod, w in rules:
        s += w
        if r <= s:
            return prod
    return rules[-1][0]


def sample_from_pcfg(grammar: Dict[str, Any], symbol: str, max_body_symbols: int = 40, depth: int = 0, _used_symbols: List[int] = None) -> List[str]:
    """Sample expansion tokens from a PCFG with a budget measured in Body-level symbols.
    A Body-level symbol is one of: Motif_*, Loop, Atom, or Tok. This allows programs to
    grow in sophistication via motifs without being capped by primitive character count.
    """
    if _used_symbols is None:
        _used_symbols = [0]
    if symbol not in grammar:
        return [symbol]
    if symbol == "Body":
        if depth > 60:
            return []
        prod = _weighted_choice(grammar[symbol]["rules"])
        out: List[str] = []
        for s in prod:
            # Stop if symbol budget exhausted
            if _used_symbols[0] >= max_body_symbols:
                break
            # Count a unit when emitting a single Body-level unit
            body_unit = False
            if isinstance(s, str):
                if s.startswith('Motif_') or s in ("Loop", "Atom", "Tok"):
                    body_unit = True
            if body_unit:
                _used_symbols[0] += 1
            out.extend(sample_from_pcfg(grammar, s, max_body_symbols, depth + 1, _used_symbols))
        return out
    prod = _weighted_choice(grammar[symbol]["rules"])
    out: List[str] = []
    for s in prod:
        out.extend(sample_from_pcfg(grammar, s, max_body_symbols, depth + 1, _used_symbols))
    return out


def sample_program(grammar: Dict[str, Any], max_symbols: int = 40, tries: int = 50) -> str:
    for _ in range(tries):
        toks = sample_from_pcfg(grammar, "Program", max_body_symbols=max_symbols)
        s = "".join(toks)
        if not (s.startswith(",") and s.endswith(".")):
            s = "," + s + "."
        # No primitive-length truncation; rely on symbol budget to control growth
        if evo.is_balanced(s):
            return s
    # Fallback: bounded random program by primitive length (compat)
    return evo.random_program(max_len=max_symbols)




# -----------------------------
# Grammar-aware mutation/crossover
# -----------------------------

def grammar_mutate(prog: str, p: float = 0.4, max_len: int = 40) -> str:
    if random.random() > p:
        return prog
    head, body, tail = prog[0], prog[1:-1], prog[-1]
    body_list = list(body)
    op = random.random()
    if op < 0.33 and body_list:
        i = random.randrange(len(body_list))
        body_list[i] = random.choice(TOKENS)
    elif op < 0.66 and len(body_list) < max_len - 2:
        i = random.randrange(len(body_list) + 1)
        body_list.insert(i, random.choice(TOKENS))
    else:
        if body_list:
            i = random.randrange(len(body_list))
            body_list.pop(i)
    child = head + "".join(body_list) + tail
    return child if evo.is_balanced(child) else prog

def _segments(body: str) -> list[str]:
    """Split into balanced loop segments and linear runs."""
    segs = []
    i, n = 0, len(body)
    while i < n:
        if body[i] == '[':
            # find matching ] with nesting
            depth, j = 1, i + 1
            while j < n and depth:
                if body[j] == '[': depth += 1
                elif body[j] == ']': depth -= 1
                j += 1
            # if unmatched, treat as linear (fallback)
            if depth != 0:
                segs.append(body[i:j])
                i = j
            else:
                segs.append(body[i:j])  # include closing ]
                i = j
        else:
            # consume linear run
            j = i
            while j < n and body[j] != '[':
                j += 1
            segs.append(body[i:j])
            i = j
    # drop empties
    return [s for s in segs if s]

def _is_loop(seg: str) -> bool:
    return len(seg) >= 2 and seg[0] == '[' and seg[-1] == ']'

def _join_with_budget(segs: list[str], budget: int) -> str:
    """Concatenate segments under a token budget (don’t cut through loops)."""
    out, used = [], 0
    for s in segs:
        if used + len(s) <= budget:
            out.append(s); used += len(s)
        else:
            # only slice linear tails; drop loops that overflow
            if not _is_loop(s):
                out.append(s[: max(0, budget - used)])
            break
    return ''.join(out)

def grammar_crossover(a: str, b: str, max_len: int = 40) -> str:
    # keep head/tail fixed
    A, B = a[1:-1], b[1:-1]
    segA, segB = _segments(A), _segments(B)
    if not segA or not segB:
        return evo.mutate(a, 1.0, max_len)

    # pick cut points on segment boundaries
    i = random.randrange(len(segA) + 1)
    # try to pick a homologous cut (loop/linear)
    want_loop = (i > 0 and _is_loop(segA[i-1])) or (i < len(segA) and i < len(segA) and _is_loop(segA[i])) 
    # candidate cut positions in B that match type
    idxB_options = [k for k in range(len(segB) + 1)
                    if k > 0 and _is_loop(segB[k-1]) == want_loop] or list(range(len(segB)+1))
    j = random.choice(idxB_options)

    # assemble child and enforce budget (minus head/tail)
    body_budget = max_len - 2
    child_body = _join_with_budget(segA[:i] + segB[j:], body_budget)
    child = ',' + child_body + '.'

    # redundancy filter (optional but recommended)
    def _redundant(s: str) -> bool:
        body = s[1:-1]
        return any(x in body for x in ("+-", "-+", "><", "<>"))

    if evo.is_balanced(child) and not _redundant(child):
        return child

    # fallback: try a few alternate cuts
    for _ in range(8):
        i = random.randrange(len(segA) + 1)
        j = random.randrange(len(segB) + 1)
        child_body = _join_with_budget(segA[:i] + segB[j:], body_budget)
        child = ',' + child_body + '.'
        if evo.is_balanced(child) and not _redundant(child):
            return child

    # last resort: mutate a safe parent
    return evo.mutate(random.choice([a, b]), 1.0, max_len)