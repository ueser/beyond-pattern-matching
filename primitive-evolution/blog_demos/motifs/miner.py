import re
import json
from collections import Counter

LOOP = re.compile(r'\[[^\[\]]+\]')

def mine(programs: list[str], min_support=2, max_len=16):
    """Mine loop motifs from successful programs."""
    c = Counter()
    for p in programs:
        for m in LOOP.findall(p):
            if 2 <= len(m) <= max_len: 
                c[m] += 1
    
    motifs = [{"name": f"M{i+1}", "pattern": pat} 
              for i, (pat, n) in enumerate(c.items()) if n >= min_support]
    return {"motifs": motifs}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_support", type=int, default=2)
    args = ap.parse_args()
    
    best = json.load(open(args.best))
    progs = [best[t]["code"] for t in best]
    lib = mine(progs, args.min_support)
    json.dump(lib, open(args.out, "w"), indent=2)