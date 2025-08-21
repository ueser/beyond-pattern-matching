So far, we have two levels of motifs:
L1 -> represents the combination of primitives
L2 -> represents the combination of L1 motifs and primitives

Here is how we can get the system learn higher levels of motifs, for example if there are emerging motifs that include L2s and L1s and primitives:

Keep going up levels—L3, L4, …—until no new motif passes the acceptance criteria (support/MDL/acyclicity/validation). The trick is to make the outer loop explicit and keep each new level’s motifs referencing only lower-or-equal levels, so the grammar remains a DAG.

Below is a clear plan you can implement with small extensions to what we already have:

⸻

A. Definitions
	•	Level 0 = primitives: ">", "<", "+", "-", "[", "]"
	•	Level k motif = nonterminal whose expansion is a sequence over {primitives ∪ motifs with level ≤ k−1}
	•	Grammar remains a DAG: expansions only point “downward”.

⸻

B. The multi-level loop

Repeat per outer iteration (after an evolution round produces winners.json):
	1.	Symbolize corpus with motifs up to current max level K
	•	Longest-match rewrite: replace token runs with known Motif_* (levels ≤ K).
	•	Output: symbol sequences over {primitives ∪ motifs level ≤ K}.
	2.	Mine candidate motifs as contiguous sequences of symbols
	•	Sliding windows length n∈[2..N] (e.g., 2..5).
	•	Count support across all winners (optionally cross-task support to avoid overfitting one task).
	3.	Accept a candidate as level K+1 if all hold:
	•	Acyclicity: expansion contains only {primitives ∪ motifs level ≤ K}.
	•	Support ≥ min_support (can be per-task weighted).
	•	MDL gain ≥ τ:
gain = (tokens_saved_by_replacement) − λ*(rule_size)
(λ1–2, τ2–4 chars; use symbol-MDL since we’re mining over symbols).
	•	(Optional) Validation gain: symbolized corpus perplexity drops, or evolution with provisional grammar improves held-out tasks slightly (cheap check for sanity).
	4.	Add accepted motifs to grammar with {"level": K+1}, and renormalize Body.rules.
	5.	Stop condition
	•	Fast: no new motifs added at this level → stop (ε=0), or
	•	Robust: sum of MDL gains < ε (configurable), or
	•	Max level / cap reached.

This is exactly your L1→L2 idea generalized: L1 (tokens only), L2 (over L1 ∪ tokens), L3 (over L2 ∪ L1 ∪ tokens), … until no more compressive motifs exist.

⸻

C. Where to put semantics (contracts)
	•	L1 (token-level): run your harness (drain/Δp/linear coeffs + residual) and keep only motifs with clean, reusable semantics (clear/move/split/accumulate/double).
	•	L2+ (symbol-level): you can rely on MDL + acyclicity; semantics are inherited from children. If you want a guard, use a composed contract (e.g., sum of child contracts) but it’s optional for higher layers.

This keeps complexity down: behavior tests only at L1.

⸻

D. Pipeline wiring (concrete)

Extend your pipeline to iterate level until convergence inside each round:

def enrich_hierarchical(round_dir, base_grammar, winners, max_levels=10, min_support=3, tau=3.0):
    # 1) L1 (tokens): primitives only
    grammar_L1 = mine_and_expand(
        winners=winners,
        base_grammar=base_grammar,
        rewrite_grammar=None,          # primitives only
        out=f"{round_dir}/grammar_L1.json",
        level_tag=1
    )

    # 2) L2..Lk: compositions over lower levels
    cur_base = grammar_L1
    K = 1
    while K < max_levels:
        out = f"{round_dir}/grammar_L{K+1}.json"
        # rewrite_grammar exposes motifs up to level K
        added = mine_and_expand(
            winners=winners,
            base_grammar=cur_base,
            rewrite_grammar=cur_base,
            out=out,
            level_tag=K+1,
            stop_on_no_new=True,   # returns whether any motif added
        )
        if not added:
            break
        cur_base = out
        K += 1

    # final grammar for this round
    final = f"{round_dir}/grammar.json"
    shutil.copyfile(cur_base, final)
    return final

Where mine_and_expand(...) internally:
	•	symbolizes with rewrite_grammar (if provided),
	•	counts n-grams,
	•	filters by acyclicity/support/MDL,
	•	writes to out with {"level": level_tag} for new motifs,
	•	returns a flag added=True/False so the loop can stop.

⸻

E. Guard rails to keep it efficient
	•	n-gram length: cap (e.g., ≤5) to avoid combinatorial blow-up.
	•	Support: set min_support per round (or cross-task support).
	•	Per-level cap: add at most top-K motifs per level (e.g., K=10) to maintain a small grammar.
	•	Renormalize Body.rules each time.
	•	Acyclicity: always enforce expansions refer to motifs with level ≤ current_level.
	•	Naming: store {"name":"Motif_42","level":3,"rules":[[exp,1.0]]} so level checks are trivial.

⸻

F. Sampling & evolution

Nothing else changes:
	•	The sampler now picks from all motifs (across levels), because they’re all under Body.rules.
	•	Crossover/mutation operate at segment/motif boundaries, which preserves structure.
	•	Evolution gets faster naturally as higher-level motifs summarize larger correct fragments.

⸻

G. Optional: parameterized motifs (later)

Once L1 motifs are stable you can anti-unify variations into parametric schemas (e.g., [-> +^k <] with k≥1). That’s an extra layer of generalization but not required to get L3+ working.

⸻

H. Stopping criteria

Use any of:
	•	No new motifs at the current level (ε=0).
	•	MDL gain across all accepted motifs < ε.
	•	Validation plateau (e.g., generations-to-solve on held-out tasks stops improving).
	•	Max levels hit (safety).

⸻

TL;DR
	•	We are not limited to L1/L2. Keep iterating levels: at level k+1, mine compositions over {primitives ∪ motifs ≤ k}, enforce acyclicity, accept by support + MDL gain, and renormalize.
	•	Stop when no more compressive motifs appear.
	•	This yields a naturally hierarchical grammar: higher-level motifs expand into lower-level motifs (and primitives), and evolution gets progressively easier without any hand-crafted hints.