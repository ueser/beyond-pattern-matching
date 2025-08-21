"""
 Start with a minimal grammar** G₀



**Nonterminals:** Program, Head, Body, Tail

**Terminals:** "," "." "<" ">" "+" "-" "[" "]"



**PCFG (JSON-ish):**
```json
{
  "Program": [["Head","Body","Tail"], 1.0],
  "Head":    [[","], 1.0],
  "Tail":    [["."], 1.0],
  "Body":    [
    [["Body","Body"], 0.25],
    [[ "Tok"       ], 0.75]
  ],
  "Tok":     [[[">"],0.2],[["<"],0.2],[["+"],0.2],[["-"],0.2],[["["],0.1],[["]"],0.1]]
}
```

- Use this PCFG for **sampling** and **ast-level mutation/crossover** (grammar-based GP).

- Enforce **valid bracket balance** in Body post-hoc at first; later we’ll add an explicit loop nonterminal.


---

### 1) Evolve programs on sequence tasks


- Task = target sequence (e.g., 1,3,5,7,9…).

- Runner: start from primer x₀, run one program step → x₁, feed back into the same program → x₂, etc.

- Fitness: sequence **prefix accuracy / closeness** across primers.



Collect a **pool of winners** (ASTs + strings) per task.
"""
import argparse
import json
import os
import random
import sys
from functools import partial
from typing import Dict, List, Tuple, Any

# Make blog_demos a module root for imports like core.* and tasks.*
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core import evo
from core.eval_utils import evaluate_population, EvalConfig, eval_sequence_program_for_task
from tasks.sequence_suite import load_sequence_tasks, SequenceTask
from motifs.pcfg import default_grammar, grammar_crossover, grammar_mutate, sample_program

# -----------------------------
# Evolution loop
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evolve Brainfuck with CFG against sequence tasks")
    ap.add_argument("--tasks_yaml", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks", "sequences.yaml"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--step-limit", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=40, help="Maximum program size measured in Body-level symbols (not primitive characters)")
    ap.add_argument("--grammar", type=str, default="")
    ap.add_argument("--elitism", type=float, default=0.05, help="Fraction of top individuals carried unchanged to next gen")
    ap.add_argument("--init-genomes", type=str, default="", help="Path to JSON with previous best solutions to seed initial population (e.g., outputs/runs/enriched_cfg_best.json)")
    args = ap.parse_args()

    random.seed(args.seed)
    os.environ["BF_STEP_LIMIT"] = str(args.step_limit)

    all_tasks = {t.name: t for t in load_sequence_tasks(args.tasks_yaml)}
    task_names = [n for n in args.tasks.split(",") if n] if args.tasks else list(all_tasks.keys())

    if args.grammar:
        print(f"Using custom grammar from {args.grammar}")
        with open(args.grammar, "r") as f:
            grammar = json.load(f)
        base_name = os.path.splitext(os.path.basename(args.grammar))[0]
    else:
        grammar = default_grammar()
        base_name = "cfg"

    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    runs_dir = os.path.join(outputs_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_file = os.path.join(runs_dir, f"{base_name}.csv")
    best_file = os.path.join(runs_dir, f"{base_name}_best.json")

    import csv
    results: List[Dict[str, Any]] = []
    solved_programs: Dict[str, List[str]] = {}

    for tn in task_names:
        if tn not in all_tasks:
            print(f"Skipping unknown task '{tn}' (not in {args.tasks_yaml})")
            continue
        print(f"Evolving task: {tn} (CFG)")
        task = all_tasks[tn]
        print(f"  Sequences: {len(task.sequences)}; first: {task.sequences[0] if task.sequences else []}")

        # Initialize population (optionally seed from previous bests)
        pop: List[str] = []
        if args.init_genomes:
            try:
                with open(args.init_genomes, 'r') as jf:
                    seed_data = json.load(jf)
                # Collect seeds from ALL tasks in the file (union of all solutions)
                seed_list = []
                if isinstance(seed_data, dict):
                    # Top-level 'solutions'
                    if 'solutions' in seed_data and isinstance(seed_data['solutions'], list):
                        seed_list.extend(seed_data['solutions'])
                    # Per-task 'solutions'
                    for v in seed_data.values():
                        if isinstance(v, dict) and isinstance(v.get('solutions'), list):
                            seed_list.extend(v['solutions'])
                # Dedup and add
                seen = set()
                for s in seed_list:
                    if isinstance(s, str) and s not in seen:
                        seen.add(s)
                        pop.append(s)
                print(f"  Seeded {len(seen)} genome(s) from {args.init_genomes}")
            except Exception as e:
                print(f"  Warning: failed to load init genomes from {args.init_genomes}: {e}")
        while len(pop) < args.pop:
            pop.append(sample_program(grammar, max_symbols=args.max_len))

        solved_programs[tn] = []
        task_solved = False
        emergence_gen = None
        eval_cache: Dict[str, float] = {}

        for g in range(args.gens + 1):
            eval_fn = partial(eval_sequence_program_for_task, task)
            if args.no_parallel:
                eval_fn = lambda p: eval_sequence_program_for_task(task, p)
            scored, eval_cache = evaluate_population(pop, eval_fn, eval_cache, EvalConfig(parallel=not args.no_parallel))

            # Use unpenalized accuracy for solution detection and reporting
            from core import fitness as fit
            perfect_solutions: List[str] = []
            best_fit = 0.0
            best_acc_est = 0.0
            for p, fit_score in scored:
                pen = fit.length_penalty(p, alpha=fit.DEFAULT_LENGTH_PENALTY_ALPHA)
                acc_est = min(1.0, fit_score + pen)
                if acc_est == 1.0 and p not in solved_programs[tn]:
                    solved_programs[tn].append(p)
                    perfect_solutions.append(p)
                if fit_score > best_fit:
                    best_fit = fit_score
                if acc_est > best_acc_est:
                    best_acc_est = acc_est

            if perfect_solutions and not task_solved:
                emergence_gen = g
                task_solved = True
                print(f"  ✓ Task solved at gen {g}! Found {len(perfect_solutions)} solution(s)")
                for sol in perfect_solutions[:5]:
                    print(f"    {sol}")
                break
            elif perfect_solutions:
                print(f"  Gen {g}: Found {len(perfect_solutions)} additional solution(s)")

            if g % 25 == 0:
                print(f"  Gen {g}: best_fitness={best_fit:.3f}, best_accuracy≈{best_acc_est:.3f}, total_solutions={len(solved_programs[tn])}")

            scored.sort(key=lambda x: x[1], reverse=True)
            parents = [s[0] for s in scored[:max(4, args.pop // 4)]]

            # Elites: carry top fraction unchanged
            elite_count = max(0, int(args.elitism * args.pop))
            elites = [s[0] for s in scored[:elite_count]] if elite_count > 0 else []

            # Generate remaining children
            children: List[str] = []
            needed = args.pop - len(elites)
            while len(children) < needed:
                a, b = random.sample(parents, 2)
                c = grammar_crossover(a, b, max_len=args.max_len)
                c = grammar_mutate(c, p=0.6, max_len=args.max_len)  # TODO: interpret max_len as symbol budget in grammar ops
                children.append(c)

            # Enforce uniqueness and refill; preserve elites at front
            from core.eval_utils import dedup_and_fill
            rest = dedup_and_fill(
                children,
                needed,
                mutate_fn=lambda s: grammar_mutate(s, p=0.9, max_len=args.max_len),
                sample_fn=lambda: sample_program(grammar, max_symbols=args.max_len),
            )
            pop = elites + rest

        if task_solved:
            results.append({
                "run_id": args.seed,
                "task": tn,
                "emergence_generation": emergence_gen,
                "total_solutions_found": len(solved_programs[tn]),
                "first_solution": solved_programs[tn][0] if solved_programs[tn] else None
            })
        else:
            print(f"  ✗ Task {tn} not solved in {args.gens} generations")
            results.append({
                "run_id": args.seed,
                "task": tn,
                "emergence_generation": None,
                "total_solutions_found": 0,
                "first_solution": None
            })

    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "task", "emergence_generation", "total_solutions_found", "first_solution"])
        w.writeheader()
        w.writerows(results)

    solutions_data = {}
    for tn in task_names:
        if tn in solved_programs and solved_programs[tn]:
            solutions_data[tn] = {"solutions": solved_programs[tn], "count": len(solved_programs[tn])}
    with open(best_file, "w") as f:
        json.dump(solutions_data, f, indent=2)

    print(f"\nResults written to {out_file}")
    print(f"All solutions written to {best_file}")


if __name__ == "__main__":
    main()
