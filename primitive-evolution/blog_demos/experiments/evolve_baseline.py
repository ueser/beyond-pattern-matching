import argparse
import csv
import json
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core import evo
from core.eval_utils import evaluate_population, EvalConfig
from tasks.sequence_suite import load_sequence_tasks, SequenceTask

# Standard output paths
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
RUNS_DIR = os.path.join(OUTPUTS_DIR, "runs") 
MOTIFS_DIR = os.path.join(OUTPUTS_DIR, "motifs")
COMPARE_DIR = os.path.join(OUTPUTS_DIR, "compare")

def ensure_output_dirs():
    """Create standard output directories if they don't exist."""
    for d in [OUTPUTS_DIR, RUNS_DIR, MOTIFS_DIR, COMPARE_DIR]:
        os.makedirs(d, exist_ok=True)

def eval_sequence_task(task: SequenceTask, code: str) -> float:
    """Evaluate program on a sequence task using recurrent generation across all sequences.
    Score = average exact fraction across all sequences for this task.
    """
    if not task.sequences:
        return 0.0
    seq_scores = []
    for target in task.sequences:
        if len(target) < 2:
            continue
        gen = task.run_generated_for(code, target)
        hits = sum(int((gen[i] if i < len(gen) else None) == (target[i] % 256)) for i in range(1, len(target)))
        seq_scores.append(hits / (len(target) - 1))
    return sum(seq_scores) / len(seq_scores) if seq_scores else 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_yaml", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks", "sequences.yaml"), help="Path to YAML file with sequence tasks")
    ap.add_argument("--tasks", default="", help="Optional comma-separated list of task names to run; if empty, run all from YAML")
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-parallel", action="store_true", help="Disable parallel evaluation")
    ap.add_argument("--step-limit", type=int, default=5000, help="Interpreter max steps per run")
    ap.add_argument("--elitism", type=float, default=0.05, help="Fraction of top individuals carried unchanged to next gen")
    args = ap.parse_args()

    ensure_output_dirs()
    random.seed(args.seed)

    # Standard output files
    out_file = os.path.join(RUNS_DIR, "baseline.csv")
    best_file = os.path.join(RUNS_DIR, "baseline_best.json")

    # Load tasks from YAML and optionally filter by names
    all_tasks = {t.name: t for t in load_sequence_tasks(args.tasks_yaml)}
    task_names = [n for n in args.tasks.split(",") if n] if args.tasks else list(all_tasks.keys())
    results = []  # Only successful solutions
    solved_programs = {}  # All programs that solve each task
    eval_cache: dict[str, float] = {}  # Cache of program -> accuracy

    for tn in task_names:
        if tn not in all_tasks:
            print(f"Skipping unknown task '{tn}' (not in {args.tasks_yaml})")
            continue
        print(f"Evolving task: {tn}")
        task = all_tasks[tn]
        print(f"  Sequence: {task.sequence}")
        # Reset evaluation cache per task to avoid cross-task contamination
        eval_cache = {}
        pop = [evo.random_program() for _ in range(args.pop)]
        solved_programs[tn] = []
        task_solved = False
        emergence_gen = None

        for g in range(args.gens + 1):
            perfect_solutions = []

            # Evaluate with caching and optional parallelism
            from functools import partial
            from core.eval_utils import eval_sequence_program_for_task
            eval_fn = eval_sequence_program_for_task if args.no_parallel else partial(eval_sequence_program_for_task, task)
            if args.no_parallel:
                eval_fn = lambda p: eval_sequence_program_for_task(task, p)

            scored, eval_cache = evaluate_population(
                programs=pop,
                eval_fn=eval_fn,
                cache=eval_cache,
                cfg=EvalConfig(parallel=not args.no_parallel)
            )

            # Collect all perfect solutions (100% accuracy)
            for p, accuracy in scored:
                if accuracy == 1.0:
                    if p not in solved_programs[tn]:
                        solved_programs[tn].append(p)
                        perfect_solutions.append(p)

            # Record emergence when first solution found
            if perfect_solutions and not task_solved:
                emergence_gen = g
                task_solved = True
                print(f"  ✓ Task solved at gen {g}! Found {len(perfect_solutions)} solution(s)")
                
                break
            elif perfect_solutions:
                print(f"  Gen {g}: Found {len(perfect_solutions)} additional solution(s)")
                for sol in perfect_solutions:
                    print(f"    {sol}")

            best_accuracy = max(s[1] for s in scored) if scored else 0.0
            if g % 25 == 0:
                print(f"  Gen {g}: best_accuracy={best_accuracy:.3f}, total_solutions={len(solved_programs[tn])}")

            # Simple tournament + variation
            scored.sort(key=lambda x: x[1], reverse=True)
            parents = [s[0] for s in scored[:max(4, args.pop // 4)]]

            # Elites: carry top fraction unchanged
            elite_count = max(0, int(args.elitism * args.pop))
            elites = [s[0] for s in scored[:elite_count]] if elite_count > 0 else []

            # Generate remaining children
            children = []
            needed = args.pop - len(elites)
            while len(children) < needed:
                a, b = random.sample(parents, 2)
                c = evo.crossover(a, b)
                c = evo.mutate(c, p=0.6)
                children.append(c)
            # Enforce uniqueness and refill via mutate/sample; preserve elites
            from core.eval_utils import dedup_and_fill
            rest = dedup_and_fill(
                children,
                needed,
                mutate_fn=lambda s: evo.mutate(s, p=0.9),
                sample_fn=lambda: evo.random_program(),
            )
            pop = elites + rest
        
        # Record result only if task was solved
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
    
    # Write results - only emergence data for solved tasks
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "task", "emergence_generation", "total_solutions_found", "first_solution"])
        w.writeheader()
        w.writerows(results)
    
    # Write all solutions that achieved 100% accuracy
    solutions_data = {}
    for tn in task_names:
        if solved_programs[tn]:  # Only include tasks with solutions
            solutions_data[tn] = {
                "solutions": solved_programs[tn],
                "count": len(solved_programs[tn])
            }
    
    with open(best_file, "w") as f:
        json.dump(solutions_data, f, indent=2)
    
    print(f"\nResults written to {out_file}")
    print(f"All solutions written to {best_file}")
    
    # Summary
    solved_tasks = [r for r in results if r["emergence_generation"] is not None]
    print(f"\nSUMMARY:")
    print(f"  Tasks attempted: {len(task_names)}")
    print(f"  Tasks solved: {len(solved_tasks)}")
    if solved_tasks:
        avg_emergence = sum(r["emergence_generation"] for r in solved_tasks) / len(solved_tasks)
        total_solutions = sum(r["total_solutions_found"] for r in solved_tasks)
        print(f"  Average emergence time: {avg_emergence:.1f} generations")
        print(f"  Total unique solutions found: {total_solutions}")