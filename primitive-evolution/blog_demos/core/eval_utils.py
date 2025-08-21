from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, List, Optional
from multiprocessing import Pool, cpu_count

# Shared utilities for evolution experiments: caching and parallel eval

@dataclass
class EvalConfig:
    parallel: bool = True
    processes: int = max(1, cpu_count() - 1)


def evaluate_population(
    programs: List[str],
    eval_fn: Callable[[str], float],
    cache: Dict[str, float] | None = None,
    cfg: EvalConfig = EvalConfig(),
) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """Evaluate a list of program strings with caching and optional parallelism.
    Returns (scored, updated_cache) where scored is a list of (program, score).
    Note: eval_fn must be a top-level function or functools.partial of one for spawn.
    """
    cache = cache or {}

    # Determine which need evaluation
    to_eval: List[str] = [p for p in programs if p not in cache]

    if to_eval:
        if cfg.parallel and len(to_eval) > 1 and cfg.processes > 1:
            with Pool(processes=cfg.processes) as pool:
                scores = pool.map(eval_fn, to_eval)
        else:
            scores = [eval_fn(p) for p in to_eval]
        for p, s in zip(to_eval, scores):
            cache[p] = s

    scored = [(p, cache[p]) for p in programs]
    return scored, cache


# Top-level eval function for sequence tasks to be picklable by multiprocessing
# Avoid importing SequenceTask at module import to prevent circular deps.

def eval_sequence_program_for_task(task, program: str) -> float:
    """Evaluate a program against a sequence task using the modular fitness API.
    Default uses length-penalized exact score.
    """
    from core import fitness
    return fitness.sequence_exact_with_length_penalty(task, program, alpha=fitness.DEFAULT_LENGTH_PENALTY_ALPHA)


def dedup_and_fill(pop: List[str], target_size: int, mutate_fn, sample_fn) -> List[str]:
    """Deduplicate a population (preserving order) and refill to target_size.
    - mutate_fn: function(str) -> str used to generate variants from existing genomes
    - sample_fn: function() -> str used to sample fresh individuals if needed
    """
    seen = set()
    uniq: List[str] = []
    for p in pop:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    # Refill
    i = 0
    while len(uniq) < target_size and uniq:
        candidate = mutate_fn(uniq[i % len(uniq)])
        if candidate not in seen:
            seen.add(candidate)
            uniq.append(candidate)
        i += 1
        if i > target_size * 4:
            break  # safety
    while len(uniq) < target_size:
        candidate = sample_fn()
        if candidate not in seen:
            seen.add(candidate)
            uniq.append(candidate)
    return uniq[:target_size]

