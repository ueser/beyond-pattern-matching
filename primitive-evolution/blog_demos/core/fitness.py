def exact_score(task, code):
    """Exact match fitness: fraction of test cases producing correct output."""
    hits = 0
    for inp, exp in task.examples:
        y = task.run(code, inp)
        hits += int(y == exp)
    return hits / len(task.examples)

def close_score(task, code):
    """Close match fitness: 1 - normalized circular distance."""
    errs = []
    for inp, exp in task.examples:
        y = task.run(code, inp)
        if y is None:
            errs.append(1.0)
            continue
        d = abs((y - exp) % 256)
        d = min(d, 256 - d)
        errs.append(d / 127.5)
    return 1.0 - (sum(errs) / len(errs))

# New modular fitness for sequence tasks (recurrent evaluation with multiple sequences)
from typing import List, Optional

# Default strength of length penalty used by default fitness
DEFAULT_LENGTH_PENALTY_ALPHA: float = 0.05

# We import lazily to avoid circular imports

def _sequence_hits(task, code) -> Optional[float]:
    """Compute average exact-match fraction across all sequences of a SequenceTask.
    Returns float in [0,1]. If task has no sequences, returns 0.0.
    """
    try:
        # Import here to avoid circular imports at module load time
        from tasks.sequence_suite import SequenceTask
    except Exception:
        return 0.0

    if not hasattr(task, 'sequences'):
        return 0.0
    seqs = getattr(task, 'sequences', [])
    if not seqs:
        return 0.0

    seq_scores: List[float] = []
    for target in seqs:
        if not isinstance(target, (list, tuple)) or len(target) < 2:
            continue
        gen = task.run_generated_for(code, target)
        hits = 0
        for i in range(1, len(target)):
            y = gen[i] if i < len(gen) else None
            hits += int(y == (target[i] % 256))
        seq_scores.append(hits / (len(target) - 1))

    return sum(seq_scores) / len(seq_scores) if seq_scores else 0.0


def sequence_exact_score(task, code) -> float:
    """Default fitness for sequence tasks: average exact-match fraction.
    Delegated from eval_utils.eval_sequence_program_for_task.
    """
    return float(_sequence_hits(task, code) or 0.0)


def sequence_exact_with_length_penalty(task, code, alpha: float = 0.0) -> float:
    """Exact-match fitness minus alpha * normalized program length.
    Assumes code is a string; normalization uses a smooth penalty < alpha.
    """
    base = float(_sequence_hits(task, code) or 0.0)
    if not isinstance(code, str) or alpha <= 0:
        return base
    return max(0.0, base - length_penalty(code, alpha))


def length_penalty(code: str, alpha: float = DEFAULT_LENGTH_PENALTY_ALPHA) -> float:
    """Compute the length-based penalty term used in default fitness.
    Returns a value in [0, alpha). If code is not a string or alpha<=0, returns 0.0.
    """
    if not isinstance(code, str) or alpha <= 0.0:
        return 0.0
    return alpha * (len(code) / (len(code) + 32.0))
