from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import json
import ast
from brainfuck import BrainfuckInterpreter

# Make blog_demos a module root for imports like core.bf_runner
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import yaml  # Optional; we'll fall back to JSON or a simple parser if unavailable
except Exception:
    yaml = None  # type: ignore

from core.bf_runner import run_step_persistent


@dataclass
class SequenceTask:
    name: str
    sequences: List[List[int]]

    def run_generated_for(self, code: str, seq: List[int]) -> List[Optional[int]]:
        """Generate a sequence using recurrent evaluation with persistent memory for a given target seq."""
        if not seq:
            return []
        gen: List[Optional[int]] = [seq[0]]
        x = int(seq[0]) % 256
        itp = BrainfuckInterpreter()
        for _ in range(1, len(seq)):
            y = run_step_persistent(itp, code, x)
            gen.append(None if y is None else (int(y) % 256))
            if y is not None:
                x = int(y) % 256
        return gen

    def run_generated(self, code: str) -> List[Optional[int]]:
        """Back-compat: run on first sequence if present."""
        return self.run_generated_for(code, self.sequences[0] if self.sequences else [])


def _coerce_sequence(seq: Any) -> List[int]:
    if not isinstance(seq, (list, tuple)):
        raise ValueError("sequence must be a list of integers")
    out: List[int] = []
    for v in seq:
        if not isinstance(v, int):
            raise ValueError("sequence elements must be integers")
        out.append(v % 256)
    return out


def _parse_simple_mapping(text: str) -> Dict[str, Any]:
    """Very small YAML-like parser for lines: name: [1,2,3] or name: [[...],[...]]"""
    mapping: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            continue
        name, rest = line.split(':', 1)
        name = name.strip()
        rest = rest.strip()
        try:
            seqs = ast.literal_eval(rest)
        except Exception:
            continue
        mapping[name] = seqs
    return mapping


def _coerce_sequences(val: Any) -> List[List[int]]:
    """Accept either a single sequence (list[int]) or a list of sequences."""
    if isinstance(val, (list, tuple)) and val and all(isinstance(x, int) for x in val):
        return [_coerce_sequence(val)]
    if isinstance(val, (list, tuple)) and all(isinstance(x, (list, tuple)) for x in val):
        return [_coerce_sequence(x) for x in val]
    if isinstance(val, (list, tuple)) and len(val) == 0:
        return []
    raise ValueError("expected a sequence (list[int]) or list of sequences (list[list[int]])")



def load_sequence_tasks(path: str) -> List[SequenceTask]:
    """Load sequence tasks from a YAML file.
    Supported formats:
      1) { tasks: [ { name, sequence } or { name, sequences }, ... ] }
      2) Mapping of name -> sequence or sequences list, e.g.: { my_task: [1,2,3] } or { my_task: [[1,2,3],[2,4,6]] }
      3) A list of { name, sequence } or { name, sequences } objects
    """
    with open(path, "r") as f:
        text = f.read()

    data: Any = None
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _parse_simple_mapping(text)

    if data is None:
        raise RuntimeError(
            "Failed to parse tasks file. Install PyYAML with 'poetry add pyyaml' or provide JSON/mapping format."
        )

    items: List[Dict[str, Any]]
    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            items = data["tasks"]
        else:
            # mapping of name -> sequence(s)
            items = [{"name": k, ("sequences" if isinstance(v, list) and v and isinstance(v[0], (list, tuple)) else "sequence"): v} for k, v in data.items()]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unsupported tasks structure; expected dict or list")

    tasks: List[SequenceTask] = []
    for obj in items:
        name = obj.get("name")
        if not name:
            raise ValueError("Each task must have 'name'")
        if "sequences" in obj:
            sequences = _coerce_sequences(obj["sequences"])
        elif "sequence" in obj:
            sequences = _coerce_sequences(obj["sequence"])
        else:
            raise ValueError("Task must have 'sequence' or 'sequences'")
        tasks.append(SequenceTask(name=name, sequences=sequences))
    return tasks

