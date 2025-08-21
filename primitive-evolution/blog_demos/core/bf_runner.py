from typing import Optional
import os
from brainfuck import BrainfuckInterpreter

DEFAULT_STEP_LIMIT = int(os.environ.get("BF_STEP_LIMIT", "5000"))


def run_once(code: str, x: int, step_limit: int = DEFAULT_STEP_LIMIT) -> Optional[int]:
    """Execute BF code with single byte input, return single byte output.
    Resets memory each time (stateless).
    """
    itp = BrainfuckInterpreter()
    try:
        s = itp.run(code, '\x00' if x == 0 else chr(x), max_steps=step_limit)
        return ord(s[0]) if s else None
    except Exception:
        return None


def run_step_persistent(itp: BrainfuckInterpreter, code: str, x: int, step_limit: int = DEFAULT_STEP_LIMIT) -> Optional[int]:
    """Run one step with persistent interpreter state.
    Feeds x as single input, executes until first output or step limit, returns the byte.
    """
    try:
        s = itp.run_step(code, '\x00' if x == 0 else chr(x), max_steps=step_limit, preserve_memory=True, output_limit=1)
        return ord(s[0]) if s else None
    except Exception:
        return None