#!/usr/bin/env python3
"""
Blog Demo 2: Experimental Setup - Tasks vs Macros

This script demonstrates the experimental setup for discovering emergent macros
through evolutionary pressure. The key insight is the separation between:

1. EXTERNAL TASKS: Problems we pose to the system (increment, double, sum)
2. INTERNAL MACROS: Code patterns that emerge as reusable building blocks

We don't tell evolution "discover [->+<]" - instead we give it tasks like
doubling where [->+<] happens to be an efficient subroutine. The macro
emerges naturally as a solution to multiple different external pressures.
"""

import argparse

import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from brainfuck import BrainfuckInterpreter

# Import our motif detection system
import sys
sys.path.append('.')
from motif_evolution import MotifParser, MotifLibrary

# =========================
# Staged fitness (inline)
# =========================
from dataclasses import dataclass
import numpy as np

@dataclass
class FitnessVector:
    """Vector of metrics used for staged/lexicase selection."""
    exact_double: float      # [0..1]
    close_double: float      # [0..1]
    io_contract: float       # 0/1 (or soft [0..1])
    delta_constancy: float   # [0..1]
    delta_slope2: float      # [0..1]
    work_increasing: float   # [0..1]
    loop_effective: float    # [0..1] heuristic for a draining/copying loop

class StagedFitnessEvaluator:
    """
    Computes a vector of objectives:
      - Primary (task): exactness and closeness for doubling
      - Auxiliary (behavioural): IO contract, Δ-constancy, Δ≈2, work increasing
    """
    def __init__(self, probe=None, validate=None):
        # Use unit-spaced probes for finite differences
        self.probe = probe or [2, 3, 4, 5, 6]
        self.validate = validate or [2, 3, 5, 7, 11, 13, 17, 19]

    def evaluate(self, program: str, run_with_meta) -> FitnessVector:
        outs, steps, reads, writes, timeouts = [], [], [], [], 0
        for x in self.probe:
            meta = run_with_meta(program, x)  # dict: {'out':int|None,'steps':int,'reads':int,'writes':int,'timeout':bool}
            y = meta['out'] if meta['out'] is not None else 0
            outs.append(y)
            steps.append(meta.get('steps', 0))
            reads.append(meta.get('reads', 0))
            writes.append(meta.get('writes', 0))
            timeouts += int(meta.get('timeout', False))

        # Stage 0: IO contract (exactly one read and one write across tests)
        io_ok = (all(r == 1 for r in reads) and all(w == 1 for w in writes))
        io_contract = 1.0 if io_ok else 0.0

        # Primary task metrics on probes
        exact_hits = [(outs[i] == (2 * self.probe[i]) % 256) for i in range(len(self.probe))]
        exact = float(np.mean(exact_hits)) if len(exact_hits) else 0.0
        # Circular absolute error helper
        def circ_abs(a, M=256):
            d = abs(a) % M
            return min(d, M - d)
        close = 1.0 - float(np.mean([circ_abs(outs[i] - ((2 * self.probe[i]) % 256)) / 127.5 for i in range(len(self.probe))])) if len(self.probe) else 0.0

        # Loop effectiveness heuristic
        def loop_effectiveness(prog: str) -> float:
            # Heuristic: at least one balanced loop that (a) contains '-' and both '>' and '<'
            # and (b) is not empty '[]'. Multiple such loops increase score slightly (cap at 1.0).
            score = 0.0
            body = prog[1:-1] if len(prog) >= 2 else prog
            stack = []
            for i, ch in enumerate(body):
                if ch == '[':
                    stack.append(i)
                elif ch == ']' and stack:
                    start = stack.pop()
                    seg = body[start+1:i]
                    if seg and ('-' in seg) and ('>' in seg) and ('<' in seg):
                        score += 0.5
            return 1.0 if score >= 1.0 else score
        loop_eff = loop_effectiveness(program) if is_balanced(program) else 0.0

        # Auxiliary: Δ metrics with unit-spaced probes
        if len(outs) >= 2:
            deltas = [ (outs[i+1] - outs[i]) % 256 for i in range(len(outs) - 1) ]
            # Constancy: small mean absolute successive difference -> closer to 1.0
            if len(deltas) >= 2:
                mean_abs_succ_diff = float(np.mean(np.abs(np.diff(deltas))))
            else:
                mean_abs_succ_diff = 0.0
            # Normalize by a small scale since values are in [0,255]; choose 4.0 as a crisp target
            delta_constancy = max(0.0, min(1.0, 1.0 - (mean_abs_succ_diff / 4.0)))
            # Slope target: mean Δ should be 2 for doubling on unit steps
            mean_delta = float(np.mean(deltas))
            delta_slope2 = max(0.0, min(1.0, 1.0 - (abs(mean_delta - 2.0) / 4.0)))
        else:
            delta_constancy = 0.0
            delta_slope2 = 0.0

        # Slope hard gate for IO gating in selection
        slope_gate = 1.0 if delta_slope2 >= 0.95 else 0.0

        # Auxiliary: "work" proxy (correlation of steps with input)
        if len(set(steps)) > 1 and np.std(steps) > 0:
            x = np.array(self.probe, dtype=float)
            s = np.array(steps, dtype=float)
            corr = float(np.corrcoef(x, s)[0, 1]) if np.std(x) > 0 else 0.0
            work_increasing = max(0.0, corr)
        else:
            work_increasing = 0.0

        return FitnessVector(
            exact_double=exact,
            close_double=close,
            io_contract=io_contract * slope_gate,
            delta_constancy=delta_constancy,
            delta_slope2=delta_slope2,
            work_increasing=work_increasing,
            loop_effective=loop_eff
        )

def run_with_meta(program: str, x: int) -> dict:
    """
    Execute a BF program and return a small meta-dict for staged fitness.
    We try to pull interpreter internals if available; otherwise we fall back
    to safe approximations so the demo runs without modifying the interpreter.
    """
    bf = BrainfuckInterpreter()
    try:
        inp = '\x00' if x == 0 else chr(x)
    except ValueError:
        inp = '\x00'
    out = None
    timeout = False
    steps = 0

    try:
        # If your interpreter supports step limits / instrumentation, you can extend here.
        result = bf.run(program, inp)
        out = ord(result[0]) if result and len(result) > 0 else None
        # Optional introspection
        steps = getattr(bf, 'step_count', 0)
        timeout = bool(getattr(bf, 'hit_step_limit', False))
        reads = getattr(bf, 'input_reads', None)
        writes = getattr(bf, 'output_writes', None)
    except Exception:
        timeout = True
        reads = None
        writes = None

    # Fallbacks if interpreter doesn't expose counts
    if reads is None:
        reads = program.count(',')
    if writes is None:
        writes = program.count('.')
    if steps == 0:
        # crude proxy so work_increasing has *some* signal
        steps = len(program) * 10

    return {'out': out, 'steps': steps, 'reads': reads, 'writes': writes, 'timeout': timeout}

# =========================
# Staged / lexicase selection helper
# =========================
def select_parent_lexicase(population, fitness_vecs):
    """
    Stage 0: gate by IO contract
    Stage 1: lexicase over auxiliary behaviours (loop_presence, delta_constancy, delta_slope2, work_increasing)
    Stage 2: tie-break by primary task (exact_double, close_double)
    Returns the selected individual's index.
    """
    import random
    # Pair up
    pairs = list(zip(range(len(population)), fitness_vecs))
    # Gate by IO
    gated = [p for p in pairs if p[1].io_contract >= 0.999]
    if not gated:
        gated = pairs  # fallback

    # Lexicase over auxiliaries
    cases = ['delta_slope2', 'loop_effective', 'delta_constancy', 'work_increasing']
    random.shuffle(cases)
    candidates = gated
    for c in cases:
        if len(candidates) <= 1:
            break
        vals = np.array([getattr(fv, c) for (_, fv) in candidates])
        if len(vals) == 0:
            break
        
        # Use epsilon-lexicase: keep individuals within epsilon of the best
        max_val = np.max(vals)
        epsilon = 0.1  # Allow some tolerance
        candidates = [(i, fv) for (i, fv) in candidates if getattr(fv, c) >= max_val - epsilon]
        
        # Stop early if we have reasonable diversity
        if len(candidates) <= max(2, len(gated) // 5):
            break

    # Final tie-break by primary task
    if not candidates:
        # Fallback if all candidates filtered out
        candidates = gated
    
    candidates.sort(key=lambda kv: (kv[1].exact_double, kv[1].close_double), reverse=True)
    winner_idx = candidates[0][0]
    return winner_idx

def print_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subheader(title):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

@dataclass
class CurriculumTask:
    """Represents an external task we give to the evolutionary system."""
    name: str
    description: str
    test_cases: Dict[int, int]  # input -> expected_output
    mathematical_form: str
    
    def evaluate_solution(self, program: str) -> Tuple[float, bool]:
        """Evaluate how well a program solves this task."""
        interpreter = BrainfuckInterpreter()
        correct_count = 0
        
        for input_val, expected_output in self.test_cases.items():
            try:
                interpreter.__init__()  # Reset
                result = interpreter.run(program, chr(input_val))
                
                if result and len(result) > 0:
                    actual_output = ord(result[0])
                    if actual_output == expected_output:
                        correct_count += 1
                        
            except Exception:
                continue
        
        accuracy = correct_count / len(self.test_cases)
        is_perfect = accuracy == 1.0
        return accuracy * 100, is_perfect

def define_curriculum_tasks():
    """Define the external tasks that create evolutionary pressure."""
    print_header("Curriculum Tasks: External Evaluation Pressures")
    
    tasks = [
        CurriculumTask(
            name="identity",
            description="Return input unchanged",
            test_cases={0: 0, 1: 1, 2: 2, 3: 3, 5: 5},
            mathematical_form="f(x) = x"
        ),
        
        CurriculumTask(
            name="increment", 
            description="Add 1 to input",
            test_cases={0: 1, 1: 2, 2: 3, 3: 4, 5: 6},
            mathematical_form="f(x) = x + 1"
        ),
        
        CurriculumTask(
            name="clear",
            description="Always output 0 (clear cell)",
            test_cases={1: 0, 2: 0, 3: 0, 5: 0, 10: 0},
            mathematical_form="f(x) = 0"
        ),
        
        CurriculumTask(
            name="double",
            description="Multiply input by 2", 
            test_cases={1: 2, 2: 4, 3: 6, 4: 8, 5: 10},
            mathematical_form="f(x) = 2*x"
        ),
        
        CurriculumTask(
            name="add_three",
            description="Add 3 to input",
            test_cases={0: 3, 1: 4, 2: 5, 3: 6, 5: 8},
            mathematical_form="f(x) = x + 3"
        ),
        
        CurriculumTask(
            name="triple",
            description="Multiply input by 3",
            test_cases={1: 3, 2: 6, 3: 9, 4: 12, 5: 15},
            mathematical_form="f(x) = 3*x"
        )
    ]
    
    print("\nThese are EXTERNAL PRESSURES we place on evolution:")
    print("\nTask Name    | Mathematical Form | Description")
    print("-" * 55)
    
    for task in tasks:
        print(f"{task.name:12} | {task.mathematical_form:15} | {task.description}")
    
    print(f"\n💡 Key Point: We define tasks at the FUNCTION level")
    print("   - We specify what input→output mapping we want")  
    print("   - We don't specify HOW the program should work internally")
    print("   - Evolution must discover the mechanisms")
    
    return tasks

def analyze_macro_emergence():
    """Demonstrate how macros emerge from multiple tasks."""
    print_header("Macro Emergence: Internal Building Blocks")
    
    # Simulate some evolved solutions to different tasks
    evolved_solutions = [

    ]
    
    print("\nSimulated evolved solutions across different tasks:")
    print("\nTask     | Solution           | Description")
    print("-" * 55)
    
    for task, solution, desc in evolved_solutions:
        print(f"{task:8} | {solution:18} | {desc}")
    
    # Extract motifs using our motif detection system
    print_subheader("Motif Detection: Finding Recurring Patterns")
    
    parser = MotifParser()
    library = MotifLibrary()
    
    print("\nAnalyzing solutions for recurring motifs...")
    
    motif_occurrences = {}
    
    for task, solution, desc in evolved_solutions:
        motifs = parser.parse_program(solution)
        print(f"\n'{solution}' contains motifs:")
        
        for motif in motifs:
            if motif.effect:
                cluster = library._classify_motif(motif.effect)
                library.add_motif(motif.effect)
                
                # Track where each motif appears
                motif_key = f"{cluster}:{motif.pattern}"
                if motif_key not in motif_occurrences:
                    motif_occurrences[motif_key] = []
                motif_occurrences[motif_key].append(task)
                
                print(f"  - {motif.pattern} → {cluster} motif")
    
    # Show cross-task patterns
    print_subheader("Cross-Task Pattern Discovery")
    
    print("\nMotifs that appear across multiple tasks:")
    print("\nMotif Type | Pattern  | Tasks Using It")
    print("-" * 40)
    
    for motif_key, tasks in motif_occurrences.items():
        cluster, pattern = motif_key.split(':', 1)
        if len(set(tasks)) > 1:  # Appears in multiple different tasks
            unique_tasks = list(set(tasks))
            print(f"{cluster:10} | {pattern:8} | {', '.join(unique_tasks)}")
    
    print(f"\n💡 Key Observation:")
    print("   - TASKS are external pressures (double, triple, clear)")
    print("   - MACROS are internal patterns that emerge to solve multiple tasks")
    print("   - The same building blocks ([-], [->++<]) get reused!")
    
    print(f"\n🔬 This is TRUE EMERGENCE:")
    print("   - We never told evolution 'discover [->++<]'")
    print("   - We gave it tasks like doubling and tripling") 
    print("   - The [->X+<] pattern emerged as the efficient solution")
    print("   - Now it can be reused as a MACRO for new tasks!")

def demonstrate_evolutionary_advantage():
    """Show how discovered macros accelerate learning on new tasks."""
    print_header("Evolutionary Advantage of Discovered Macros")
    
    print("\nImagine evolution has discovered these macros:")
    
    discovered_macros = {
        'CLEAR': '[-]',
        'MOVE': '[->+<]', 
        'DOUBLE_MOVE': '[->++<]',
        'TRIPLE_MOVE': '[->+++<]',
        'SPLIT': '[->+>+<<]'
    }
    
    print("\nDiscovered Macro Library:")
    for name, pattern in discovered_macros.items():
        print(f"  {name:12} → {pattern}")
    
    print_subheader("Solving New Tasks Using Macros")
    
    # Show how macros can be composed for new tasks
    composite_solutions = [
        (
            "f(x) = 4*x (quadruple)",
            ", [->++++<] >.",
            "Could use MOVE pattern with ++++ inside"
        ),
        (
            "f(x) = x + 5", 
            ", +++++ .",
            "Simple linear addition (no macro needed)"
        ),
        (
            "copy value to two places",
            ", [->+>+<<] >.",
            "Uses SPLIT macro directly"
        ),
        (
            "f(x) = 2*x + 1",
            ", [->++<] > + .",
            "DOUBLE_MOVE macro + linear addition"
        )
    ]
    
    print("\nNew tasks that could benefit from discovered macros:")
    print("\nTask                    | Solution Strategy")
    print("-" * 60)
    
    for task, solution, strategy in composite_solutions:
        print(f"{task:22} | {strategy}")
    
    print(f"\n💡 Evolutionary Acceleration:")
    print("   - Without macros: Evolution explores random program space")
    print("   - With macros: Evolution composes known building blocks")
    print("   - This is BOOTSTRAPPING: Evolution learns to learn!")
    
    print(f"\n🧬 The System Becomes Self-Improving:")
    print("   - Stage 1: Random exploration discovers basic motifs") 
    print("   - Stage 2: Motifs get reused, accelerating complex task learning")
    print("   - Stage 3: Meta-motifs emerge (compositions of basic motifs)")
    print("   - Evolution transitions from pattern-matching to symbolic reasoning!")

def demonstrate_curriculum_design():
    """Show how curriculum design influences macro emergence."""
    print_header("Curriculum Design: Guiding Macro Discovery")
    
    print("\nStrategic curriculum progression:")
    
    curriculum_stages = [
        ("Stage 1: Foundation", [
            "identity (f(x)=x) → Forces basic I/O structure",
            "clear (f(x)=0) → Forces [-] macro to emerge", 
            "increment (f(x)=x+1) → Forces linear arithmetic"
        ]),
        
        ("Stage 2: Basic Motifs", [
            "double (f(x)=2*x) → Forces [->++<] loop macro",
            "add_three (f(x)=x+3) → Reinforces linear vs loop tradeoffs",
            "triple (f(x)=3*x) → Generalizes the [->X+<] pattern"
        ]),
        
        ("Stage 3: Composition", [
            "quadruple (f(x)=4*x) → Should reuse doubling macro twice",
            "copy (output x to two places) → Forces [->+>+<<] split macro",
            "sum_neighbors → Forces accumulation patterns"
        ])
    ]
    
    for stage_name, tasks in curriculum_stages:
        print(f"\n{stage_name}:")
        for task in tasks:
            print(f"  • {task}")
    
    print_subheader("Why This Progression Works")
    
    print(f"""
Key principles of curriculum design:

1. GRADUAL COMPLEXITY
   - Start with simple tasks that require basic motifs
   - Build complexity by combining simpler building blocks
   
2. OVERLAPPING PRESSURES  
   - Multiple tasks should benefit from the same macro
   - This creates selection pressure for reusable patterns
   
3. COMPOSITIONAL STRUCTURE
   - Later tasks should be solvable by combining earlier discoveries
   - Tests whether the system learned building blocks vs memorized solutions

4. AVOID CIRCULAR DEPENDENCY
   - Tasks are external requirements (what we want)
   - Macros are internal discoveries (how evolution solves it)
   - The mapping from tasks to macros emerges naturally!
""")


# =========================
# Minimal evolutionary loop (doubling, staged fitness)
# =========================

BF_TOKENS = "><+-[],"  # body tokens; ',' is allowed in body for generality but we gate IO via fitness
MAX_LEN = 40

def random_program(max_len: int = MAX_LEN) -> str:
    # structured: ',' + body + '.'
    body_len = random.randint(1, max(1, max_len - 2))
    body = []
    depth = 0
    for _ in range(body_len):
        # prefer to close if deep
        if depth > 0 and random.random() < 0.25:
            body.append(']')
            depth -= 1
        else:
            c = random.choice(BF_TOKENS)
            if c == '[':
                depth += 1
            body.append(c)
    body.extend(']' * depth)
    code = ',' + ''.join(body) + '.'
    return code[:max_len]

def is_balanced(s: str) -> bool:
    d = 0
    for ch in s:
        if ch == '[': d += 1
        elif ch == ']':
            d -= 1
            if d < 0: return False
    return d == 0

def mutate(program: str, max_len: int = MAX_LEN, p: float = 0.3) -> str:
    if random.random() > p:
        return program
    body = list(program[1:-1] if len(program) >= 2 else "+")
    op = random.random()
    if op < 0.33 and len(body) > 0:
        # point mutation
        i = random.randrange(len(body))
        choices = list(BF_TOKENS.replace(body[i], '')) or list(BF_TOKENS)
        body[i] = random.choice(choices)
    elif op < 0.66 and len(body) < max_len - 2:
        # insertion
        i = random.randrange(len(body) + 1)
        body.insert(i, random.choice(BF_TOKENS))
    elif len(body) > 1:
        # deletion
        i = random.randrange(len(body))
        body.pop(i)
    # rebuild and lightly repair by truncation; bracket validity enforced by rejection below
    candidate = ',' + ''.join(body)[:max_len-2] + '.'
    # Small chance to insert a minimal draining loop if none exists, to seed structure
    if '[' not in candidate and random.random() < 0.2 and len(candidate) < max_len:
        body2 = list(candidate[1:-1])
        insert_at = random.randrange(len(body2)+1)
        # Insert a tiny loop shell '[]' and let later mutations fill it
        body2.insert(insert_at, '[')
        body2.insert(min(insert_at+1, len(body2)), ']')
        candidate = ',' + ''.join(body2) + '.'
    # If unbalanced too often, try a few retries; else return as-is (fitness will punish)
    for _ in range(5):
        if is_balanced(candidate):
            return candidate
        # flip a bracket to try to balance
        body2 = list(candidate[1:-1])
        if '[' in body2 or ']' in body2:
            j = random.randrange(len(body2))
            body2[j] = random.choice('><+-')
        candidate = ',' + ''.join(body2) + '.'
    return candidate

def crossover(a: str, b: str) -> str:
    # single-point crossover inside body; keep head ',' and tail '.'
    abody, bbody = a[1:-1], b[1:-1]
    i = random.randrange(len(abody)+1) if abody else 0
    j = random.randrange(len(bbody)+1) if bbody else 0
    child_body = (abody[:i] + bbody[j:])[:MAX_LEN-2]
    child = ',' + child_body + '.'
    return child if is_balanced(child) else mutate(child, p=1.0)

@dataclass
class Ind:
    code: str
    fit: FitnessVector = None

def evolve_doubling_demo(pop_size=200, gens=150, report_every=10, seed=None):
    if seed is not None:
        random.seed(seed)
    print_header("Running evolutionary loop (doubling with staged fitness)")
    evaluator = StagedFitnessEvaluator()
    # init
    pop = [Ind(random_program()) for _ in range(pop_size)]

    def eval_pop(pop):
        fvs = []
        best = None
        for ind in pop:
            fv = evaluator.evaluate(ind.code, run_with_meta)
            ind.fit = fv
            fvs.append(fv)
            if best is None or (fv.exact_double, fv.close_double) > (best.fit.exact_double, best.fit.close_double):
                best = ind
        return fvs, best

    fvs, best = eval_pop(pop)
    print(f"Gen 0 | best exact={best.fit.exact_double:.2f} close={best.fit.close_double:.2f} IO={best.fit.io_contract:.2f}")

    for g in range(1, gens+1):
        # parents via staged lexicase
        children = []
        while len(children) < pop_size:
            i1 = select_parent_lexicase(pop, fvs)
            i2 = select_parent_lexicase(pop, fvs)
            p1, p2 = pop[i1].code, pop[i2].code
            child = crossover(p1, p2)
            child = mutate(child, p=0.6)
            children.append(Ind(child))
        pop = children
        fvs, best = eval_pop(pop)
        if g % report_every == 0 or best.fit.exact_double == 1.0:
            print(f"Gen {g} | best exact={best.fit.exact_double:.2f} close={best.fit.close_double:.2f} "
                  f"Δ2={best.fit.delta_slope2:.2f} Δconst={best.fit.delta_constancy:.2f} IO={best.fit.io_contract:.2f} "
                  f"loopEff={best.fit.loop_effective:.2f} | code: {best.code}")
        if best.fit.exact_double == 1.0:
            break

    # Final verification on a fixed validation set
    passed = True
    for x in evaluator.validate:
        meta = run_with_meta(best.code, x)
        y = meta['out'] if meta['out'] is not None else -1
        if y != (2*x) % 256:
            passed = False
            break
    print_subheader("Result")
    print(f"Best code: {best.code}")
    print(f"Perfect on validation: {passed}")

def demonstrate_staged_fitness_and_selection():
    print_header("Staged Fitness & Lexicase Selection (Doubling)")

    # Candidate genomes (toy)
    candidates = [
        (",+.", "increment (wrong task)"),
        (",[->++<].", "canonical doubling"),
        (",[->+<].", "move only (under-multiplies)"),
        (",[->+++<].", "tripling (wrong slope)"),
        (",.", "identity (wrong task)"),
    ]

    evaluator = StagedFitnessEvaluator()
    fitness_vecs = []
    for code, label in candidates:
        fv = evaluator.evaluate(code, run_with_meta)
        fitness_vecs.append(fv)
        print(f"\n{label:24} {code}")
        print(f"  IOContract     : {fv.io_contract:.2f}")
        print(f"  ΔConstancy     : {fv.delta_constancy:.2f}")
        print(f"  Δ≈2            : {fv.delta_slope2:.2f}")
        print(f"  Work↑          : {fv.work_increasing:.2f}")
        print(f"  Exact@Double   : {fv.exact_double:.2f}")
        print(f"  Close@Double   : {fv.close_double:.2f}")
        print(f"  ΔSlope≈2       : {fv.delta_slope2:.2f}")
        print(f"  LoopEffective  : {fv.loop_effective:.2f}")

    winner_idx = select_parent_lexicase([c[0] for c in candidates], fitness_vecs)
    print_subheader("Selected Parent (lexicase, staged)")
    print(f"Winner: {candidates[winner_idx][1]}  {candidates[winner_idx][0]}")

def main():
    """Main demonstration of experimental setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "evolve"], default="demo",
                        help="demo = print illustrative setup; evolve = run evolutionary loop for doubling")
    parser.add_argument("--gens", type=int, default=150)
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print("🧠 BRAINFUCK EVOLUTION BLOG DEMO")
    print("Part 2: Experimental Setup - Tasks vs Macros")

    if args.mode == "demo":
        # Define what we're testing
        curriculum_tasks = define_curriculum_tasks()
        # Demonstrate staged fitness & selection for doubling
        demonstrate_staged_fitness_and_selection()
        # Demonstrate macro emergence analysis
        analyze_macro_emergence()
        # Show evolutionary advantages
        demonstrate_evolutionary_advantage()
        # Explain curriculum design
        demonstrate_curriculum_design()
    else:
        # Run actual evolution
        evolve_doubling_demo(pop_size=args.pop, gens=args.gens, seed=args.seed)

    print_header("Summary: The Task/Macro Distinction")
    print("""
🎯 EXTERNAL TASKS (What we evaluate):
   • Functional requirements: f(x) = 2*x, f(x) = x+1, etc.
   • Black-box evaluation: only input/output matters
   • These create evolutionary PRESSURE

🔧 INTERNAL MACROS (What emerges): 
   • Code patterns: [-], [->+<], [->++<], etc.
   • Discovered through statistical analysis of evolved solutions
   • These become reusable BUILDING BLOCKS

🔄 THE BOOTSTRAPPING CYCLE:
   1. Tasks create pressure for efficient solutions
   2. Recurring patterns emerge across multiple tasks  
   3. Patterns get identified and labeled as macros
   4. Macros accelerate learning on new, complex tasks
   5. Meta-patterns emerge from macro composition

This demonstrates evolution transitioning from pattern-matching
(random exploration) to symbolic reasoning (compositional reuse)!
""")

if __name__ == "__main__":
    main()