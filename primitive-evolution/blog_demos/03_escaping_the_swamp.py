#!/usr/bin/env python3
"""
Blog Demo 3: Escaping the Search Space Swamp

This script demonstrates why structural constraints and staged fitness are
essential for evolutionary program synthesis. Without them, the search space
is so noisy that meaningful patterns never emerge.

Key demonstrations:
1. BASELINE (The Swamp): Unconstrained evolution - shows pathological solutions
2. CONSTRAINED: Evolution with structural scaffolding - shows meaningful progress  
3. COMPARISON: Clear metrics showing the dramatic difference

The goal is to show that without scaffolding, evolution drowns in noise.
With scaffolding, it can discover useful building blocks and make progress.
"""

import argparse
import random
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from brainfuck import BrainfuckInterpreter

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
class EvolutionStats:
    """Statistics for comparing evolution approaches."""
    valid_brackets: float          # % with balanced brackets
    single_input: float           # % with exactly one input
    single_output: float          # % with exactly one output
    io_contract: float            # % with exactly one input AND one output
    terminates: float             # % that terminate (don't timeout)
    non_constant: float           # % that produce different outputs for different inputs
    meaningful_fitness: float     # % with fitness > 10 (some signal)
    best_fitness: float           # Best fitness achieved
    best_accuracy: float          # Best accuracy (exact matches)
    example_solutions: List[str]  # Top 3 solutions found

class BaselineEvolution:
    """Unconstrained evolution - the search space swamp."""
    
    def __init__(self, max_length: int = 50):
        self.max_length = max_length
        self.bf_chars = '><+-.,[]'
        self.interpreter = BrainfuckInterpreter()
    
    def generate_random_program(self) -> str:
        """Generate completely random BF program with no constraints."""
        length = random.randint(3, self.max_length)
        return ''.join(random.choice(self.bf_chars) for _ in range(length))
    
    def mutate(self, program: str, rate: float = 0.3) -> str:
        """Random mutation with no structural preservation."""
        if random.random() > rate:
            return program
        
        program = list(program)
        mutation_type = random.random()
        
        if mutation_type < 0.4 and len(program) > 1:
            # Point mutation
            pos = random.randint(0, len(program) - 1)
            program[pos] = random.choice(self.bf_chars)
        elif mutation_type < 0.7 and len(program) < self.max_length:
            # Insertion
            pos = random.randint(0, len(program))
            program.insert(pos, random.choice(self.bf_chars))
        elif len(program) > 3:
            # Deletion
            pos = random.randint(0, len(program) - 1)
            program.pop(pos)
        
        return ''.join(program)[:self.max_length]
    
    def crossover(self, parent1: str, parent2: str) -> str:
        """Simple crossover with no structural awareness."""
        if not parent1 or not parent2:
            return parent1 or parent2
        
        cut1 = random.randint(0, len(parent1))
        cut2 = random.randint(0, len(parent2))
        
        child = parent1[:cut1] + parent2[cut2:]
        return child[:self.max_length]
    
    def evaluate_fitness(self, program: str, test_cases: Dict[int, int]) -> Tuple[float, dict]:
        """Primitive fitness evaluation - just count correct outputs."""
        total_score = 0.0
        correct_count = 0
        outputs = []
        timeouts = 0
        
        for input_val, expected_output in test_cases.items():
            try:
                self.interpreter.__init__()  # Reset
                result = self.interpreter.run(program, chr(input_val))
                
                if self.interpreter.hit_step_limit:
                    timeouts += 1
                    outputs.append(None)
                    continue
                
                if result:
                    actual_output = ord(result[0])
                    outputs.append(actual_output)
                    
                    # Simple distance-based scoring
                    error = abs(actual_output - expected_output)
                    case_score = max(0, 100 - error)
                    total_score += case_score
                    
                    if actual_output == expected_output:
                        correct_count += 1
                else:
                    outputs.append(0)
                    
            except Exception:
                outputs.append(None)
                timeouts += 1
        
        avg_score = total_score / len(test_cases) if test_cases else 0
        
        metadata = {
            'outputs': outputs,
            'timeouts': timeouts,
            'total_timeouts': timeouts / len(test_cases),
            'reads': program.count(','),
            'writes': program.count('.'),
            'terminates': timeouts == 0
        }
        
        return avg_score, metadata

class ConstrainedEvolution:
    """Evolution with structural constraints and staged fitness."""
    
    def __init__(self, max_length: int = 40):
        self.max_length = max_length
        self.body_chars = '><+-[]'
        self.interpreter = BrainfuckInterpreter()
    
    def generate_random_program(self) -> str:
        """Generate structured BF program: ,<body>."""
        body_length = random.randint(1, max(1, self.max_length - 2))
        body = self._generate_balanced_body(body_length)
        return ',' + body + '.'
    
    def _generate_balanced_body(self, length: int) -> str:
        """Generate a balanced bracket body."""
        body = []
        depth = 0
        
        for _ in range(length):
            if depth > 0 and random.random() < 0.3:
                # Close bracket
                body.append(']')
                depth -= 1
            else:
                char = random.choice(self.body_chars)
                if char == '[':
                    depth += 1
                body.append(char)
        
        # Close remaining brackets
        body.extend(']' * depth)
        return ''.join(body[:length])
    
    def _is_balanced(self, program: str) -> bool:
        """Check if brackets are balanced."""
        depth = 0
        for char in program:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0
    
    def mutate(self, program: str, rate: float = 0.3) -> str:
        """Structure-preserving mutation."""
        if random.random() > rate:
            return program
        
        if len(program) < 3:
            return self.generate_random_program()
        
        # Extract body (between , and .)
        body = list(program[1:-1])
        
        mutation_type = random.random()
        
        if mutation_type < 0.4 and body:
            # Point mutation in body
            pos = random.randint(0, len(body) - 1)
            body[pos] = random.choice(self.body_chars)
        elif mutation_type < 0.7 and len(body) < self.max_length - 2:
            # Insertion in body
            pos = random.randint(0, len(body))
            body.insert(pos, random.choice(self.body_chars))
        elif len(body) > 1:
            # Deletion in body
            pos = random.randint(0, len(body) - 1)
            body.pop(pos)
        
        # Reconstruct program
        candidate = ',' + ''.join(body) + '.'
        
        # Ensure balanced brackets
        if not self._is_balanced(candidate):
            # Try to fix by removing problematic brackets
            fixed_body = []
            depth = 0
            for char in body:
                if char == '[':
                    fixed_body.append(char)
                    depth += 1
                elif char == ']' and depth > 0:
                    fixed_body.append(char)
                    depth -= 1
                elif char not in '[]':
                    fixed_body.append(char)
            # Close remaining
            fixed_body.extend(']' * depth)
            candidate = ',' + ''.join(fixed_body) + '.'
        
        return candidate[:self.max_length]
    
    def crossover(self, parent1: str, parent2: str) -> str:
        """Structure-preserving crossover on bodies only."""
        if len(parent1) < 3 or len(parent2) < 3:
            return random.choice([parent1, parent2])
        
        # Extract bodies
        body1 = parent1[1:-1]
        body2 = parent2[1:-1]
        
        # Simple body crossover
        if body1 and body2:
            cut1 = random.randint(0, len(body1))
            cut2 = random.randint(0, len(body2))
            child_body = body1[:cut1] + body2[cut2:]
        else:
            child_body = body1 or body2
        
        child = ',' + child_body + '.'
        
        # Ensure balanced and within length
        if not self._is_balanced(child) or len(child) > self.max_length:
            return self.mutate(random.choice([parent1, parent2]), rate=1.0)
        
        return child
    
    def evaluate_fitness_staged(self, program: str, test_cases: Dict[int, int]) -> Tuple[float, dict]:
        """Staged fitness evaluation with behavioral shaping."""
        
        # Stage 0: Structural gating
        if not (program.startswith(',') and program.endswith('.')):
            return 0.0, {'stage': 0, 'reason': 'no_io_structure'}
        
        if not self._is_balanced(program):
            return 5.0, {'stage': 0, 'reason': 'unbalanced_brackets'}
        
        reads = program.count(',')
        writes = program.count('.')
        
        if reads != 1 or writes != 1:
            return 10.0, {'stage': 0, 'reason': 'wrong_io_count'}
        
        # Execute and gather behavioral data
        outputs = []
        timeouts = 0
        
        for input_val in test_cases.keys():
            try:
                self.interpreter.__init__()
                result = self.interpreter.run(program, chr(input_val))
                
                if self.interpreter.hit_step_limit:
                    timeouts += 1
                    outputs.append(None)
                else:
                    actual = ord(result[0]) if result else 0
                    outputs.append(actual)
                    
            except Exception:
                timeouts += 1
                outputs.append(None)
        
        if timeouts > len(test_cases) * 0.5:  # More than half timeout
            return 15.0, {'stage': 1, 'reason': 'excessive_timeouts'}
        
        # Stage 1: Behavioral shaping
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs:
            return 20.0, {'stage': 1, 'reason': 'no_valid_outputs'}
        
        # Check for input dependency (not constant)
        input_dependency = len(set(valid_outputs)) > 1
        if not input_dependency:
            return 25.0, {'stage': 1, 'reason': 'constant_output'}
        
        # Behavioral score based on slope consistency (for doubling)
        behavioral_score = 30.0
        test_pairs = list(test_cases.items())
        
        if len(valid_outputs) >= len(test_pairs):
            # Calculate deltas for consecutive inputs
            deltas = []
            for i in range(len(test_pairs) - 1):
                x1, _ = test_pairs[i]
                x2, _ = test_pairs[i + 1]
                if x2 == x1 + 1:  # Consecutive inputs
                    actual1 = outputs[i] if outputs[i] is not None else 0
                    actual2 = outputs[i + 1] if outputs[i + 1] is not None else 0
                    delta = (actual2 - actual1) % 256
                    deltas.append(delta)
            
            if deltas:
                # Reward deltas close to 2 (for doubling)
                target_delta = 2
                delta_consistency = 1.0 - np.std(deltas) / 10.0
                delta_correctness = 1.0 - abs(np.mean(deltas) - target_delta) / 10.0
                
                behavioral_score += max(0, delta_consistency * 20)
                behavioral_score += max(0, delta_correctness * 20)
        
        # Stage 2: Exactness
        correct_count = 0
        total_score = behavioral_score
        
        for i, (input_val, expected_output) in enumerate(test_cases.items()):
            if i < len(outputs) and outputs[i] is not None:
                if outputs[i] == expected_output:
                    correct_count += 1
                    total_score += 10  # Bonus for exact matches
        
        accuracy = correct_count / len(test_cases)
        
        metadata = {
            'stage': 2,
            'outputs': outputs,
            'accuracy': accuracy,
            'behavioral_score': behavioral_score,
            'input_dependency': input_dependency,
            'timeouts': timeouts,
            'total_timeouts': timeouts / len(test_cases),
            'terminates': timeouts == 0,
            'reads': reads,
            'writes': writes
        }
        
        return total_score, metadata

def analyze_population_quality(programs: List[str], approach_name: str) -> EvolutionStats:
    """Analyze the quality of a population."""
    print(f"\n📊 Analyzing {approach_name} Population Quality")
    
    valid_brackets = 0
    single_input = 0
    single_output = 0
    io_contract = 0
    terminates = 0
    non_constant = 0
    meaningful_fitness = 0
    
    fitness_scores = []
    accuracies = []
    example_solutions = []
    
    # Test cases for doubling
    test_cases = {1: 2, 2: 4, 3: 6, 4: 8}
    
    if 'Baseline' in approach_name:
        evaluator = BaselineEvolution()
        eval_func = evaluator.evaluate_fitness
    else:
        evaluator = ConstrainedEvolution()
        eval_func = evaluator.evaluate_fitness_staged
    
    for program in programs[:100]:  # Sample first 100 for analysis
        # Structural analysis
        depth = 0
        is_balanced = True
        for char in program:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth < 0:
                    is_balanced = False
                    break
        
        if depth == 0 and is_balanced:
            valid_brackets += 1
        
        reads = program.count(',')
        writes = program.count('.')
        
        if reads == 1:
            single_input += 1
        if writes == 1:
            single_output += 1
        if reads == 1 and writes == 1:
            io_contract += 1
        
        # Behavioral analysis
        try:
            fitness, metadata = eval_func(program, test_cases)
            fitness_scores.append(fitness)
            
            if 'accuracy' in metadata:
                accuracies.append(metadata['accuracy'])
            
            # Check termination - prefer explicit terminates flag, fall back to timeout analysis
            if 'terminates' in metadata:
                if metadata['terminates']:
                    terminates += 1
            elif 'timeouts' in metadata:
                # Fallback: program terminates if no timeouts occurred
                if metadata.get('timeouts', 0) == 0:
                    terminates += 1
            
            # Check for non-constant output
            if 'outputs' in metadata:
                outputs = [o for o in metadata['outputs'] if o is not None]
                if len(set(outputs)) > 1:
                    non_constant += 1
            
            if fitness > 10:
                meaningful_fitness += 1
                example_solutions.append((program, fitness))
                
        except Exception:
            fitness_scores.append(0)
            accuracies.append(0)
    
    # Sort examples by fitness
    example_solutions.sort(key=lambda x: x[1], reverse=True)
    best_examples = [prog for prog, _ in example_solutions[:3]]
    
    total = len(programs[:100])
    return EvolutionStats(
        valid_brackets=valid_brackets / total * 100,
        single_input=single_input / total * 100,
        single_output=single_output / total * 100,
        io_contract=io_contract / total * 100,
        terminates=terminates / total * 100,
        non_constant=non_constant / total * 100,
        meaningful_fitness=meaningful_fitness / total * 100,
        best_fitness=max(fitness_scores) if fitness_scores else 0,
        best_accuracy=max(accuracies) if accuracies else 0,
        example_solutions=best_examples
    )

def run_baseline_evolution(pop_size: int = 100, generations: int = 50) -> List[str]:
    """Run unconstrained evolution - the search space swamp."""
    print_subheader("Running Baseline Evolution (The Swamp)")
    
    evolver = BaselineEvolution()
    test_cases = {1: 2, 2: 4, 3: 6, 4: 8}  # Doubling function
    
    # Initialize random population
    population = [evolver.generate_random_program() for _ in range(pop_size)]
    
    print(f"Initial population examples:")
    for i in range(5):
        print(f"  {population[i]}")
    
    best_fitness = 0
    best_program = ""
    
    for gen in range(generations):
        # Evaluate population
        fitness_scores = []
        for program in population:
            fitness, _ = evolver.evaluate_fitness(program, test_cases)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_program = program
        
        if gen % 10 == 0 or gen == generations - 1:
            avg_fitness = np.mean(fitness_scores)
            print(f"Gen {gen:2d}: avg_fitness={avg_fitness:.1f}, best_fitness={best_fitness:.1f}")
        
        # Selection and reproduction
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            indices = np.random.choice(len(population), 3)
            tournament_scores = [fitness_scores[i] for i in indices]
            parent1 = population[indices[np.argmax(tournament_scores)]]
            
            indices = np.random.choice(len(population), 3)
            tournament_scores = [fitness_scores[i] for i in indices]
            parent2 = population[indices[np.argmax(tournament_scores)]]
            
            child = evolver.crossover(parent1, parent2)
            child = evolver.mutate(child)
            new_population.append(child)
        
        population = new_population
    
    print(f"\nBest solution found: '{best_program}' (fitness: {best_fitness:.1f})")
    return population

def run_constrained_evolution(pop_size: int = 100, generations: int = 50) -> List[str]:
    """Run constrained evolution with structural scaffolding."""
    print_subheader("Running Constrained Evolution (With Scaffolding)")
    
    evolver = ConstrainedEvolution()
    test_cases = {1: 2, 2: 4, 3: 6, 4: 8}  # Doubling function
    
    # Initialize structured population
    population = [evolver.generate_random_program() for _ in range(pop_size)]
    
    print(f"Initial population examples:")
    for i in range(5):
        print(f"  {population[i]}")
    
    best_fitness = 0
    best_program = ""
    
    for gen in range(generations):
        # Evaluate population
        fitness_scores = []
        for program in population:
            fitness, _ = evolver.evaluate_fitness_staged(program, test_cases)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_program = program
        
        if gen % 10 == 0 or gen == generations - 1:
            avg_fitness = np.mean(fitness_scores)
            print(f"Gen {gen:2d}: avg_fitness={avg_fitness:.1f}, best_fitness={best_fitness:.1f}")
        
        # Selection and reproduction with better selection pressure
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            tournament_size = 5
            contestants = random.sample(list(zip(population, fitness_scores)), tournament_size)
            parent1 = max(contestants, key=lambda x: x[1])[0]
            
            contestants = random.sample(list(zip(population, fitness_scores)), tournament_size)
            parent2 = max(contestants, key=lambda x: x[1])[0]
            
            child = evolver.crossover(parent1, parent2)
            child = evolver.mutate(child)
            new_population.append(child)
        
        population = new_population
    
    print(f"\nBest solution found: '{best_program}' (fitness: {best_fitness:.1f})")
    return population

def demonstrate_pathological_solutions():
    """Show examples of pathological solutions from unconstrained search."""
    print_header("Pathological Solutions: Why Constraints Matter")
    
    # Generate some typical "bad" solutions
    pathological_examples = [
        (
            ".,.,.,.", 
            "Multiple I/O: Reads nothing, outputs 4 zeros",
            "Violates single input/output contract"
        ),
        (
            ",[[[[+]]]", 
            "Infinite loop: Never terminates", 
            "Unproductive computation"
        ),
        (
            ".+.+.+.", 
            "No input: Outputs constants regardless of input",
            "Ignores input completely"
        ),
        (
            ",[>+<-,[>+<-].", 
            "Multiple inputs: Complex but wrong I/O pattern",
            "Structural chaos"
        ),
        (
            "++++++++++++++++.", 
            "Constant output: Always outputs 16",
            "No input dependency"
        )
    ]
    
    print("\nCommon pathological patterns in unconstrained search:")
    print("\nProgram              | Behavior                    | Problem")
    print("-" * 75)
    
    for program, behavior, problem in pathological_examples:
        print(f"{program:20} | {behavior:27} | {problem}")
    
    print(f"\n💡 Key Problems:")
    print("   • Multiple inputs/outputs break the function contract")
    print("   • Infinite loops waste computation without progress")
    print("   • Constant outputs ignore input (no learning signal)")
    print("   • Structural chaos makes crossover destructive")
    print("   • Fitness signals are drowned in noise")

def compare_approaches():
    """Run both approaches and compare results."""
    print_header("The Great Comparison: Swamp vs Scaffolding")
    
    print("Running baseline evolution (no constraints)...")
    baseline_pop = run_baseline_evolution(pop_size=200, generations=50)
    baseline_stats = analyze_population_quality(baseline_pop, "Baseline (Swamp)")
    
    print("\nRunning constrained evolution (with scaffolding)...")
    constrained_pop = run_constrained_evolution(pop_size=200, generations=50)
    constrained_stats = analyze_population_quality(constrained_pop, "Constrained (Scaffolding)")
    
    # Print comparison table
    print_subheader("Comparison Results")
    
    print("\nQuality Metrics Comparison:")
    print(f"{'Metric':<20} | {'Baseline':>10} | {'Constrained':>12} | {'Improvement':>12}")
    print("-" * 65)
    
    metrics = [
        ('Valid Brackets %', baseline_stats.valid_brackets, constrained_stats.valid_brackets),
        ('Single Input %', baseline_stats.single_input, constrained_stats.single_input),
        ('Single Output %', baseline_stats.single_output, constrained_stats.single_output),
        ('I/O Contract %', baseline_stats.io_contract, constrained_stats.io_contract),
        ('Terminates %', baseline_stats.terminates, constrained_stats.terminates),
        ('Non-Constant %', baseline_stats.non_constant, constrained_stats.non_constant),
        ('Meaningful Fit %', baseline_stats.meaningful_fitness, constrained_stats.meaningful_fitness),
        ('Best Fitness', baseline_stats.best_fitness, constrained_stats.best_fitness),
        ('Best Accuracy', baseline_stats.best_accuracy * 100, constrained_stats.best_accuracy * 100)
    ]
    
    for metric_name, baseline_val, constrained_val in metrics:
        if baseline_val == 0:
            improvement = "∞" if constrained_val > 0 else "0"
        else:
            improvement = f"{constrained_val/baseline_val:.1f}x"
        
        print(f"{metric_name:<20} | {baseline_val:>10.1f} | {constrained_val:>12.1f} | {improvement:>12}")
    
    # Show example solutions
    print_subheader("Example Solutions")
    
    print("\nBaseline (Swamp) - Best Solutions:")
    for i, solution in enumerate(baseline_stats.example_solutions[:3], 1):
        print(f"  {i}. {solution}")
    
    print("\nConstrained (Scaffolding) - Best Solutions:")
    for i, solution in enumerate(constrained_stats.example_solutions[:3], 1):
        print(f"  {i}. {solution}")
    
    return baseline_stats, constrained_stats

def main():
    """Main demonstration of escaping the search space swamp."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "compare", "pathological"], default="demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    print("🧠 BRAINFUCK EVOLUTION BLOG DEMO")
    print("Part 3: Escaping the Search Space Swamp")
    
    if args.mode == "pathological":
        demonstrate_pathological_solutions()
    elif args.mode == "compare":
        baseline_stats, constrained_stats = compare_approaches()
    else:
        # Full demo
        demonstrate_pathological_solutions()
        baseline_stats, constrained_stats = compare_approaches()
        
        print_header("Summary: Why Scaffolding Matters")
        print(f"""
🌊 THE SEARCH SPACE SWAMP:
   • Random programs violate basic contracts (I/O, brackets, termination)
   • Fitness signals are drowned in structural noise  
   • Evolution wastes time on pathological solutions
   • No meaningful building blocks can emerge
   • Progress: {baseline_stats.best_accuracy*100:.1f}% accuracy

🏗️ STRUCTURAL SCAFFOLDING:
   • Enforced contracts create "program-shaped" genomes
   • Staged fitness rewards behavioral progress
   • Evolution can focus on meaningful improvements  
   • Building blocks have space to emerge and be preserved
   • Progress: {constrained_stats.best_accuracy*100:.1f}% accuracy

📈 THE TRANSFORMATION:
   • I/O Contract: {baseline_stats.io_contract:.1f}% → {constrained_stats.io_contract:.1f}%
   • Termination: {baseline_stats.terminates:.1f}% → {constrained_stats.terminates:.1f}%
   • Meaningful Fitness: {baseline_stats.meaningful_fitness:.1f}% → {constrained_stats.meaningful_fitness:.1f}%

💡 Key Insight: Without scaffolding, evolution drowns in noise.
   With scaffolding, it can discover and compose building blocks.
   This is the foundation for emergent symbolic reasoning!
""")

if __name__ == "__main__":
    main()