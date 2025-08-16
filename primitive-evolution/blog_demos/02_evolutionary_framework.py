#!/usr/bin/env python3
"""
Blog Demo 2: Evolutionary Programming Framework

This script introduces the evolutionary approach to program synthesis.
Shows how genetic algorithms can automatically discover Brainfuck programs
that solve mathematical functions through natural selection.

Key concepts demonstrated:
- Population-based search
- Genetic operators (mutation, crossover, selection)
- Fitness evaluation
- Evolution converging on optimal solutions
"""

from brainfuck_evolution import EvolutionConfig, EvolutionRunner, Individual
import numpy as np
import time

def print_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def demonstrate_evolutionary_concepts():
    """Explain the evolutionary programming approach"""
    print_header("Evolutionary Programming Concepts")
    
    print("""
Evolutionary Algorithm for Program Synthesis:

1. POPULATION: Generate random Brainfuck programs
   - Each program is an "individual" in the population
   - Population size: 50-500 individuals
   - Initial programs are random sequences of BF commands

2. FITNESS EVALUATION: Test each program on target function
   - Run program with multiple test inputs
   - Compare outputs to expected results
   - Fitness = percentage of correct outputs

3. SELECTION: Choose parents based on fitness
   - Higher fitness = higher chance of reproduction
   - Tournament selection: pick best from random subsets

4. GENETIC OPERATORS:
   - Crossover: Combine parts of two parent programs
   - Mutation: Randomly change commands in programs
   - Create new generation of offspring

5. REPEAT: Continue until perfect solution found
""")

def demonstrate_simple_evolution():
    """Run evolution on a simple increment function"""
    print_header("Evolution Demo: f(x) = x + 1")
    
    # Create configuration for increment function
    config = EvolutionConfig(
        population_size=50,
        max_generations=20,
        input_output_mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},  # f(x) = x + 1
        use_genome_repository=False,  # Start fresh
        mutation_rate=0.15,
        max_program_length=20,
        target_fitness=100.0,
        function_name="f(x) = x + 1"
    )
    
    print("Evolution Configuration:")
    print(f"- Target function: f(x) = x + 1")
    print(f"- Population size: {config.population_size}")
    print(f"- Test cases: {list(config.input_output_mapping.items())}")
    print(f"- Mutation rate: {config.mutation_rate}")
    print(f"- Max program length: {config.max_program_length}")
    
    print("\nStarting evolution...")
    runner = EvolutionRunner(config)
    
    # Track evolution progress
    start_time = time.time()
    results = runner.run_evolution(interactive=False)
    end_time = time.time()
    
    print(f"\n🎉 Evolution completed in {end_time - start_time:.2f} seconds!")
    print(f"- Best fitness: {results['best_accuracy']:.1f}%")
    print(f"- Generations: {results['generations']}")
    print(f"- Best program: {results['best_code']}")
    print(f"- Program length: {len(results['best_code'])} characters")
    
    # Verify the solution manually
    if results['best_code']:
        print("\n🔍 Manual Verification:")
        from brainfuck import BrainfuckInterpreter
        interpreter = BrainfuckInterpreter()
        
        all_correct = True
        for inp, expected in config.input_output_mapping.items():
            try:
                result = interpreter.run(results['best_code'], chr(inp))
                actual = ord(result[0]) if result else 0
                correct = actual == expected
                all_correct &= correct
                status = "✓" if correct else "✗"
                print(f"  f({inp}) = {actual} (expected {expected}) {status}")
            except Exception as e:
                print(f"  f({inp}) = ERROR: {e} ✗")
                all_correct = False
        
        if all_correct:
            print("🏆 Perfect solution discovered by evolution!")
        else:
            print("⚠️ Partial solution found")

def analyze_population_diversity():
    """Demonstrate population diversity and convergence"""
    print_header("Population Diversity Analysis")
    
    # Create a fresh evolution run to analyze
    config = EvolutionConfig(
        population_size=30,
        max_generations=1,  # Just initialize population
        input_output_mapping={1: 2, 2: 4, 3: 6},  # f(x) = 2*x (simple)
        use_genome_repository=False,
        mutation_rate=0.2,
        max_program_length=15,
        function_name="Analysis Demo"
    )
    
    runner = EvolutionRunner(config)
    
    # Initialize population and evaluate
    runner.engine.initialize_population()
    
    # Analyze initial population
    population = runner.engine.population
    
    print("Initial Population Diversity:")
    print(f"- Population size: {len(population)}")
    
    # Show variety in program lengths
    lengths = [len(ind.code) for ind in population]
    print(f"- Program lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
    
    # Show some random programs
    print("\nSample Random Programs:")
    for i, individual in enumerate(population[:8]):
        print(f"  {i+1:2d}. '{individual.code}' (length: {len(individual.code)})")
    
    # Show fitness distribution
    fitness_values = [ind.fitness for ind in population]
    print(f"\nFitness Distribution:")
    print(f"- Best fitness: {max(fitness_values):.1f}%")
    print(f"- Average fitness: {np.mean(fitness_values):.1f}%")
    print(f"- Worst fitness: {min(fitness_values):.1f}%")
    
    # Count how many different programs we have
    unique_programs = len(set(ind.code for ind in population))
    print(f"- Unique programs: {unique_programs}/{len(population)} ({unique_programs/len(population)*100:.1f}%)")

def demonstrate_evolution_progress():
    """Show how evolution progresses over generations"""
    print_header("Evolution Progress Over Generations")
    
    config = EvolutionConfig(
        population_size=40,
        max_generations=15,
        input_output_mapping={0: 0, 1: 2, 2: 4, 3: 6, 4: 8},  # f(x) = 2*x
        use_genome_repository=False,
        mutation_rate=0.12,
        max_program_length=25,
        function_name="f(x) = 2*x tracking demo"
    )
    
    print("Tracking evolution progress for f(x) = 2*x")
    print("Generation | Best Fitness | Avg Fitness | Best Program")
    print("-" * 65)
    
    runner = EvolutionRunner(config)
    
    # We'll manually step through generations to show progress
    runner.engine.initialize_population()
    
    for gen in range(config.max_generations):
        # Get current stats
        stats = runner.engine.get_stats()
        best_individual = max(runner.engine.population, key=lambda x: x.fitness)
        
        print(f"{gen:9d} | {stats['best_fitness']:11.1f}% | {stats['avg_fitness']:10.1f}% | {best_individual.code}")
        
        # Check if we found perfect solution
        if stats['best_fitness'] >= config.target_fitness:
            print(f"\n🎉 Perfect solution found in generation {gen}!")
            break
        
        # Evolve to next generation
        result = runner.engine.evolve_generation()
        if result == 'PERFECT':
            break
    
    print(f"\nEvolution completed!")
    final_stats = runner.engine.get_stats()
    print(f"Final best fitness: {final_stats['best_fitness']:.1f}%")

def main():
    """Main demonstration function"""
    print("🧬 BRAINFUCK EVOLUTION BLOG DEMO")
    print("Part 2: Evolutionary Programming Framework")
    
    demonstrate_evolutionary_concepts()
    demonstrate_simple_evolution()
    analyze_population_diversity()
    demonstrate_evolution_progress()
    
    print_header("Summary")
    print("""
Key Takeaways:
1. Evolution automatically discovers programs without human design
2. Population diversity enables exploration of solution space
3. Fitness selection drives convergence toward optimal solutions
4. Simple functions can be evolved quickly (usually <20 generations)
5. Genetic operators (mutation/crossover) create program variations

Next: We'll see evolution tackle progressively harder problems and
build up a repository of successful solutions!
""")

if __name__ == "__main__":
    main()