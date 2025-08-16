#!/usr/bin/env python3
"""
Brainfuck Evolution: General Function Calculator
A genetic algorithm implementation for evolving Brainfuck programs.
"""

import random
import time
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import dependencies from local copies
from brainfuck import BrainfuckInterpreter
from genome_repository import GenomeRepository, get_global_repository


@dataclass
class TestResult:
    """Result of testing an individual on a single test case."""
    input_val: int
    expected: int
    actual: int
    fitness: float
    error: Optional[str] = None


@dataclass
class EvolutionConfig:
    """Configuration parameters for the evolution process."""
    population_size: int = 50
    mutation_rate: float = 0.1
    max_program_length: int = 50
    execution_timeout_ms: int = 100
    test_cases: List[int] = None
    expected_outputs: List[int] = None
    input_output_mapping: Dict[int, int] = None
    generation_delay_ms: int = 100
    elite_ratio: float = 0.1
    tournament_size: int = 3
    crossover_rate: float = 0.7
    max_generations: int = 1000
    target_fitness: float = 100.0
    function_name: str = None  # Description of the target function

    # Migration options
    migration_rate: float = 0.05  # Fraction of population to replace during migration
    migration_frequency: int = 10  # Migration occurs every N generations

    # Genome repository options
    use_genome_repository: bool = True  # Whether to use stored genomes for seeding
    save_successful_genomes: bool = True  # Whether to save successful genomes
    repository_seed_ratio: float = 0.3  # Fraction of population to seed from repository
    min_accuracy_to_save: float = 100.0  # Minimum accuracy to save to repository

    def __post_init__(self):
        # Handle different ways of specifying test cases and expected outputs
        if self.input_output_mapping is not None:
            # Use explicit input-output mapping
            self.test_cases = list(self.input_output_mapping.keys())
            self.expected_outputs = list(self.input_output_mapping.values())
        elif self.test_cases is not None and self.expected_outputs is not None:
            # Use separate lists for inputs and outputs
            if len(self.test_cases) != len(self.expected_outputs):
                raise ValueError("test_cases and expected_outputs must have the same length")
        elif self.test_cases is None and self.expected_outputs is None:
            raise ValueError("Either test_cases and expected_outputs must be specified, or input_output_mapping must be specified")
        else:
            raise ValueError("Either test_cases and expected_outputs must be specified, or input_output_mapping must be specified")

        # Create mapping for easy lookup
        if self.input_output_mapping is None:
            self.input_output_mapping = dict(zip(self.test_cases, self.expected_outputs))


class Individual:
    """Represents a single Brainfuck program in the population."""

    def __init__(self, code: str = "", max_length: int = 50):
        self.max_length = max_length
        self.code = code or self._generate_random()
        self.fitness = 0.0
        self.test_results: List[TestResult] = []
        self.generation_created = 0
    
    def _generate_random(self) -> str:
        """Generate a random Brainfuck program."""
        commands = '><+-.,[]'
        # Use instance max_length if available, otherwise default
        max_program_length = getattr(self, 'max_length', 50)
        length = random.randint(5, max_program_length)
        code = ''
        bracket_depth = 0
        
        for _ in range(length):
            # Bias towards closing brackets if we have open ones
            if bracket_depth > 0 and random.random() < 0.3:
                cmd = ']'
                bracket_depth -= 1
            else:
                cmd = random.choice(commands)
                if cmd == '[':
                    bracket_depth += 1
            code += cmd
        
        # Close any remaining brackets
        code += ']' * bracket_depth
        
        return code
    
    def mutate(self, config: EvolutionConfig) -> None:
        """Apply mutation to this individual."""
        if random.random() > config.mutation_rate:
            return

        commands = '><+-.,[]'
        new_code = self.code

        # Choose mutation type
        mutation_type = random.random()

        if mutation_type < 0.4 and len(new_code) > 0:
            # Point mutation
            pos = random.randint(0, len(new_code) - 1)
            new_cmd = random.choice(commands)
            new_code = new_code[:pos] + new_cmd + new_code[pos + 1:]
        elif mutation_type < 0.7:
            # Insertion
            pos = random.randint(0, len(new_code))
            new_cmd = random.choice(commands)
            new_code = new_code[:pos] + new_cmd + new_code[pos:]
        elif len(new_code) > 1:
            # Deletion
            pos = random.randint(0, len(new_code) - 1)
            new_code = new_code[:pos] + new_code[pos + 1:]

        # Ensure bracket matching and length limits
        self.code = self._fix_brackets(new_code)
        if len(self.code) > config.max_program_length:
            self.code = self.code[:config.max_program_length]
            self.code = self._fix_brackets(self.code)
    
    def _fix_brackets(self, code: str) -> str:
        """Fix bracket matching in the code."""
        depth = 0
        fixed = ''
        
        for char in code:
            if char == '[':
                depth += 1
                fixed += char
            elif char == ']' and depth > 0:
                depth -= 1
                fixed += char
            elif char != ']':
                fixed += char
        
        # Close remaining brackets
        fixed += ']' * depth
        
        return fixed
    
    def clone(self) -> 'Individual':
        """Create a copy of this individual."""
        clone = Individual(self.code, max_length=getattr(self, 'max_length', 50))
        clone.fitness = self.fitness
        clone.generation_created = self.generation_created
        clone.test_results = self.test_results.copy()  # Copy test results!
        return clone


class EvolutionEngine:
    """Main evolution engine for Brainfuck programs."""
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.population: List[Individual] = []
        self.generation = 0
        self.running = False
        self.interpreter = BrainfuckInterpreter()
        self.best_ever: Optional[Individual] = None
        self.best_accuracy_ever = 0.0
        self.evolution_log: List[str] = []
        self.genome_repository = get_global_repository() if config.use_genome_repository else None

    def initialize_population(self) -> None:
        """Initialize the population with random individuals and genome repository seeding."""
        self.population = []
        seeded_count = 0

        # Seed from genome repository if enabled
        if self.genome_repository and self.config.use_genome_repository:
            seed_count = int(self.config.population_size * self.config.repository_seed_ratio)
            seed_genomes = self.genome_repository.export_for_seeding(
                function_name=self.config.function_name,
                limit=seed_count
            )

            for genome_code in seed_genomes:
                individual = Individual(genome_code)
                individual.generation_created = 0
                self.population.append(individual)
                seeded_count += 1

            if seeded_count > 0:
                self.log(f"Seeded {seeded_count} individuals from genome repository")

        # Fill remaining population with random individuals
        while len(self.population) < self.config.population_size:
            individual = Individual(max_length=self.config.max_program_length)
            individual.generation_created = 0
            self.population.append(individual)

        self.generation = 0
        self.best_ever = None
        self.best_accuracy_ever = 0.0

        random_count = len(self.population) - seeded_count
        self.log(f"Population initialized: {seeded_count} seeded, {random_count} random individuals")
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate the fitness of an individual against all test cases."""
        total_fitness = 0.0
        individual.test_results = []

        for input_val in self.config.test_cases:
            try:
                expected = self.config.input_output_mapping[input_val]
                # Create fresh interpreter for each test to avoid state issues
                interpreter = BrainfuckInterpreter()
                result = interpreter.run(individual.code, chr(input_val))

                if result:
                    actual = ord(result[0])
                else:
                    actual = 0

                # Calculate fitness based on accuracy
                if actual == expected:
                    case_fitness = 100.0
                else:
                    case_fitness = 0.0

                total_fitness += case_fitness
                individual.test_results.append(TestResult(
                    input_val=input_val,
                    expected=expected,
                    actual=actual,
                    fitness=case_fitness
                ))

            except Exception as e:
                expected = self.config.input_output_mapping.get(input_val, 0)
                individual.test_results.append(TestResult(
                    input_val=input_val,
                    expected=expected,
                    actual=0,
                    fitness=0.0,
                    error=str(e)
                ))

        individual.fitness = total_fitness / len(self.config.test_cases)
        return individual.fitness

    def evolve_generation(self) -> str:
        """Evolve one generation and return status."""
        # Evaluate fitness for all individuals
        for individual in self.population:
            self.evaluate_fitness(individual)

        # Sort by fitness (best first)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Track best individual
        if not self.best_ever or self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0].clone()
            self.best_accuracy_ever = self.best_ever.fitness

            # Check for perfect solution
            if self.best_accuracy_ever >= self.config.target_fitness:
                self.log("🎉 PERFECT SOLUTION FOUND! 🎉")
                self.log(f"Final code: {self.best_ever.code}")
                self.log("All test cases passed with 100% accuracy!")
                return 'PERFECT'

        # Create new generation
        new_population = []
        elite_count = max(1, int(self.config.population_size * self.config.elite_ratio))

        # Elite selection (keep best individuals)
        for i in range(elite_count):
            elite = self.population[i].clone()
            elite.generation_created = self.generation + 1
            new_population.append(elite)

        # Tournament selection and reproduction
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            child = self.crossover(parent1, parent2)
            child.mutate(self.config)
            child.generation_created = self.generation + 1
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        return 'CONTINUE'

    def tournament_select(self) -> Individual:
        """Select an individual using tournament selection."""
        tournament = random.sample(self.population,
                                 min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create offspring through crossover."""
        if random.random() < self.config.crossover_rate:
            # Single point crossover
            point1 = random.randint(0, len(parent1.code))
            point2 = random.randint(0, len(parent2.code))

            new_code = parent1.code[:point1] + parent2.code[point2:]
            child = Individual(max_length=self.config.max_program_length)
            child.code = child._fix_brackets(new_code)

            # Enforce maximum program length
            if len(child.code) > self.config.max_program_length:
                child.code = child.code[:self.config.max_program_length]
                child.code = child._fix_brackets(child.code)

            return child
        else:
            # Return copy of random parent
            return random.choice([parent1, parent2]).clone()

    def get_stats(self) -> Dict[str, float]:
        """Get current population statistics."""
        if not self.population:
            return {
                'avg_fitness': 0.0,
                'success_rate': 0.0,
                'best_fitness': 0.0,
                'best_accuracy': 0.0,
                'avg_accuracy': 0.0
            }

        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        success_count = sum(1 for ind in self.population if ind.fitness > 90)
        success_rate = (success_count / len(self.population)) * 100
        best_fitness = max(ind.fitness for ind in self.population)

        # Calculate accuracy statistics
        accuracies = []
        for ind in self.population:
            if ind.test_results:
                correct_count = sum(1 for r in ind.test_results if r.actual == r.expected)
                total_count = len(ind.test_results)
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
                accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        best_accuracy = max(accuracies) if accuracies else 0

        return {
            'avg_fitness': avg_fitness,
            'success_rate': success_rate,
            'best_fitness': best_fitness,
            'best_accuracy': best_accuracy,
            'avg_accuracy': avg_accuracy
        }

    def save_successful_genomes(self, force_save: bool = False) -> int:
        """Save successful genomes to the repository."""
        if not self.genome_repository or not self.config.save_successful_genomes:
            return 0

        saved_count = 0

        for individual in self.population:
            if individual.test_results:
                # Calculate accuracy
                correct_count = sum(1 for r in individual.test_results if r.actual == r.expected)
                total_count = len(individual.test_results)
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

                # Save if meets criteria
                if accuracy >= self.config.min_accuracy_to_save or force_save:
                    self.genome_repository.add_genome(
                        code=individual.code,
                        fitness=individual.fitness,
                        accuracy=accuracy,
                        function_name=self.config.function_name,
                        test_cases=self.config.test_cases,
                        expected_outputs=self.config.expected_outputs,
                        generation_found=individual.generation_created,
                        task_id=f"{self.config.function_name}_{self.generation}",
                        metadata={
                            'population_size': self.config.population_size,
                            'mutation_rate': self.config.mutation_rate,
                            'generation': self.generation
                        }
                    )
                    saved_count += 1

        if saved_count > 0:
            self.genome_repository.save_repository()
            self.log(f"Saved {saved_count} genomes to repository")

        return saved_count

    def log(self, message: str) -> None:
        """Add a message to the evolution log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.evolution_log.append(log_entry)
        print(log_entry)


class EvolutionRunner:
    """High-level runner for the evolution process."""

    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.engine = EvolutionEngine(self.config)

    def run_evolution(self, interactive: bool = True) -> Dict[str, Any]:
        """Run the complete evolution process."""
        self.engine.log(f"🧬 Starting Brainfuck Evolution: {self.config.function_name} 🧬")
        self.engine.log(f"Population: {self.config.population_size}, "
                       f"Mutation Rate: {self.config.mutation_rate}, "
                       f"Test Cases: {self.config.test_cases}")
        self.engine.log(f"Target Function: {self.config.function_name}")
        self.engine.log(f"Input→Output Mapping: {dict(list(self.config.input_output_mapping.items())[:6])}{'...' if len(self.config.input_output_mapping) > 6 else ''}")

        # Initialize population
        self.engine.initialize_population()

        start_time = time.time()
        perfect_found = False

        try:
            for gen in range(self.config.max_generations):
                # Evolve one generation
                result = self.engine.evolve_generation()

                # Get current stats
                stats = self.engine.get_stats()

                # Check for perfect solution
                if result == 'PERFECT':
                    perfect_found = True
                    break

                # Add delay if specified
                if self.config.generation_delay_ms > 0:
                    time.sleep(self.config.generation_delay_ms / 1000.0)

        except KeyboardInterrupt:
            self.engine.log("Evolution interrupted by user")

        # Final results
        end_time = time.time()
        duration = end_time - start_time

        final_stats = self.engine.get_stats()
        self.engine.log(f"Evolution completed in {duration:.2f} seconds")
        self.engine.log(f"Final generation: {self.engine.generation}")
        self.engine.log(f"Best fitness achieved: {self.engine.best_accuracy_ever:.2f}%")

        # Calculate and report final accuracy
        if self.engine.best_ever and self.engine.best_ever.test_results:
            correct_count = sum(1 for r in self.engine.best_ever.test_results if r.actual == r.expected)
            total_count = len(self.engine.best_ever.test_results)
            final_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            self.engine.log(f"Best accuracy: {correct_count}/{total_count} correct ({final_accuracy:.1f}%)")

        if perfect_found:
            self.engine.log("🏆 Perfect solution found!")

        # Save successful genomes to repository
        if self.engine.config.save_successful_genomes:
            saved_count = self.engine.save_successful_genomes(force_save=perfect_found)
            if saved_count > 0:
                self.engine.log(f"💾 Saved {saved_count} genomes to repository")

        return {
            'success': perfect_found,
            'generations': self.engine.generation,
            'duration': duration,
            'best_fitness': self.engine.best_accuracy_ever,
            'best_accuracy': final_stats.get('best_accuracy', 0),
            'final_stats': final_stats,
            'best_code': self.engine.best_ever.code if self.engine.best_ever else None
        }