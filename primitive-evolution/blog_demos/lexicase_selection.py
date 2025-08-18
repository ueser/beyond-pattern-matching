#!/usr/bin/env python3
"""
Lexicase Selection for Motif Evolution

Implements lexicase selection to protect partial solutions that excel on specific
test cases. This is crucial for motif discovery because it preserves specialists
that are "almost there" - e.g., a program that perfectly doubles f(3)=6 but 
fails on other inputs shouldn't be discarded.

Key advantages over tournament selection:
- Protects diversity by preserving specialists
- Allows gradual improvement on different test cases
- Prevents premature convergence to "good enough" generalists
- Essential for discovering complex motifs that emerge gradually

Also includes ECO-Lexicase variant for computational efficiency.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Individual:
    """Individual with fitness breakdown per test case."""
    code: str
    total_fitness: float
    test_case_fitnesses: Dict[int, float]  # test_case_id -> fitness
    test_case_accuracies: Dict[int, bool]  # test_case_id -> exact_match
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LexicaseSelector:
    """Implements lexicase and ECO-lexicase selection."""
    
    def __init__(self, epsilon_threshold: float = 0.0, use_eco: bool = True):
        """
        Initialize lexicase selector.
        
        Args:
            epsilon_threshold: For epsilon-lexicase, treat differences < epsilon as equal
            use_eco: Use ECO-lexicase for computational efficiency
        """
        self.epsilon_threshold = epsilon_threshold
        self.use_eco = use_eco
    
    def select_parent(self, population: List[Individual], test_case_ids: List[int]) -> Individual:
        """
        Select a parent using lexicase selection.
        
        Args:
            population: List of individuals to select from
            test_case_ids: Test case IDs to use for selection
        
        Returns:
            Selected individual
        """
        if self.use_eco:
            return self._eco_lexicase_select(population, test_case_ids)
        else:
            return self._standard_lexicase_select(population, test_case_ids)
    
    def _standard_lexicase_select(self, population: List[Individual], test_case_ids: List[int]) -> Individual:
        """Standard lexicase selection algorithm."""
        if not population:
            raise ValueError("Empty population")
        
        if len(population) == 1:
            return population[0]
        
        # Start with entire population
        candidates = population.copy()
        
        # Randomly shuffle test case order (key to lexicase)
        shuffled_cases = test_case_ids.copy()
        random.shuffle(shuffled_cases)
        
        # Filter candidates by each test case in sequence
        for test_case_id in shuffled_cases:
            if len(candidates) <= 1:
                break
            
            # Get fitness values for this test case
            fitnesses = []
            for individual in candidates:
                fitness = individual.test_case_fitnesses.get(test_case_id, 0.0)
                fitnesses.append(fitness)
            
            if not fitnesses:
                continue
            
            # Find best fitness for this test case
            best_fitness = max(fitnesses)
            
            # Keep only individuals within epsilon of best
            new_candidates = []
            for individual, fitness in zip(candidates, fitnesses):
                if abs(fitness - best_fitness) <= self.epsilon_threshold:
                    new_candidates.append(individual)
            
            candidates = new_candidates
        
        # Randomly select from remaining candidates
        if candidates:
            return random.choice(candidates)
        else:
            # Fallback - shouldn't happen
            return random.choice(population)
    
    def _eco_lexicase_select(self, population: List[Individual], test_case_ids: List[int]) -> Individual:
        """
        ECO-Lexicase: Computationally efficient variant that stops early
        when diversity is sufficient.
        """
        if not population:
            raise ValueError("Empty population")
        
        if len(population) == 1:
            return population[0]
        
        # Start with entire population
        candidates = population.copy()
        
        # Randomly shuffle test case order
        shuffled_cases = test_case_ids.copy()
        random.shuffle(shuffled_cases)
        
        # Target diversity level (stop when we reach this many candidates)
        target_diversity = max(2, len(population) // 10)  # Keep at least 2, at most 10%
        
        # Filter candidates by each test case in sequence
        for test_case_id in shuffled_cases:
            if len(candidates) <= target_diversity:
                break  # ECO: Stop early when diversity is sufficient
            
            # Get fitness values for this test case
            fitnesses = []
            for individual in candidates:
                fitness = individual.test_case_fitnesses.get(test_case_id, 0.0)
                fitnesses.append(fitness)
            
            if not fitnesses:
                continue
            
            # Find best fitness for this test case
            best_fitness = max(fitnesses)
            
            # Keep only individuals within epsilon of best
            new_candidates = []
            for individual, fitness in zip(candidates, fitnesses):
                if abs(fitness - best_fitness) <= self.epsilon_threshold:
                    new_candidates.append(individual)
            
            # ECO: Only update if we still have reasonable diversity
            if len(new_candidates) >= 1:
                candidates = new_candidates
            else:
                break  # Stop filtering to preserve some diversity
        
        # Randomly select from remaining candidates
        return random.choice(candidates) if candidates else random.choice(population)
    
    def select_parents_batch(self, population: List[Individual], test_case_ids: List[int], 
                           num_parents: int) -> List[Individual]:
        """Select multiple parents efficiently."""
        return [self.select_parent(population, test_case_ids) for _ in range(num_parents)]


class LexicaseAnalyzer:
    """Analyzes lexicase selection behavior and specialist preservation."""
    
    @staticmethod
    def analyze_specialists(population: List[Individual], test_case_ids: List[int]) -> Dict[str, Any]:
        """Analyze specialist diversity in the population."""
        if not population or not test_case_ids:
            return {}
        
        analysis = {
            'total_individuals': len(population),
            'test_cases': len(test_case_ids),
            'specialists_per_case': {},
            'generalists': 0,
            'diversity_metrics': {}
        }
        
        # Find specialists for each test case
        for test_case_id in test_case_ids:
            # Get all fitnesses for this test case
            case_fitnesses = []
            for individual in population:
                fitness = individual.test_case_fitnesses.get(test_case_id, 0.0)
                case_fitnesses.append((individual, fitness))
            
            # Sort by fitness for this case
            case_fitnesses.sort(key=lambda x: x[1], reverse=True)
            
            # Count how many individuals are "specialists" (top performers) on this case
            if case_fitnesses:
                best_fitness = case_fitnesses[0][1]
                specialists = [ind for ind, fit in case_fitnesses if fit >= best_fitness * 0.9]
                analysis['specialists_per_case'][test_case_id] = len(specialists)
        
        # Count generalists (individuals that perform well across many cases)
        generalist_threshold = 0.8  # Must be good on 80% of cases
        for individual in population:
            good_cases = 0
            for test_case_id in test_case_ids:
                fitness = individual.test_case_fitnesses.get(test_case_id, 0.0)
                if fitness > 50:  # Arbitrary "good" threshold
                    good_cases += 1
            
            if good_cases >= len(test_case_ids) * generalist_threshold:
                analysis['generalists'] += 1
        
        # Diversity metrics
        total_unique_specialists = len(set(
            individual.code for individual in population
            if any(individual.test_case_fitnesses.get(tid, 0) > 70 for tid in test_case_ids)
        ))
        
        analysis['diversity_metrics'] = {
            'unique_specialists': total_unique_specialists,
            'specialist_ratio': total_unique_specialists / len(population) if population else 0,
            'avg_specialists_per_case': np.mean(list(analysis['specialists_per_case'].values())) if analysis['specialists_per_case'] else 0
        }
        
        return analysis


def test_lexicase_selection():
    """Test lexicase selection vs tournament selection on specialist preservation."""
    print("🎯 TESTING LEXICASE SELECTION")
    print("=" * 50)
    
    # Create a test population with known specialists
    test_cases = [1, 2, 3, 4, 5]
    
    # Create individuals with different specialization patterns
    population = []
    
    # Generalist (decent on all cases)
    population.append(Individual(
        code=",+.",
        total_fitness=60.0,
        test_case_fitnesses={1: 70, 2: 60, 3: 50, 4: 65, 5: 55},
        test_case_accuracies={1: True, 2: False, 3: False, 4: True, 5: False}
    ))
    
    # Specialist 1 (perfect on case 1, poor elsewhere)
    population.append(Individual(
        code=",.",
        total_fitness=40.0,
        test_case_fitnesses={1: 100, 2: 10, 3: 15, 4: 20, 5: 5},
        test_case_accuracies={1: True, 2: False, 3: False, 4: False, 5: False}
    ))
    
    # Specialist 2 (perfect on case 3, poor elsewhere)
    population.append(Individual(
        code=",++.",
        total_fitness=45.0,
        test_case_fitnesses={1: 20, 2: 25, 3: 100, 4: 15, 5: 10},
        test_case_accuracies={1: False, 2: False, 3: True, 4: False, 5: False}
    ))
    
    # Specialist 3 (good on cases 4,5, poor elsewhere)
    population.append(Individual(
        code=",+++.",
        total_fitness=50.0,
        test_case_fitnesses={1: 15, 2: 20, 3: 25, 4: 85, 5: 90},
        test_case_accuracies={1: False, 2: False, 3: False, 4: True, 5: True}
    ))
    
    # Poor performer (low on all)
    population.append(Individual(
        code=",[-].",
        total_fitness=25.0,
        test_case_fitnesses={1: 30, 2: 25, 3: 20, 4: 25, 5: 25},
        test_case_accuracies={1: False, 2: False, 3: False, 4: False, 5: False}
    ))
    
    print(f"Population of {len(population)} individuals:")
    for i, ind in enumerate(population):
        print(f"  {i+1}. '{ind.code}' - Total: {ind.total_fitness:.1f}, Per-case: {ind.test_case_fitnesses}")
    
    # Test lexicase selection
    print(f"\n🔄 LEXICASE SELECTION TEST")
    print("-" * 30)
    
    selector = LexicaseSelector(epsilon_threshold=5.0, use_eco=True)
    
    # Track selection frequencies
    selection_counts = defaultdict(int)
    num_selections = 1000
    
    for _ in range(num_selections):
        selected = selector.select_parent(population, test_cases)
        selection_counts[selected.code] += 1
    
    print(f"Selection frequencies over {num_selections} selections:")
    for code, count in selection_counts.items():
        percentage = (count / num_selections) * 100
        print(f"  '{code}': {count} ({percentage:.1f}%)")
    
    # Analyze specialist preservation
    print(f"\n📊 SPECIALIST ANALYSIS")
    print("-" * 25)
    
    analyzer = LexicaseAnalyzer()
    analysis = analyzer.analyze_specialists(population, test_cases)
    
    print(f"Population size: {analysis['total_individuals']}")
    print(f"Test cases: {analysis['test_cases']}")
    print(f"Generalists: {analysis['generalists']}")
    print(f"Specialists per test case:")
    for case_id, specialist_count in analysis['specialists_per_case'].items():
        print(f"  Case {case_id}: {specialist_count} specialists")
    
    diversity = analysis['diversity_metrics']
    print(f"Diversity metrics:")
    print(f"  Unique specialists: {diversity['unique_specialists']}")
    print(f"  Specialist ratio: {diversity['specialist_ratio']:.2f}")
    print(f"  Avg specialists per case: {diversity['avg_specialists_per_case']:.1f}")
    
    # Compare with tournament selection
    print(f"\n⚔️ TOURNAMENT SELECTION COMPARISON")
    print("-" * 35)
    
    tournament_counts = defaultdict(int)
    for _ in range(num_selections):
        # Simple tournament selection (k=3)
        tournament = random.sample(population, min(3, len(population)))
        winner = max(tournament, key=lambda x: x.total_fitness)
        tournament_counts[winner.code] += 1
    
    print(f"Tournament selection frequencies:")
    for code, count in tournament_counts.items():
        percentage = (count / num_selections) * 100
        print(f"  '{code}': {count} ({percentage:.1f}%)")
    
    print(f"\n💡 Key Observations:")
    print(f"• Lexicase should preserve specialists even if their total fitness is lower")
    print(f"• Tournament selection biases toward high total fitness (generalists)")  
    print(f"• Specialists like ',.' (perfect on case 1) should survive in lexicase")
    print(f"• This diversity is crucial for gradual motif discovery")


def test_eco_lexicase_efficiency():
    """Test ECO-lexicase computational efficiency."""
    print(f"\n⚡ TESTING ECO-LEXICASE EFFICIENCY")
    print("=" * 40)
    
    # Create larger population to test efficiency
    large_population = []
    for i in range(100):
        # Random fitness patterns
        fitnesses = {j: random.uniform(0, 100) for j in range(10)}
        accuracies = {j: random.choice([True, False]) for j in range(10)}
        
        large_population.append(Individual(
            code=f"program_{i}",
            total_fitness=sum(fitnesses.values()) / len(fitnesses),
            test_case_fitnesses=fitnesses,
            test_case_accuracies=accuracies
        ))
    
    test_cases = list(range(10))
    
    # Test both variants
    standard_selector = LexicaseSelector(use_eco=False)
    eco_selector = LexicaseSelector(use_eco=True)
    
    import time
    
    # Time standard lexicase
    start_time = time.time()
    for _ in range(100):
        standard_selector.select_parent(large_population, test_cases)
    standard_time = time.time() - start_time
    
    # Time ECO lexicase  
    start_time = time.time()
    for _ in range(100):
        eco_selector.select_parent(large_population, test_cases)
    eco_time = time.time() - start_time
    
    print(f"Performance comparison (100 selections, 100 individuals, 10 test cases):")
    print(f"  Standard lexicase: {standard_time:.3f}s")
    print(f"  ECO lexicase: {eco_time:.3f}s")
    print(f"  Speedup: {standard_time/eco_time:.1f}x")


if __name__ == "__main__":
    test_lexicase_selection()
    test_eco_lexicase_efficiency()