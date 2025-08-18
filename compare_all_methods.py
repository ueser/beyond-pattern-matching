#!/usr/bin/env python3
"""
Comprehensive comparison of all Brainfuck program synthesis methods:
- Genetic Algorithm (Evolution)  
- Reinforcement Learning (PPO)
- Hill Climbing (Local Search)
- Simulated Annealing (Local Search)
"""

import sys
sys.path.append('primitive-evolution/blog_demos')

import time
from typing import Dict, Any

# Import local search methods (working and tested)
from bf_local_search import SearchConfig, HillClimbing, SimulatedAnnealing
from bf_memetic_algorithm import MemeticAlgorithm, MemeticConfig

# Optional imports for future comparison
try:
    from brainfuck_evolution import EvolutionConfig, EvolutionRunner
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False

try:
    import torch
    from bf_rl_trainer import BFPPOTrainer, TrainingConfig
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

def test_genetic_algorithm() -> Dict[str, Any]:
    """Test the genetic algorithm approach."""
    if not GA_AVAILABLE:
        return {
            'method': 'Genetic Algorithm',
            'program': '',
            'fitness': 0.0,
            'time': 0.0,
            'converged': False,
            'error': 'GA module not available'
        }
    
    print("🧬 GENETIC ALGORITHM")
    print("=" * 40)
    
    # Create function mapping for doubling
    input_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    input_output_mapping = {inp: inp * 2 for inp in input_pool}
    
    config = EvolutionConfig(
        population_size=50,  # Smaller for quick test
        max_generations=30,
        max_program_length=24,
        function_name="double",
        use_dynamic_sampling=True,
        samples_per_generation=5,
        input_pool_mapping=input_output_mapping
    )
    
    start_time = time.time()
    runner = EvolutionRunner(config)
    result = runner.run_evolution(interactive=False)
    ga_time = time.time() - start_time
    
    return {
        'method': 'Genetic Algorithm',
        'program': result.get('best_code', ''),
        'fitness': result.get('best_accuracy', 0) / 100.0,  # Convert percentage to 0-1
        'time': ga_time,
        'converged': result.get('success', False),
        'stats': result
    }

def test_reinforcement_learning() -> Dict[str, Any]:
    """Test the reinforcement learning approach."""
    if not RL_AVAILABLE:
        return {
            'method': 'Reinforcement Learning',
            'program': '',
            'fitness': 0.0,
            'time': 0.0,
            'converged': False,
            'error': 'RL module not available (torch/RL deps)'
        }
    
    print("🧠 REINFORCEMENT LEARNING (PPO)")
    print("=" * 40)
    
    config = TrainingConfig(
        # Environment
        max_length=24,
        eval_inputs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        target_function="2*x",
        
        # Dynamic sampling
        input_pool=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        n_sample_inputs=5,  # Sample 5 inputs each update
        
        # Training - reduced for quick test
        n_episodes_per_update=16,
        n_ppo_epochs=3,
        batch_size=8,
        learning_rate=5e-3,
        entropy_coef=0.3,  # Higher exploration
        
        # Model - smaller for speed
        d_model=64,
        n_layers=2,
        architecture="gru",
        
        # Curriculum - quick phases
        curriculum_phases=[
            {"max_length": 16, "updates": 20},
            {"max_length": 24, "updates": 30}
        ],
        
        log_interval=10,
        use_wandb=False
    )
    
    start_time = time.time()
    trainer = BFPPOTrainer(config)
    trainer.train()  # Use curriculum
    rl_time = time.time() - start_time
    
    return {
        'method': 'Reinforcement Learning',
        'program': trainer.best_program,
        'fitness': trainer.best_reward,
        'time': rl_time,
        'converged': trainer.best_reward > 0.9,
        'stats': {'updates': trainer.update_count, 'episodes': trainer.episode_count}
    }

def test_memetic_algorithm() -> Dict[str, Any]:
    """Test the memetic algorithm (GA + Hill Climbing)."""
    print("🧬🔍 MEMETIC ALGORITHM (GA + HC)")
    print("=" * 40)
    
    config = MemeticConfig(
        target_function="2*x",
        population_size=25,  # Smaller for comparison speed
        generations=30,
        local_search_intensity=50,
        local_search_frequency=3,
        adaptive_frequency=True,
        probe_inputs=[1, 2, 3, 4, 5]  # Quick evaluation
    )
    
    start_time = time.time()
    memetic = MemeticAlgorithm(config)
    result = memetic.run()
    memetic_time = time.time() - start_time
    
    return {
        'method': 'Memetic Algorithm',
        'program': result['program'],
        'fitness': result['fitness'],
        'time': memetic_time,
        'converged': result['converged'],
        'stats': result['stats']
    }

def test_hill_climbing() -> Dict[str, Any]:
    """Test the hill climbing approach."""
    print("🏔️ HILL CLIMBING")
    print("=" * 40)
    
    config = SearchConfig(
        target_function="2*x",
        max_length=24,
        hc_restarts=200,  # More restarts needed without seeds
        hc_iterations=500,
        probe_inputs=[1, 2, 3, 4, 5]  # Quick evaluation
    )
    
    start_time = time.time()
    hc = HillClimbing(config)
    program, fitness, stats = hc.search()
    hc_time = time.time() - start_time
    
    return {
        'method': 'Hill Climbing',
        'program': program,
        'fitness': fitness,
        'time': hc_time,
        'converged': fitness > 0.9,
        'stats': stats
    }

def test_simulated_annealing() -> Dict[str, Any]:
    """Test the simulated annealing approach."""
    print("🌡️ SIMULATED ANNEALING")
    print("=" * 40)
    
    config = SearchConfig(
        target_function="2*x",
        max_length=24,
        sa_epochs=200,        # More epochs without seeds
        sa_steps_per_epoch=150,
        sa_initial_temp=0.5,  # Higher temperature for more exploration
        probe_inputs=[1, 2, 3, 4, 5]  # Quick evaluation
    )
    
    start_time = time.time()
    sa = SimulatedAnnealing(config)
    program, fitness, stats = sa.search()
    sa_time = time.time() - start_time
    
    return {
        'method': 'Simulated Annealing',
        'program': program,
        'fitness': fitness,
        'time': sa_time,
        'converged': fitness > 0.9,
        'stats': stats
    }

def validate_solution(program: str, target_function: str = "2*x") -> Dict[str, Any]:
    """Validate a program solution on all test cases."""
    if not program:
        return {'exact_matches': 0, 'total_tests': 0, 'accuracy': 0.0, 'errors': []}
    
    from brainfuck import BrainfuckInterpreter
    test_inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    exact_matches = 0
    errors = []
    
    for test_input in test_inputs:
        try:
            interpreter = BrainfuckInterpreter()
            result = interpreter.run(program, chr(test_input))
            actual = ord(result[0]) if result else 0
            
            if target_function == "2*x":
                expected = test_input * 2
            elif target_function == "x+1":
                expected = test_input + 1
            else:
                expected = test_input
                
            if actual == expected:
                exact_matches += 1
            else:
                errors.append(f"f({test_input})={actual}, expected {expected}")
                
        except Exception as e:
            errors.append(f"f({test_input})=ERROR: {e}")
    
    return {
        'exact_matches': exact_matches,
        'total_tests': len(test_inputs),
        'accuracy': exact_matches / len(test_inputs),
        'errors': errors[:3]  # Show first 3 errors
    }

def main():
    """Compare all methods on the Brainfuck doubling problem."""
    print("🚀 BRAINFUCK PROGRAM SYNTHESIS COMPARISON")
    print("=" * 70)
    print("Target: f(x) = 2*x (doubling function)")
    print("Goal: Learn Brainfuck programs that double their input")
    print()
    
    # Test all available methods
    methods_to_test = [
        ("Hill Climbing", test_hill_climbing),
        ("Simulated Annealing", test_simulated_annealing),
        ("Memetic Algorithm", test_memetic_algorithm),
    ]
    
    # Add optional methods if available
    if GA_AVAILABLE:
        methods_to_test.append(("Genetic Algorithm", test_genetic_algorithm))
    
    if RL_AVAILABLE:
        methods_to_test.append(("Reinforcement Learning", test_reinforcement_learning))
    
    results = []
    
    for method_name, test_func in methods_to_test:
        try:
            print(f"\n{'='*20} {method_name.upper()} {'='*20}")
            result = test_func()
            
            # Validate the solution
            validation = validate_solution(result['program'], "2*x")
            result['validation'] = validation
            
            results.append(result)
            
            print(f"✅ {method_name} completed:")
            print(f"   Program: {result['program']}")
            print(f"   Fitness: {result['fitness']:.3f}")
            print(f"   Time: {result['time']:.1f}s")
            print(f"   Accuracy: {validation['accuracy']:.1%} ({validation['exact_matches']}/{validation['total_tests']})")
            
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            results.append({
                'method': method_name,
                'program': '',
                'fitness': float('-inf'),
                'time': 0,
                'converged': False,
                'error': str(e)
            })
    
    # Final comparison
    print(f"\n{'='*25} FINAL COMPARISON {'='*25}")
    print("Method               | Program           | Fitness | Time   | Accuracy")
    print("-" * 75)
    
    # Sort by fitness (best first)
    results.sort(key=lambda x: x.get('fitness', float('-inf')), reverse=True)
    
    for result in results:
        method = result['method']
        program = result['program'][:15] + "..." if len(result['program']) > 15 else result['program']
        fitness = result.get('fitness', 0)
        time_taken = result.get('time', 0)
        accuracy = result.get('validation', {}).get('accuracy', 0)
        
        print(f"{method:<20} | {program:<17} | {fitness:7.3f} | {time_taken:6.1f}s | {accuracy:7.1%}")
    
    # Find the best solution
    if results and results[0].get('validation', {}).get('accuracy', 0) >= 1.0:
        best = results[0]
        print(f"\n🎉 PERFECT SOLUTION FOUND!")
        print(f"Method: {best['method']}")
        print(f"Program: {best['program']}")
        print(f"Length: {len(best['program'])} characters")
        print(f"Time: {best['time']:.1f} seconds")
        print(f"All {best['validation']['total_tests']} test cases passed!")
    elif results:
        best = results[0]
        print(f"\n🥇 BEST SOLUTION:")
        print(f"Method: {best['method']}")
        print(f"Program: {best['program']}")
        print(f"Accuracy: {best.get('validation', {}).get('accuracy', 0):.1%}")
        
        if best.get('validation', {}).get('errors'):
            print(f"Sample errors: {best['validation']['errors']}")
    
    print(f"\n🏁 Comparison completed!")

if __name__ == "__main__":
    main()