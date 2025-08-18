#!/usr/bin/env python3
"""
Staged Fitness System for Motif Evolution

Implements the staged fitness approach:
- Stage 0 (Gating): Basic I/O contract (exactly 1 read, 1 write)
- Stage 1 (Behavioral Shaping): Behavioral properties before exactness
- Stage 2 (Correctness): Exact matching with validation

This creates dense gradients toward the right behavioral scaffolds
before rewarding exact correctness.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from brainfuck import BrainfuckInterpreter


@dataclass
class ExecutionTrace:
    """Captures execution behavior for fitness analysis."""
    inputs_read: int
    outputs_written: int
    output_values: List[int]
    step_count: int
    timeout: bool
    success: bool
    input_values: List[int]
    
    def output_dependency(self) -> float:
        """Measure how much output depends on input (not constant)."""
        if len(self.output_values) == 0:
            return 0.0
        
        # If all outputs are the same, it's likely a constant function
        output_variance = np.var(self.output_values) if len(self.output_values) > 1 else 0.0
        
        # Reward non-zero variance (outputs that depend on inputs)
        return min(1.0, output_variance / 100.0)  # Normalize to [0, 1]
    
    def finite_difference_consistency(self) -> float:
        """Measure consistency of Δf(x) = f(x+1) - f(x)."""
        if len(self.output_values) < 2 or len(self.input_values) < 2:
            return 0.0
        
        # Sort by input to compute differences
        sorted_pairs = sorted(zip(self.input_values, self.output_values))
        
        deltas = []
        for i in range(len(sorted_pairs) - 1):
            x1, y1 = sorted_pairs[i]
            x2, y2 = sorted_pairs[i + 1]
            
            # Only consider consecutive inputs
            if x2 == x1 + 1:
                delta = (y2 - y1) % 256  # Handle wraparound
                deltas.append(delta)
        
        if len(deltas) < 2:
            return 0.0
        
        # Reward low variance in deltas (consistent finite differences)
        delta_variance = np.var(deltas)
        consistency = 1.0 / (1.0 + delta_variance / 10.0)  # Higher consistency for lower variance
        
        return consistency
    
    def slope_prior(self, target_slope: float = 2.0) -> float:
        """Reward mean delta close to target slope (e.g., 2 for doubling)."""
        if len(self.output_values) < 2 or len(self.input_values) < 2:
            return 0.0
        
        # Calculate mean delta
        sorted_pairs = sorted(zip(self.input_values, self.output_values))
        deltas = []
        
        for i in range(len(sorted_pairs) - 1):
            x1, y1 = sorted_pairs[i]
            x2, y2 = sorted_pairs[i + 1]
            
            if x2 == x1 + 1:  # Consecutive inputs
                delta = (y2 - y1) % 256
                deltas.append(delta)
        
        if not deltas:
            return 0.0
        
        mean_delta = np.mean(deltas)
        
        # Smooth sigmoid around target slope
        distance = abs(mean_delta - target_slope)
        return 1.0 / (1.0 + distance)
    
    def work_correlation(self) -> float:
        """Measure correlation between input magnitude and computational work."""
        if len(self.input_values) < 2:
            return 0.0
        
        # Rough proxy for work: step count (would be better with actual + - operations)
        # For now, assume programs that handle larger inputs do more work
        input_range = max(self.input_values) - min(self.input_values)
        
        if input_range == 0:
            return 0.0
        
        # Reward programs that take more steps for larger inputs
        # This is a rough heuristic - in practice we'd track per-input work
        work_per_input_range = self.step_count / max(1, input_range)
        
        # Normalize: reasonable programs should do O(input) work for doubling
        return min(1.0, work_per_input_range / 50.0)


class StagedFitnessEvaluator:
    """Evaluates programs using staged fitness approach."""
    
    def __init__(self, target_slope: float = 2.0):
        self.interpreter = BrainfuckInterpreter()
        self.target_slope = target_slope
        
        # Stage thresholds and weights
        self.stage1_threshold = 0.3  # Must pass behavioral shaping to reach correctness
        self.behavioral_weights = {
            'dependency': 0.25,
            'consistency': 0.35, 
            'slope': 0.30,
            'work': 0.10
        }
    
    def evaluate_program(self, code: str, test_cases: Dict[int, int]) -> Dict[str, float]:
        """Evaluate a program through all fitness stages."""
        
        # Execute program on all test cases
        trace = self._execute_and_trace(code, test_cases)
        
        # Stage 0: Gating (I/O contract)
        gate_score = self._stage0_gating(trace)
        if gate_score < 0.5:  # Failed basic I/O contract
            return {
                'total_fitness': gate_score * 10,  # Very low score
                'gate_score': gate_score,
                'behavioral_score': 0.0,
                'correctness_score': 0.0,
                'stage': 0
            }
        
        # Stage 1: Behavioral shaping
        behavioral_score = self._stage1_behavioral(trace)
        
        # Stage 2: Correctness (only if behavioral is good enough)
        if behavioral_score >= self.stage1_threshold:
            correctness_score = self._stage2_correctness(trace, test_cases)
            stage = 2
        else:
            correctness_score = 0.0
            stage = 1
        
        # Combine scores with appropriate weighting
        total_fitness = self._combine_scores(gate_score, behavioral_score, correctness_score)
        
        return {
            'total_fitness': total_fitness,
            'gate_score': gate_score,
            'behavioral_score': behavioral_score,
            'correctness_score': correctness_score,
            'stage': stage,
            'trace': trace
        }
    
    def _execute_and_trace(self, code: str, test_cases: Dict[int, int]) -> ExecutionTrace:
        """Execute program and collect execution trace."""
        
        output_values = []
        input_values = list(test_cases.keys())
        total_steps = 0
        timeouts = 0
        total_reads = 0
        total_writes = 0
        
        for input_val in input_values:
            try:
                # Reset interpreter for each test case
                self.interpreter.__init__()
                
                # Execute with this input (as raw byte value, not ASCII)
                input_chr = chr(input_val)
                result = self.interpreter.run(code, input_chr)
                
                if self.interpreter.hit_step_limit:
                    timeouts += 1
                
                # Collect I/O statistics for this execution
                total_reads += self.interpreter.input_reads
                total_writes += self.interpreter.output_writes
                
                # Collect output (convert to int)
                if self.interpreter.output:
                    # Output is stored as characters, convert to ASCII values
                    output_char = self.interpreter.output[0]
                    if isinstance(output_char, str):
                        output_values.append(ord(output_char))
                    else:
                        output_values.append(int(output_char))
                else:
                    output_values.append(0)  # No output
                
                total_steps += len(code)  # Rough proxy for actual steps
                
            except Exception:
                output_values.append(0)
                timeouts += 1
        
        return ExecutionTrace(
            inputs_read=total_reads,
            outputs_written=total_writes,
            output_values=output_values,
            step_count=total_steps,
            timeout=timeouts > 0,
            success=timeouts == 0,
            input_values=input_values
        )
    
    def _stage0_gating(self, trace: ExecutionTrace) -> float:
        """Stage 0: Basic I/O contract."""
        
        # Must read exactly 1 input and write exactly 1 output
        correct_reads = (trace.inputs_read == len(trace.input_values))
        correct_writes = (trace.outputs_written == len(trace.input_values))
        no_timeout = not trace.timeout
        
        # All conditions must be met
        if correct_reads and correct_writes and no_timeout:
            return 1.0
        else:
            return 0.0  # Hard gate - either pass or fail
    
    def _stage1_behavioral(self, trace: ExecutionTrace) -> float:
        """Stage 1: Behavioral shaping before exactness."""
        
        components = {
            'dependency': trace.output_dependency(),
            'consistency': trace.finite_difference_consistency(),
            'slope': trace.slope_prior(self.target_slope),
            'work': trace.work_correlation()
        }
        
        # Weighted combination of behavioral components
        behavioral_score = sum(
            components[comp] * self.behavioral_weights[comp]
            for comp in components
        )
        
        return behavioral_score
    
    def _stage2_correctness(self, trace: ExecutionTrace, test_cases: Dict[int, int]) -> float:
        """Stage 2: Exact correctness."""
        
        if len(trace.output_values) != len(test_cases):
            return 0.0
        
        correct_count = 0
        total_count = len(test_cases)
        
        for i, (input_val, expected_output) in enumerate(test_cases.items()):
            if i < len(trace.output_values):
                actual_output = trace.output_values[i]
                if actual_output == expected_output:
                    correct_count += 1
        
        return correct_count / total_count if total_count > 0 else 0.0
    
    def _combine_scores(self, gate: float, behavioral: float, correctness: float) -> float:
        """Combine scores across stages."""
        
        if gate < 0.5:
            # Failed gating - very low score
            return gate * 10
        
        if correctness > 0:
            # Stage 2: Correctness matters most
            return 50 + (correctness * 50)  # 50-100 range
        else:
            # Stage 1: Behavioral shaping
            return 10 + (behavioral * 40)  # 10-50 range


def test_staged_fitness():
    """Test the staged fitness system on various programs."""
    
    print("🎯 TESTING STAGED FITNESS SYSTEM")
    print("=" * 50)
    
    evaluator = StagedFitnessEvaluator(target_slope=2.0)  # For doubling
    
    # Test programs with different properties
    test_programs = [
        (",.", "Identity - should pass gating but not behavioral"),
        (",+.", "Increment - good behavioral shaping"),
        (",++.", "Add 2 - perfect doubling behavior"),
        (",..", "Invalid - multiple outputs"),
        (",[-].", "Clear - passes gating, poor behavioral"),
        (",>++++++++[<++>-]<.", "Proper doubling via loop - should get correctness"),
        (",>+<.", "Constant output - poor dependency")
    ]
    
    # Test cases for doubling function
    test_cases = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
    
    for program, description in test_programs:
        print(f"\n🧬 Testing: {program}")
        print(f"   Expected: {description}")
        
        try:
            results = evaluator.evaluate_program(program, test_cases)
            
            print(f"   📊 Results:")
            print(f"      Total Fitness: {results['total_fitness']:.2f}")
            print(f"      Gate Score: {results['gate_score']:.2f}")
            print(f"      Behavioral Score: {results['behavioral_score']:.2f}")
            print(f"      Correctness Score: {results['correctness_score']:.2f}")
            print(f"      Stage Reached: {results['stage']}")
            
            if 'trace' in results:
                trace = results['trace']
                print(f"      I/O: {trace.inputs_read} reads, {trace.outputs_written} writes")
                print(f"      Outputs: {trace.output_values[:5]}{'...' if len(trace.output_values) > 5 else ''}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n💡 The staged fitness system creates dense gradients:")
    print(f"   Stage 0 (0-10): Basic I/O contract")
    print(f"   Stage 1 (10-50): Behavioral shaping (dependency, consistency, slope)")
    print(f"   Stage 2 (50-100): Exact correctness")


if __name__ == "__main__":
    test_staged_fitness()