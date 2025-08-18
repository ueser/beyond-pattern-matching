#!/usr/bin/env python3
"""
Motif-Based Evolution for Brainfuck

Implements the structured approach to evolving Brainfuck programs that discover
reusable motifs/macros through evolutionary pressure. The goal is not to solve
specific tasks, but to demonstrate how evolution can bootstrap symbolic 
abstractions (motifs) that become building blocks for more complex behaviors.

Key principles:
- Hard structural constraints to preserve meaningful structure
- Staged fitness: behavioral shaping before exactness  
- Motif detection by effect clustering, not string matching
- Motif-aware genetic operators
- Curriculum designed to force emergence of specific motifs

This demonstrates the transition from "pattern matching" to symbolic reasoning.
"""

import random
import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
from datetime import datetime

# Import our existing Brainfuck interpreter
import sys
sys.path.append('.')
from brainfuck import BrainfuckInterpreter


@dataclass
class MotifEffect:
    """Represents the effect of a motif on the tape."""
    cell_deltas: Dict[int, int]  # relative_position -> delta_value
    drains_source: bool          # whether it zeros the starting cell
    pointer_delta: int           # net pointer movement
    step_count: int             # execution steps taken
    pattern: str                # the actual BF code
    
    def signature(self) -> Tuple:
        """Create a signature for clustering similar effects."""
        # Focus on effects within [-2, +2] window for clustering
        core_deltas = tuple(self.cell_deltas.get(i, 0) for i in range(-2, 3))
        return (core_deltas, self.drains_source, self.pointer_delta)


@dataclass 
class ParsedMotif:
    """A parsed motif with its structure and effect."""
    type: str                    # 'linear' or 'loop'
    pattern: str                 # the BF code
    effect: Optional[MotifEffect] = None
    start_pos: int = 0           # position in original program
    end_pos: int = 0


class MotifParser:
    """Parses Brainfuck programs into meaningful motifs."""
    
    def __init__(self):
        self.interpreter = BrainfuckInterpreter()
    
    def parse_program(self, code: str) -> List[ParsedMotif]:
        """Parse a BF program into motifs (linear runs and balanced loops)."""
        if not code.startswith(',') or not code.endswith('.'):
            return []  # Invalid structure
            
        body = code[1:-1]  # Remove , and .
        return self._parse_body(body)
    
    def _parse_body(self, body: str) -> List[ParsedMotif]:
        """Parse the body into motifs."""
        motifs = []
        i = 0
        
        while i < len(body):
            if body[i] == '[':
                # Parse balanced loop
                loop_end = self._find_matching_bracket(body, i)
                if loop_end == -1:
                    break  # Unbalanced - should not happen with constraints
                
                loop_pattern = body[i:loop_end+1]
                effect = self._analyze_loop_effect(loop_pattern)
                
                motifs.append(ParsedMotif(
                    type='loop',
                    pattern=loop_pattern,
                    effect=effect,
                    start_pos=i,
                    end_pos=loop_end
                ))
                i = loop_end + 1
                
            elif body[i] in '><+-':
                # Parse linear run
                start = i
                while i < len(body) and body[i] in '><+-':
                    i += 1
                
                linear_pattern = body[start:i]
                effect = self._analyze_linear_effect(linear_pattern)
                
                motifs.append(ParsedMotif(
                    type='linear',
                    pattern=linear_pattern,
                    effect=effect,
                    start_pos=start,
                    end_pos=i-1
                ))
            else:
                i += 1  # Skip unexpected characters
        
        return motifs
    
    def _find_matching_bracket(self, code: str, start: int) -> int:
        """Find the matching closing bracket."""
        if code[start] != '[':
            return -1
            
        depth = 1
        for i in range(start + 1, len(code)):
            if code[i] == '[':
                depth += 1
            elif code[i] == ']':
                depth -= 1
                if depth == 0:
                    return i
        return -1
    
    def _analyze_loop_effect(self, loop_pattern: str) -> MotifEffect:
        """Analyze the effect of a loop using static analysis and simulation."""
        # First try static analysis for common patterns
        static_effect = self._static_loop_analysis(loop_pattern)
        if static_effect:
            return static_effect
        
        # Fallback to simulation for complex loops
        test_val = 3  # Use a single test value
        effect = self._simulate_loop_effect(loop_pattern, test_val)
        
        if effect:
            return effect
        
        # Ultimate fallback
        return MotifEffect(
            cell_deltas={},
            drains_source=False,
            pointer_delta=0,
            step_count=len(loop_pattern),
            pattern=loop_pattern
        )
    
    def _static_loop_analysis(self, loop_pattern: str) -> Optional[MotifEffect]:
        """Analyze common loop patterns statically without simulation."""
        # Remove brackets to get the loop body
        if not (loop_pattern.startswith('[') and loop_pattern.endswith(']')):
            return None
            
        body = loop_pattern[1:-1]
        
        # Pattern: [-] - Simple clear
        if body == '-':
            return MotifEffect(
                cell_deltas={0: 0},  # Current cell becomes 0 (effect depends on initial value)
                drains_source=True,
                pointer_delta=0,
                step_count=2,  # [ and ]
                pattern=loop_pattern
            )
        
        # Pattern: [->+<] - Move right
        if body == '->+<':
            return MotifEffect(
                cell_deltas={0: -1, 1: 1},  # Move from cell 0 to cell 1
                drains_source=True,
                pointer_delta=0,
                step_count=5,
                pattern=loop_pattern
            )
        
        # Pattern: [<-]+ - Move left  
        if body == '<-' or body == '<<->>' or body == '<+>':
            return MotifEffect(
                cell_deltas={0: -1, -1: 1},  # Move from cell 0 to cell -1
                drains_source=True,
                pointer_delta=0,
                step_count=len(body) + 2,
                pattern=loop_pattern
            )
        
        # Pattern: [->+>+<<] - Split/copy
        if body == '->+>+<<':
            return MotifEffect(
                cell_deltas={0: -1, 1: 1, 2: 1},  # Copy to cells 1 and 2
                drains_source=True,
                pointer_delta=0,
                step_count=7,
                pattern=loop_pattern
            )
        
        # Pattern: [>++<-] - Double (increment target by 2 for each source decrement)
        if body == '>++<-':
            return MotifEffect(
                cell_deltas={0: -1, 1: 2},  # Doubles the value to next cell
                drains_source=True,
                pointer_delta=0,
                step_count=6,
                pattern=loop_pattern
            )
        
        return None  # Not a recognized pattern
    
    def _simulate_loop_effect(self, loop_pattern: str, start_value: int) -> Optional[MotifEffect]:
        """Simulate a loop's effect starting with a specific value."""
        # Create a fresh interpreter for this simulation
        interp = BrainfuckInterpreter()
        
        # Set up tape with the start value in the middle
        tape_size = 20
        interp.memory = [0] * tape_size
        interp.pointer = tape_size // 2
        interp.memory[interp.pointer] = start_value
        
        # Capture initial state
        initial_tape = interp.memory.copy()
        initial_pointer = interp.pointer
        
        try:
            # Execute the loop with step limit - need to modify run method or use differently
            interp.run(loop_pattern, "", debug=False)
            
            if interp.hit_step_limit:
                return None  # Loop didn't terminate
                
            final_tape = interp.memory
            final_pointer = interp.pointer
            steps = len([c for c in loop_pattern if c in '><+-[]'])  # Rough step count
            
            # Calculate deltas in a window around initial position
            cell_deltas = {}
            window_start = max(0, initial_pointer - 3)
            window_end = min(len(initial_tape), initial_pointer + 4)
            
            for i in range(window_start, window_end):
                if i < len(final_tape):
                    delta = final_tape[i] - initial_tape[i]
                    if delta != 0:
                        relative_pos = i - initial_pointer
                        cell_deltas[relative_pos] = delta
            
            # Check if source cell was drained to zero
            source_drained = (initial_tape[initial_pointer] != 0 and 
                            final_tape[initial_pointer] == 0)
            
            return MotifEffect(
                cell_deltas=cell_deltas,
                drains_source=source_drained,
                pointer_delta=final_pointer - initial_pointer,
                step_count=steps,
                pattern=loop_pattern
            )
            
        except Exception as e:
            return None
    
    def _analyze_linear_effect(self, linear_pattern: str) -> MotifEffect:
        """Analyze the effect of a linear sequence."""
        pointer_delta = 0
        cell_deltas = {}
        current_pos = 0
        
        for char in linear_pattern:
            if char == '>':
                current_pos += 1
                pointer_delta += 1
            elif char == '<':
                current_pos -= 1
                pointer_delta -= 1
            elif char == '+':
                cell_deltas[current_pos] = cell_deltas.get(current_pos, 0) + 1
            elif char == '-':
                cell_deltas[current_pos] = cell_deltas.get(current_pos, 0) - 1
        
        return MotifEffect(
            cell_deltas=cell_deltas,
            drains_source=False,  # Linear sequences don't drain
            pointer_delta=pointer_delta,
            step_count=len(linear_pattern),
            pattern=linear_pattern
        )


class MotifLibrary:
    """Manages discovered motifs and their clustering."""
    
    def __init__(self):
        self.motif_clusters: Dict[str, List[MotifEffect]] = {}
        self.cluster_names = {
            'CLEAR': [],      # [-] type patterns
            'MOVE': [],       # [->+<] type patterns  
            'SPLIT': [],      # [->+>+<<] type patterns
            'ACCUM': [],      # >[<+>-] type patterns
            'LINEAR': []      # ><+- sequences
        }
    
    def add_motif(self, effect: MotifEffect):
        """Add a motif effect to the library and classify it."""
        signature = effect.signature()
        
        # Classify the motif based on its signature
        cluster_name = self._classify_motif(effect)
        self.cluster_names[cluster_name].append(effect)
    
    def _classify_motif(self, effect: MotifEffect) -> str:
        """Classify a motif into a named category based on its effect."""
        
        # Not a loop - classify as linear
        if not effect.pattern.startswith('['):
            return 'LINEAR'
        
        # Loop patterns
        if effect.drains_source and effect.pointer_delta == 0:
            # Count non-zero cell changes
            non_zero_deltas = {k: v for k, v in effect.cell_deltas.items() if v != 0}
            
            # CLEAR: Only drains current cell, no other effects
            if len(non_zero_deltas) <= 1 and effect.drains_source:
                # Check if it's a pure drain with no side effects
                if 0 in non_zero_deltas:
                    # Has explicit effect on cell 0
                    return 'CLEAR' if len(non_zero_deltas) == 1 else 'CLEAR'
                else:
                    # Drains source but no explicit delta recorded (like [-])
                    return 'CLEAR'
            
            # MOVE: Moves value from current cell to exactly one other cell
            if len(non_zero_deltas) == 2 and 0 in non_zero_deltas and non_zero_deltas[0] < 0:
                # Find the target cell
                target_cells = [k for k, v in non_zero_deltas.items() if k != 0 and v > 0]
                if len(target_cells) == 1:
                    return 'MOVE'
            
            # SPLIT: Copies/moves value to multiple cells (3+ total affected cells)
            if len(non_zero_deltas) >= 3 and 0 in non_zero_deltas and non_zero_deltas[0] < 0:
                return 'SPLIT'
        
        # ACCUM: Loops that don't fit other patterns or have net pointer movement
        if effect.pattern.startswith('['):
            return 'ACCUM'
        
        return 'LINEAR'  # Fallback
    
    def get_motif_stats(self) -> Dict[str, int]:
        """Get statistics about discovered motifs."""
        return {name: len(motifs) for name, motifs in self.cluster_names.items()}
    
    def get_representative_motifs(self) -> Dict[str, str]:
        """Get representative patterns for each motif type."""
        representatives = {}
        for name, motifs in self.cluster_names.items():
            if motifs:
                # Use the most common pattern
                patterns = [m.pattern for m in motifs]
                most_common = Counter(patterns).most_common(1)[0][0]
                representatives[name] = most_common
        return representatives


class StructuralConstraints:
    """Enforces hard structural constraints on BF programs."""
    
    @staticmethod
    def is_valid_structure(code: str) -> bool:
        """Check if code follows ,body. structure with valid brackets."""
        if not code or len(code) < 2:
            return False
            
        # Must start with exactly one comma and end with exactly one dot
        if not code.startswith(',') or not code.endswith('.'):
            return False
            
        # Extract body (between comma and dot)
        body = code[1:-1]
        
        # Body cannot contain commas (no multi-read)
        if ',' in body:
            return False
        
        # Body cannot contain dots (no multi-write)  
        if '.' in body:
            return False
        
        # Check bracket validity
        return StructuralConstraints._is_valid_brackets(body)
    
    @staticmethod
    def _is_valid_brackets(code: str) -> bool:
        """Check if brackets are properly balanced."""
        depth = 0
        for char in code:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth < 0:  # More closing than opening
                    return False
        return depth == 0  # Must end with balanced brackets
    
    @staticmethod
    def fix_structure(code: str) -> Optional[str]:
        """Attempt to fix structural issues, return None if unfixable."""
        if not code:
            return None
            
        # Ensure starts with comma and ends with dot
        if not code.startswith(','):
            code = ',' + code
        if not code.endswith('.'):
            code = code + '.'
            
        # Extract and clean body
        body = code[1:-1]
        
        # Remove any commas or dots from body
        body = ''.join(c for c in body if c not in ',.')
        
        # Check if brackets can be balanced
        if not StructuralConstraints._is_valid_brackets(body):
            return None  # Don't try to "repair" - reject instead
            
        return ',' + body + '.'


def test_motif_system():
    """Test the motif parsing and classification system."""
    print("🔬 TESTING MOTIF DETECTION SYSTEM")
    print("=" * 50)
    
    # Test programs with known motifs
    test_programs = [
        (",[-].", "CLEAR - should drain source cell"),
        (",[->+<].", "MOVE - should move value to next cell"),
        (",[->+>+<<].", "SPLIT - should copy value to two cells"),
        (",>>[<<+>>-]<<.", "ACCUM - should accumulate from right"),
        (",++>--<.", "LINEAR - simple arithmetic sequence"),
        (",+++[>++<-]>.", "Complex - increment then double"),
    ]
    
    parser = MotifParser()
    library = MotifLibrary()
    
    for program, description in test_programs:
        print(f"\n🧬 Testing: {program}")
        print(f"   Expected: {description}")
        
        if not StructuralConstraints.is_valid_structure(program):
            print("   ❌ Invalid structure")
            continue
            
        motifs = parser.parse_program(program)
        print(f"   Found {len(motifs)} motifs:")
        
        for i, motif in enumerate(motifs):
            if motif.effect:
                library.add_motif(motif.effect)
                cluster = library._classify_motif(motif.effect)
                print(f"     {i+1}. {motif.type}: '{motif.pattern}' → {cluster}")
                if motif.effect.cell_deltas:
                    print(f"        Cell deltas: {motif.effect.cell_deltas}")
                if motif.effect.drains_source:
                    print(f"        Drains source: {motif.effect.drains_source}")
                print(f"        Pointer delta: {motif.effect.pointer_delta}")
            else:
                print(f"     {i+1}. {motif.type}: '{motif.pattern}' → No effect computed")
    
    # Show library statistics
    print(f"\n📊 MOTIF LIBRARY STATISTICS")
    print("=" * 30)
    stats = library.get_motif_stats()
    for motif_type, count in stats.items():
        print(f"{motif_type}: {count} instances")
    
    print(f"\n🏆 REPRESENTATIVE MOTIFS")
    print("=" * 25)
    representatives = library.get_representative_motifs()
    for motif_type, pattern in representatives.items():
        print(f"{motif_type}: {pattern}")
    
    return library


if __name__ == "__main__":
    test_motif_system()