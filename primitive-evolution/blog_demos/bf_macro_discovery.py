#!/usr/bin/env python3
"""
Macro Discovery for Brainfuck Evolution

Analyzes successful genomes from genome_repository.json to discover
statistically enriched patterns (macros) that can accelerate future evolution.

The discovered macros represent common building blocks that evolution has
found useful across different tasks, essentially creating a library of
evolved programming primitives.

Key Features:
- Pattern extraction from successful genomes
- Statistical enrichment analysis vs random programs
- Frequency-based filtering and ranking
- Macro validation on known tasks
- Automatic macro_repository.json generation
"""

import json
import re
import random
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import itertools
from datetime import datetime


@dataclass
class MacroCandidate:
    """A candidate macro pattern discovered from genomes."""
    pattern: str
    frequency: int
    tasks: Set[str]  # Tasks where this pattern appears
    genomes: List[str]  # Genomes containing this pattern
    enrichment_score: float  # Statistical enrichment vs random
    functionality_score: float  # How useful the pattern seems
    
    def to_dict(self):
        return {
            'pattern': self.pattern,
            'frequency': self.frequency,
            'tasks': list(self.tasks),
            'example_genomes': self.genomes[:5],  # Store first 5 examples
            'enrichment_score': self.enrichment_score,
            'functionality_score': self.functionality_score
        }


class MacroDiscovery:
    """Discovers useful macro patterns from successful genomes."""
    
    def __init__(self, genome_repository_path: str = "primitive-evolution/blog_demos/genome_repository.json"):
        self.genome_repository_path = genome_repository_path
        self.genomes_data = None
        self.successful_genomes = []
        self.macro_candidates = []
        
        # Configuration
        self.min_pattern_length = 2
        self.max_pattern_length = 8
        self.min_frequency = 3  # Pattern must appear at least 3 times
        self.min_enrichment = 2.0  # Must be 2x more common than in random programs
        self.min_tasks = 2  # Must appear in at least 2 different tasks
        
    def load_genome_repository(self) -> bool:
        """Load genomes from the repository file."""
        try:
            with open(self.genome_repository_path, 'r') as f:
                self.genomes_data = json.load(f)
            
            # Extract successful genomes (high accuracy)
            self.successful_genomes = []
            genomes = self.genomes_data.get('genomes', [])
            if isinstance(genomes, list):
                # Handle list format
                for genome_info in genomes:
                    accuracy = genome_info.get('accuracy', 0)
                    if accuracy >= 90.0:  # Only consider highly successful genomes
                        self.successful_genomes.append({
                            'code': genome_info['code'],
                            'function_name': genome_info.get('function_name', 'unknown'),
                            'accuracy': accuracy,
                            'fitness': genome_info.get('fitness', 0)
                        })
            else:
                # Handle dict format (legacy)  
                for genome_info in genomes.values():
                    accuracy = genome_info.get('accuracy', 0)
                    if accuracy >= 90.0:  # Only consider highly successful genomes
                        self.successful_genomes.append({
                            'code': genome_info['code'],
                            'function_name': genome_info.get('function_name', 'unknown'),
                            'accuracy': accuracy,
                            'fitness': genome_info.get('fitness', 0)
                        })
            
            print(f"📚 Loaded {len(self.successful_genomes)} successful genomes from repository")
            return len(self.successful_genomes) > 0
            
        except FileNotFoundError:
            print(f"❌ Repository file {self.genome_repository_path} not found")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {self.genome_repository_path}")
            return False
    
    def extract_body_patterns(self, code: str) -> str:
        """Extract the body part of a ,<body>. program."""
        if len(code) >= 3 and code.startswith(',') and code.endswith('.'):
            return code[1:-1]  # Remove , and .
        return code  # Return as-is if not structured properly
    
    def generate_all_subpatterns(self, body: str) -> List[str]:
        """Generate all substring patterns of various lengths."""
        patterns = []
        for length in range(self.min_pattern_length, min(self.max_pattern_length + 1, len(body) + 1)):
            for start in range(len(body) - length + 1):
                pattern = body[start:start + length]
                # Only keep patterns that contain meaningful operations
                if self.is_meaningful_pattern(pattern):
                    patterns.append(pattern)
        return patterns
    
    def is_meaningful_pattern(self, pattern: str) -> bool:
        """Check if a pattern is potentially meaningful."""
        # Skip patterns that are just repetitions of single characters
        if len(set(pattern)) == 1:
            return False
        
        # Skip patterns with unbalanced brackets
        if not self.has_valid_brackets(pattern):
            return False
        
        # Must contain at least one operation (not just movement)
        if not any(c in pattern for c in '+-[]'):
            return False
        
        return True
    
    def has_valid_brackets(self, pattern: str) -> bool:
        """Check if brackets in pattern are balanced."""
        depth = 0
        for char in pattern:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0
    
    def extract_patterns_from_genomes(self) -> Dict[str, MacroCandidate]:
        """Extract all patterns from successful genomes."""
        print("🔍 Extracting patterns from successful genomes...")
        
        pattern_info = defaultdict(lambda: {
            'frequency': 0,
            'tasks': set(),
            'genomes': []
        })
        
        for genome in self.successful_genomes:
            body = self.extract_body_patterns(genome['code'])
            task = genome['function_name']
            
            # Generate all subpatterns from this genome's body
            patterns = self.generate_all_subpatterns(body)
            
            for pattern in patterns:
                pattern_info[pattern]['frequency'] += 1
                pattern_info[pattern]['tasks'].add(task)
                if len(pattern_info[pattern]['genomes']) < 10:  # Store up to 10 examples
                    pattern_info[pattern]['genomes'].append(genome['code'])
        
        # Convert to MacroCandidate objects
        candidates = {}
        for pattern, info in pattern_info.items():
            if (info['frequency'] >= self.min_frequency and 
                len(info['tasks']) >= self.min_tasks):
                
                candidates[pattern] = MacroCandidate(
                    pattern=pattern,
                    frequency=info['frequency'],
                    tasks=info['tasks'],
                    genomes=info['genomes'],
                    enrichment_score=0.0,  # Will be computed later
                    functionality_score=0.0  # Will be computed later
                )
        
        print(f"📊 Found {len(candidates)} candidate patterns")
        return candidates
    
    def compute_enrichment_scores(self, candidates: Dict[str, MacroCandidate]) -> None:
        """Compute statistical enrichment vs random programs."""
        print("📈 Computing enrichment scores...")
        
        # Generate random programs for comparison
        num_random = 1000
        random_programs = []
        for _ in range(num_random):
            length = random.randint(5, 20)
            commands = '><+-[]'
            program = ''.join(random.choice(commands) for _ in range(length))
            random_programs.append(program)
        
        # Count pattern frequencies in random programs
        random_pattern_counts = defaultdict(int)
        for program in random_programs:
            patterns = self.generate_all_subpatterns(program)
            for pattern in patterns:
                random_pattern_counts[pattern] += 1
        
        # Compute enrichment for each candidate
        for pattern, candidate in candidates.items():
            random_freq = random_pattern_counts.get(pattern, 0)
            random_rate = random_freq / num_random if num_random > 0 else 0
            
            successful_rate = candidate.frequency / len(self.successful_genomes) if len(self.successful_genomes) > 0 else 0
            
            if random_rate > 0:
                enrichment = successful_rate / random_rate
            else:
                enrichment = successful_rate * 100  # High enrichment if not in random
            
            candidate.enrichment_score = enrichment
    
    def compute_functionality_scores(self, candidates: Dict[str, MacroCandidate]) -> None:
        """Compute functionality scores based on pattern structure."""
        print("⚙️ Computing functionality scores...")
        
        for candidate in candidates.values():
            pattern = candidate.pattern
            score = 0.0
            
            # Bonus for balanced loops
            if '[' in pattern and ']' in pattern:
                if self.has_valid_brackets(pattern):
                    score += 2.0  # Balanced loops are very valuable
                    
                    # Extra bonus for common loop patterns
                    if pattern == '[-]':
                        score += 3.0  # Clear cell
                    elif pattern == '[+]':
                        score += 2.0  # Infinite loop (less useful but common)
                    elif '+' in pattern and '<' in pattern and '>' in pattern:
                        score += 2.5  # Movement with arithmetic in loop
            
            # Bonus for arithmetic patterns
            if '++' in pattern or '--' in pattern:
                score += 1.0  # Repeated arithmetic
            
            # Bonus for movement patterns  
            if '><' in pattern or '<>' in pattern:
                score += 0.5  # Movement sequences
            
            # Bonus for appearing in multiple tasks
            score += len(candidate.tasks) * 0.5
            
            # Bonus for high frequency
            score += min(candidate.frequency / 10.0, 2.0)  # Cap at 2.0
            
            # Penalty for very long patterns (might be overfitting)
            if len(pattern) > 6:
                score -= 0.5
            
            candidate.functionality_score = score
    
    def filter_and_rank_candidates(self, candidates: Dict[str, MacroCandidate]) -> List[MacroCandidate]:
        """Filter and rank candidates by quality."""
        print("🏆 Filtering and ranking candidates...")
        
        # Filter by enrichment and other criteria
        filtered = []
        for candidate in candidates.values():
            if (candidate.enrichment_score >= self.min_enrichment and
                candidate.frequency >= self.min_frequency and
                len(candidate.tasks) >= self.min_tasks):
                filtered.append(candidate)
        
        # Sort by combined score (enrichment + functionality)
        filtered.sort(key=lambda c: c.enrichment_score + c.functionality_score, reverse=True)
        
        print(f"✅ Selected {len(filtered)} high-quality macros")
        return filtered
    
    def save_macro_repository(self, macros: List[MacroCandidate], output_path: str = "primitive-evolution/blog_demos/macro_repository.json") -> None:
        """Save discovered macros to repository file."""
        print(f"💾 Saving macro repository to {output_path}")
        
        # Create repository structure
        repository = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'source_genomes': len(self.successful_genomes),
                'total_candidates': len(macros),
                'discovery_config': {
                    'min_pattern_length': self.min_pattern_length,
                    'max_pattern_length': self.max_pattern_length,
                    'min_frequency': self.min_frequency,
                    'min_enrichment': self.min_enrichment,
                    'min_tasks': self.min_tasks
                }
            },
            'macros': [macro.to_dict() for macro in macros],
            'useful_chunks': [macro.pattern for macro in macros[:20]]  # Top 20 for evolution
        }
        
        with open(output_path, 'w') as f:
            json.dump(repository, f, indent=2)
        
        print(f"✅ Saved {len(macros)} macros to repository")
    
    def discover_macros(self, output_path: str = "primitive-evolution/blog_demos/macro_repository.json") -> List[MacroCandidate]:
        """Run complete macro discovery pipeline."""
        print("🧬 BRAINFUCK MACRO DISCOVERY")
        print("=" * 40)
        
        # Step 1: Load genome repository
        if not self.load_genome_repository():
            print("❌ Could not load genome repository")
            return []
        
        # Step 2: Extract patterns
        candidates = self.extract_patterns_from_genomes()
        if not candidates:
            print("❌ No candidate patterns found")
            return []
        
        # Step 3: Compute enrichment scores
        self.compute_enrichment_scores(candidates)
        
        # Step 4: Compute functionality scores
        self.compute_functionality_scores(candidates)
        
        # Step 5: Filter and rank
        final_macros = self.filter_and_rank_candidates(candidates)
        
        # Step 6: Save to repository
        if final_macros:
            self.save_macro_repository(final_macros, output_path)
            
            # Show top discoveries
            print("\n🏆 TOP DISCOVERED MACROS:")
            print("-" * 60)
            print(f"{'Pattern':<12} {'Freq':<6} {'Tasks':<6} {'Enrich':<8} {'Func':<6} {'Score':<8}")
            print("-" * 60)
            
            for i, macro in enumerate(final_macros[:15]):
                total_score = macro.enrichment_score + macro.functionality_score
                print(f"{macro.pattern:<12} {macro.frequency:<6} {len(macro.tasks):<6} "
                      f"{macro.enrichment_score:<8.1f} {macro.functionality_score:<6.1f} {total_score:<8.1f}")
        
        return final_macros


def load_macros_for_evolution(macro_repository_path: str = "primitive-evolution/blog_demos/macro_repository.json") -> List[str]:
    """Load discovered macros for use in evolution (useful_chunks)."""
    try:
        with open(macro_repository_path, 'r') as f:
            repository = json.load(f)
        
        useful_chunks = repository.get('useful_chunks', [])
        print(f"📚 Loaded {len(useful_chunks)} useful chunks for evolution")
        return useful_chunks
        
    except FileNotFoundError:
        print(f"⚠️ Macro repository {macro_repository_path} not found, using default chunks")
        return ['++', '--', '><', '<>', '[+]', '[-]', '[>+<-]', '>+<', '<+>']
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON in {macro_repository_path}, using default chunks")
        return ['++', '--', '><', '<>', '[+]', '[-]', '[>+<-]', '>+<', '<+>']


def analyze_existing_repository():
    """Analyze the current genome repository to show what we have."""
    try:
        with open("primitive-evolution/blog_demos/genome_repository.json", 'r') as f:
            data = json.load(f)
        
        genomes = data.get('genomes', [])
        print(f"📊 GENOME REPOSITORY ANALYSIS")
        print(f"=" * 40)
        print(f"Total genomes: {len(genomes)}")
        
        # Analyze by task
        task_counts = defaultdict(int)
        accuracy_by_task = defaultdict(list)
        
        genome_list = genomes if isinstance(genomes, list) else genomes.values()
        for genome_info in genome_list:
            task = genome_info.get('function_name', 'unknown')
            accuracy = genome_info.get('accuracy', 0)
            task_counts[task] += 1
            accuracy_by_task[task].append(accuracy)
        
        print(f"\nBy Task:")
        for task, count in task_counts.items():
            avg_accuracy = sum(accuracy_by_task[task]) / len(accuracy_by_task[task])
            high_accuracy = sum(1 for acc in accuracy_by_task[task] if acc >= 90)
            print(f"  {task}: {count} genomes, avg accuracy: {avg_accuracy:.1f}%, high accuracy: {high_accuracy}")
        
        return len(genomes) > 0
        
    except FileNotFoundError:
        print(f"❌ primitive-evolution/blog_demos/genome_repository.json not found")
        return False


def main():
    """Run macro discovery on existing genome repository."""
    print("🔬 MACRO DISCOVERY FROM EVOLVED GENOMES")
    print("=" * 50)
    
    # First, analyze what we have
    if not analyze_existing_repository():
        print("\n💡 To use macro discovery:")
        print("1. Run evolution on different tasks to populate genome_repository.json")
        print("2. Then run this script to discover useful macro patterns")
        print("3. The discovered macros will be used as 'useful_chunks' in future evolution")
        return
    
    # Run macro discovery
    discovery = MacroDiscovery()
    macros = discovery.discover_macros()
    
    if macros:
        print(f"\n✅ Discovery complete! Found {len(macros)} useful macros")
        print(f"📁 Macros saved to primitive-evolution/blog_demos/macro_repository.json")
        print(f"🧬 These macros can now be used as building blocks in evolution")
        
        # Show how to use them
        print(f"\n💡 To use discovered macros in evolution:")
        print(f"  1. Load macros: useful_chunks = load_macros_for_evolution()")
        print(f"  2. Use in mutation: chunk = random.choice(useful_chunks)")
        print(f"  3. Evolution will now have access to evolved building blocks!")
    else:
        print(f"\n⚠️ No macros discovered. Try running more evolution tasks first.")


if __name__ == "__main__":
    main()