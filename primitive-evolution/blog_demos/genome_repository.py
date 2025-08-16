#!/usr/bin/env python3
"""
Genome Repository System for Brainfuck Evolution
Manages a persistent collection of successful genomes across all tasks.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GenomeEntry:
    """Represents a genome entry in the repository."""
    code: str
    fitness: float
    accuracy: float
    function_name: str
    test_cases: List[int]
    expected_outputs: List[int]
    generation_found: int
    timestamp: str
    task_id: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenomeEntry':
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)


class GenomeRepository:
    """Manages a persistent repository of successful Brainfuck genomes."""
    
    def __init__(self, repository_path: str = "genome_repository.json"):
        self.repository_path = Path(repository_path)
        self.genomes: List[GenomeEntry] = []
        self.load_repository()
    
    def load_repository(self) -> None:
        """Load genomes from the repository file."""
        if self.repository_path.exists():
            try:
                with open(self.repository_path, 'r') as f:
                    data = json.load(f)
                    self.genomes = [GenomeEntry.from_dict(entry) for entry in data.get('genomes', [])]
                print(f"📚 Loaded {len(self.genomes)} genomes from repository")
            except Exception as e:
                print(f"⚠️ Error loading repository: {e}")
                self.genomes = []
        else:
            print(f"📚 Creating new genome repository at {self.repository_path}")
            self.genomes = []
    
    def save_repository(self) -> None:
        """Save genomes to the repository file."""
        try:
            # Create backup if file exists
            if self.repository_path.exists():
                backup_path = self.repository_path.with_suffix('.backup.json')
                self.repository_path.rename(backup_path)
            
            # Save current repository
            data = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'total_genomes': len(self.genomes),
                    'version': '1.0'
                },
                'genomes': [genome.to_dict() for genome in self.genomes]
            }
            
            with open(self.repository_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Saved {len(self.genomes)} genomes to repository")
            
        except Exception as e:
            print(f"❌ Error saving repository: {e}")
    
    def add_genome(self, code: str, fitness: float, accuracy: float,
                   function_name: str, test_cases: List[int], expected_outputs: List[int],
                   generation_found: int, task_id: str = None, metadata: Dict[str, Any] = None) -> None:
        """Add a new genome to the repository."""

        # Avoid duplicates
        if self.has_genome(code):
            print(f"🔄 Genome already exists: {code}")
            return

        # Only add high-quality genomes
        if accuracy < 80.0:  # Only store genomes with 80%+ accuracy
            print(f"📉 Genome accuracy too low ({accuracy:.1f}%), not storing")
            return

        genome = GenomeEntry(
            code=code,
            fitness=fitness,
            accuracy=accuracy,
            function_name=function_name,
            test_cases=test_cases,
            expected_outputs=expected_outputs,
            generation_found=generation_found,
            timestamp=datetime.now().isoformat(),
            task_id=task_id or f"{function_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            metadata=metadata or {}
        )
        
        self.genomes.append(genome)
        print(f"✅ Added genome to repository: {code} (accuracy: {accuracy:.1f}%)")
    
    def has_genome(self, code: str) -> bool:
        """Check if a genome already exists in the repository."""
        return any(genome.code == code for genome in self.genomes)
    
    def get_genomes_for_task(self, function_name: str = None, min_accuracy: float = 0.0) -> List[GenomeEntry]:
        """Get genomes suitable for a specific task."""
        filtered = []
        
        for genome in self.genomes:
            # Filter by function name if specified
            if function_name and genome.function_name != function_name:
                continue
            
            # Filter by minimum accuracy
            if genome.accuracy < min_accuracy:
                continue
            
            filtered.append(genome)
        
        # Sort by accuracy (best first)
        filtered.sort(key=lambda g: g.accuracy, reverse=True)
        return filtered
    
    def get_best_genomes(self, limit: int = 10) -> List[GenomeEntry]:
        """Get the best genomes overall."""
        sorted_genomes = sorted(self.genomes, key=lambda g: g.accuracy, reverse=True)
        return sorted_genomes[:limit]
    
    def get_diverse_genomes(self, limit: int = 10) -> List[GenomeEntry]:
        """Get a diverse set of genomes (different functions, lengths, etc.)."""
        # Group by function
        by_function = {}
        for genome in self.genomes:
            func = genome.function_name
            if func not in by_function:
                by_function[func] = []
            by_function[func].append(genome)
        
        # Take best from each function
        diverse = []
        for func, genomes in by_function.items():
            # Sort by accuracy and take top ones
            genomes.sort(key=lambda g: g.accuracy, reverse=True)
            diverse.extend(genomes[:2])  # Top 2 from each function
        
        # Sort overall and limit
        diverse.sort(key=lambda g: g.accuracy, reverse=True)
        return diverse[:limit]
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get statistics about the repository."""
        if not self.genomes:
            return {
                'total': 0,
                'functions': 0,
                'function_breakdown': {},
                'accuracy_stats': {'min': 0, 'max': 0, 'avg': 0},
                'code_length_stats': {'min': 0, 'max': 0, 'avg': 0},
                'perfect_solutions': 0
            }
        
        # Group by function
        by_function = {}
        for genome in self.genomes:
            func = genome.function_name
            if func not in by_function:
                by_function[func] = []
            by_function[func].append(genome)
        
        # Calculate stats
        accuracies = [g.accuracy for g in self.genomes]
        code_lengths = [len(g.code) for g in self.genomes]
        
        return {
            'total': len(self.genomes),
            'functions': len(by_function),
            'function_breakdown': {func: len(genomes) for func, genomes in by_function.items()},
            'accuracy_stats': {
                'min': min(accuracies),
                'max': max(accuracies),
                'avg': sum(accuracies) / len(accuracies)
            },
            'code_length_stats': {
                'min': min(code_lengths),
                'max': max(code_lengths),
                'avg': sum(code_lengths) / len(code_lengths)
            },
            'perfect_solutions': sum(1 for g in self.genomes if g.accuracy >= 100.0)
        }
    
    def print_repository_summary(self) -> None:
        """Print a summary of the repository."""
        stats = self.get_repository_stats()
        
        print(f"\n📚 Genome Repository Summary")
        print(f"=" * 40)
        print(f"Total genomes: {stats['total']}")
        print(f"Functions covered: {stats['functions']}")
        print(f"Perfect solutions: {stats.get('perfect_solutions', 0)}")
        
        if stats['total'] > 0:
            print(f"\nAccuracy range: {stats['accuracy_stats']['min']:.1f}% - {stats['accuracy_stats']['max']:.1f}%")
            print(f"Average accuracy: {stats['accuracy_stats']['avg']:.1f}%")
            print(f"Code length range: {stats['code_length_stats']['min']} - {stats['code_length_stats']['max']} chars")
            
            print(f"\nBy function:")
            for func, count in stats['function_breakdown'].items():
                print(f"  {func}: {count} genomes")
    
    def export_for_seeding(self, function_name: str = None, limit: int = 5) -> List[str]:
        """Export genome codes for seeding a new population."""
        suitable_genomes = self.get_genomes_for_task(function_name, min_accuracy=80.0)

        if not suitable_genomes:
            # Fall back to diverse genomes if no function-specific ones
            suitable_genomes = self.get_diverse_genomes(limit)

        return [genome.code for genome in suitable_genomes[:limit]]


# Global repository instance
_global_repository = None

def get_global_repository() -> GenomeRepository:
    """Get the global genome repository instance."""
    global _global_repository
    if _global_repository is None:
        repo_path = os.path.join(os.path.dirname(__file__), "genome_repository.json")
        _global_repository = GenomeRepository(repo_path)
    return _global_repository