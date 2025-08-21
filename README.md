# Beyond Pattern Matching: Hierarchical Motif Discovery

This repository implements a hierarchical motif discovery system for evolving Brainfuck programs on sequence generation tasks. The system discovers multi-level motifs (L1, L2, L3, ...) and uses them to evolve increasingly sophisticated programs.

## Overview

The project demonstrates how evolutionary algorithms can discover hierarchical patterns in code, moving beyond simple pattern matching to build compositional program structures. Programs evolve to solve sequence generation tasks (arithmetic progressions, oscillators, etc.) while the system automatically discovers reusable motifs at multiple abstraction levels.

## Key Features

- **Multi-level Hierarchical Motif Discovery**: Automatically discovers motifs at L1 (primitive combinations), L2 (L1 motif combinations), L3 (L1+L2 combinations), etc.
- **Grammar-based Evolution**: Uses Probabilistic Context-Free Grammars (PCFGs) for program generation and evolution
- **Behavioral Validation**: Ensures discovered motifs are behaviorally meaningful, not just syntactic patterns
- **MDL-based Quality Control**: Uses Minimum Description Length to filter high-quality, compressive motifs
- **Iterative Evolution**: Evolves programs, discovers motifs, updates grammar, and repeats for continuous improvement

## Project Structure

```
primitive-evolution/blog_demos/
├── core/                          # Core system components
│   ├── hierarchical_miner.py      # Multi-level motif discovery engine
│   ├── miner_l1.py                # L1 (token-level) motif mining
│   ├── miner_l2.py                # L2+ (symbol-level) motif mining
│   ├── grammar_updater.py         # Grammar management and updates
│   ├── fitness.py                 # Fitness functions for evolution
│   ├── evo.py                     # Evolution utilities
│   ├── bf_runner.py               # Brainfuck interpreter
│   └── eval_utils.py              # Evaluation utilities
├── experiments/                   # Main experiment scripts
│   ├── evolve_with_hierarchical_motifs.py  # Main hierarchical evolution pipeline
│   ├── evolve_with_cfg.py         # Grammar-based evolution
│   ├── evolve_baseline.py         # Baseline evolution without motifs
│   ├── evolve_with_mined_motifs.py # Evolution with pre-mined motifs
│   └── config.yaml                # Configuration for hierarchical evolution
├── motifs/                        # Grammar definitions and motif utilities
│   ├── minimal_cfg.json           # Base PCFG for Brainfuck
│   ├── pcfg.py                    # PCFG sampling and operations
│   ├── miner.py                   # Legacy motif mining utilities
│   └── *.json                     # Various enriched grammars
├── tasks/                         # Task definitions
│   ├── sequences.yaml             # Sequence generation tasks
│   └── sequence_suite.py          # Task loading and evaluation
├── docs/                          # Documentation
│   └── MULTILEVEL_IMPLEMENTATION.md # Detailed implementation guide
├── brainfuck.py                   # Brainfuck interpreter
└── brainfuck_evolution.py         # Basic evolution utilities
```

## Quick Start

### 1. Run Hierarchical Evolution Pipeline

The main experiment that demonstrates the full hierarchical motif discovery system:

```bash
cd primitive-evolution/blog_demos
python experiments/evolve_with_hierarchical_motifs.py experiments/config.yaml
```

This will:
1. Start with a minimal grammar
2. Evolve programs on sequence tasks
3. Mine hierarchical motifs from successful programs
4. Update the grammar with discovered motifs
5. Repeat for multiple iterations

### 2. Run Individual Components

**Grammar-based Evolution Only:**
```bash
python experiments/evolve_with_cfg.py --grammar motifs/minimal_cfg.json --pop 1000 --gens 100
```

**Baseline Evolution (no grammar):**
```bash
python experiments/evolve_baseline.py --pop 1000 --gens 100
```

### 3. Configuration

Edit `experiments/config.yaml` to customize:

```yaml
task_suite: sequences              # Task suite to use
max_iterations: 10                 # Number of evolution-mining cycles
evo:
  population_size: 10000          # Evolution population size
  generations: 500                # Generations per evolution round
  step_limit: 1000               # Max steps for program execution
mine:
  enabled: true                   # Enable motif mining
  min_support: 4                  # Minimum motif frequency
  min_len: 3                     # Minimum motif length
```

## System Architecture

### Hierarchical Motif Discovery

The system implements a multi-level motif discovery algorithm:

1. **L1 Motifs**: Combinations of primitives (`++`, `--`, `><`, etc.)
   - Discovered through behavioral validation and MDL scoring
   - Must be behaviorally meaningful (not just syntactic patterns)

2. **L2 Motifs**: Combinations of L1 motifs (`[L1M1, L1M3]`, etc.)
   - Discovered from symbolized programs using L1 motifs
   - Must be acyclic (no self-references)

3. **L3+ Motifs**: Combinations of lower-level motifs
   - Recursive application of the same algorithm
   - Maintains strict level hierarchy

### Key Algorithms

**Iterative L1 Mining**: Ensures complete primitive coverage by mining L1 motifs in multiple rounds until no new patterns emerge.

**Primitive Rejection**: L2+ mining rejects primitive-only patterns that should have been discovered at L1.

**MDL-based Filtering**: Uses Minimum Description Length to ensure motifs provide compression benefit.

**Acyclicity Enforcement**: Prevents grammar cycles by ensuring level-K motifs only reference motifs with level < K.

## Example Output

The system discovers motifs like:

```
L1 Motifs (primitives → tokens):
  L1M1: "++"     (increment twice)
  L1M2: "--"     (decrement twice)
  L1M3: "><"     (move right then left)

L2 Motifs (L1 motifs → symbols):
  L2M1: [L1M1, L1M2]  (increment twice, then decrement twice)
  L2M2: [L1M3, L1M1]  (move right-left, then increment twice)

L3 Motifs (L1+L2 motifs → higher-order symbols):
  L3M1: [L2M1, L1M3]  (complex behavioral pattern)
```

## Tasks

The system evolves programs to solve sequence generation tasks:

- **Arithmetic**: `1,2,3,4,5...` (increment)
- **Geometric**: `1,2,4,8,16...` (doubling)
- **Oscillators**: `1,3,1,3,1,3...` (alternating)
- **Modular**: `1,2,3,1,2,3...` (cyclic patterns)
- **Complex**: Multi-step transformations

## Research Applications

This system demonstrates:

- **Automatic Program Synthesis**: Discovering reusable code patterns
- **Hierarchical Abstraction**: Building compositional program structures
- **Evolutionary Computation**: Grammar-guided program evolution
- **Motif Discovery**: Finding meaningful patterns in code
- **Compression-based Learning**: Using MDL for pattern quality assessment

## Documentation

- `docs/MULTILEVEL_IMPLEMENTATION.md`: Detailed technical implementation
- Code comments: Extensive documentation throughout the codebase
- Configuration files: Well-documented YAML configurations

## Dependencies

- Python 3.9+
- Standard library only (no external dependencies)
- Optional: YAML for configuration files

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{hierarchical_motif_discovery_2025,
  author = {Umut Eser},
  title = {Beyond Pattern Matching: Hierarchical Motif Discovery for Program Evolution},
  year = {2025},
  url = {https://github.com/umut-eser/beyond-pattern-matching},
  note = {Blog post: [Teaching AI to Build Its Own Building](https://umuteser1.substack.com/p/teaching-ai-to-build-its-own-building)}
}
```

**APA Format:**
```
Eser, U. (2025). Beyond Pattern Matching: Hierarchical Motif Discovery for Program Evolution.
GitHub repository. https://github.com/umut-eser/beyond-pattern-matching
```

**IEEE Format:**
```
U. Eser, "Beyond Pattern Matching: Hierarchical Motif Discovery for Program Evolution,"
GitHub repository, 2025. [Online]. Available: https://github.com/umut-eser/beyond-pattern-matching
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

The codebase is designed for research and experimentation. Key areas for extension:

- New task domains beyond sequence generation
- Alternative motif quality metrics beyond MDL
- Different grammar formalisms beyond PCFGs
- Parallel/distributed evolution strategies
- Integration with other program synthesis approaches

## Acknowledgments

This work explores hierarchical motif discovery in evolutionary program synthesis, demonstrating how multi-level abstractions can emerge from simple evolutionary processes.
