# Multi-Level Hierarchical Motif Discovery Implementation

This document describes the implementation of the generalized multi-level hierarchical motif discovery system that extends beyond L1→L2 to arbitrary levels L3, L4, L5, etc.

## Overview

The system implements the multi-level loop described in `HIERARCHICAL_MOTIF_DISCOVERY.md`:

1. **Level 0**: Primitives (`>`, `<`, `+`, `-`, `[`, `]`)
2. **Level K**: Motifs whose expansion contains only {primitives ∪ motifs with level ≤ K-1}
3. **Grammar DAG**: Expansions only point "downward" in the level hierarchy

## Key Components

### 1. Generalized Mining Functions

#### `mine_lk_motifs()`
- **Purpose**: Mine level-K motifs for any K ≥ 2
- **Input**: Winners data, current grammar, target level
- **Process**: 
  - Symbolize programs using motifs up to level K-1
  - Mine frequent n-grams from symbol sequences
  - Filter by acyclicity and MDL gain
- **Output**: List of motif symbol sequences

#### `is_acyclic_for_level()`
- **Purpose**: Enforce level constraints for acyclicity
- **Rule**: Level K motifs can only reference primitives or motifs with level ≤ K-1
- **Prevents**: Cycles in the grammar DAG

#### `add_lk_motifs()`
- **Purpose**: Add level-K motifs to grammar with proper metadata
- **Features**: 
  - Assigns level metadata to each motif
  - Maintains acyclicity through double-checking
  - Updates Body rules and renormalizes weights

### 2. Enhanced HierarchicalMiner Class

#### Configuration
```python
# L1 configuration (token-level mining)
l1_config = {
    'step_limit': 1000,
    'min_support': 3,
    'mdl_lambda': 1.5,
    'mdl_thresh': 3.0,
    'max_linear_len': 6,
    'min_len': 3,
    'motif_weight': 0.1
}

# LK configuration (symbol-level mining for L2, L3, L4, ...)
lk_config = {
    'min_support': 3,
    'mdl_lambda': 1.0,
    'mdl_thresh': 2.0,
    'min_n': 2,
    'max_n': 5,
    'motif_weight': 0.1
}
```

#### Multi-Level Pipeline
```python
def run_pipeline(self, winners_data, max_iterations=5):
    for iteration in range(1, max_iterations + 1):
        # Phase 1: Mine L1 motifs (token-level)
        l1_motifs = mine_l1(...)
        if l1_motifs:
            self.current_grammar = add_l1_motifs(...)
        
        # Phase 2+: Mine L2, L3, L4, ... (symbol-level)
        level = 2
        while level <= self.max_levels:
            lk_motifs = mine_lk_motifs(..., level=level)
            if lk_motifs:
                self.current_grammar = add_lk_motifs(..., level=level)
                level += 1  # Continue to next level
            else:
                break  # No more motifs at this level
        
        # Stop if no motifs discovered in this iteration
        if total_discovered == 0:
            break
```

## Level Progression Example

### Level 1 (L1): Token-Level Motifs
- **Input**: Raw Brainfuck token sequences
- **Discovers**: `++`, `>>`, `[>+<-]`, `[>-<+]`
- **Criteria**: Behavioral validation + MDL gain

### Level 2 (L2): Symbol-Level Motifs  
- **Input**: Symbol sequences using L1 motifs
- **Example**: `["++", "Motif_1", ">>"]` → `Motif_5`
- **Discovers**: Compositions of L1 motifs and primitives

### Level 3 (L3): Higher-Order Compositions
- **Input**: Symbol sequences using L1 + L2 motifs
- **Example**: `["Motif_5", "Motif_3"]` → `Motif_8`
- **Discovers**: Compositions of L2 motifs with L1 motifs/primitives

### Level K: Arbitrary Compositions
- **Input**: Symbol sequences using motifs up to level K-1
- **Discovers**: Compositions over {primitives ∪ motifs level ≤ K-1}

## Stopping Criteria

The system stops level discovery when:

1. **No new motifs**: No motifs pass the acceptance criteria at level K
2. **MDL threshold**: Total MDL gain falls below threshold
3. **Max levels**: Configurable safety limit reached
4. **Iteration limit**: Maximum iterations completed

## Grammar Structure

### Motif Metadata
```json
{
  "Motif_42": {
    "level": 3,
    "rules": [
      [["Motif_15", "Motif_23"], 1.0]
    ]
  }
}
```

### Level Constraints
- **Level 1**: Can reference only primitives
- **Level 2**: Can reference primitives + L1 motifs  
- **Level K**: Can reference primitives + motifs with level ≤ K-1

## Usage

### Command Line
```bash
python -m core.hierarchical_miner \
    --winners winners.json \
    --base-grammar base_grammar.json \
    --output-dir output/ \
    --max-levels 10 \
    --max-iterations 5 \
    --lk-min-support 3 \
    --lk-mdl-thresh 2.0
```

### Programmatic
```python
from core.hierarchical_miner import HierarchicalMiner

miner = HierarchicalMiner(
    base_grammar_path="base_grammar.json",
    output_dir="output/",
    max_levels=10,
    lk_config={'min_support': 3, 'mdl_thresh': 2.0}
)

final_grammar = miner.run_pipeline(winners_data)
```

## Benefits

1. **Unlimited Hierarchy**: No artificial limit at L2
2. **Natural Stopping**: Discovers levels until no more compression possible
3. **DAG Preservation**: Maintains acyclic grammar structure
4. **Semantic Inheritance**: Higher levels inherit semantics from components
5. **Efficient Evolution**: Larger building blocks accelerate search

## Testing

Run the test script to see the system in action:

```bash
python test_multilevel_mining.py
```

This creates synthetic data with hierarchical patterns and demonstrates discovery of L1, L2, L3+ motifs.


## Mining Motifs
Genomes (programs) can be represented at primitive level (level-0) with just BF tokens. This is what we do when we run evolution as BrainfuckInterpreter expects level-0 representation. But we can also represent the genomes at level-1 motifs, which will be mixture of level-1 motifs and level-0 tokens. Or level-k motifs, which will be mixture of motifs from lower levels. 

When mining motifs for level-k from the winners, always use the level-{k-1} representation of the winner genome. Which means, first start mapping genomes to the highest level below the level k (i.e. level k-1) motifs we have in the motif repository. Then move all the way down to the level-1 motifs. The resulting representation will be called level-{k-1} representation. Then mine the level-k motifs from these winner genomes. 