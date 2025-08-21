# Motif Emergence Experiments

Clean, minimal experiments demonstrating two key insights for blog post Section 2:

1. **Motif Emergence**: Evolution across multiple tasks produces reusable code patterns
2. **Motif Reuse**: These patterns accelerate convergence on new tasks

## Standard Directory Structure

```
outputs/
├── runs/               # Evolution data (CSV + JSON)
├── motifs/             # Discovered motifs (JSON) 
└── compare/            # Comparison results
```

No need to specify paths - everything uses standard locations automatically.

## Quick Usage

### Individual Experiments

```bash
# Step 1: Baseline evolution (primitives only)
python experiments/01_evolve_baseline.py --tasks increment,clear,double --pop 100 --gens 50

# Step 2: Mine motifs from successful programs  
python experiments/02_mine_motifs.py

# Step 3: Evolution with motif reuse
python experiments/03_evolve_with_motifs.py --tasks increment,clear,double --pop 100 --gens 50

# Step 4: Compare results
python experiments/04_compare_runs.py
```

### Complete Experiment

```bash
# Run all steps automatically
python run_motif_experiment.py
```

## Results

The comparison script shows:
- **Best fitness** achieved per task
- **Convergence speed** (generations to reach 80% and 100% accuracy)
- **Summary** of improvements with motifs

Example output:
```
MOTIF EMERGENCE EXPERIMENT RESULTS
==================================
Task         Best Exact           Gens to 80%          Gens to 100%        
             Base      Motif     Base      Motif     Base      Motif    
------------------------------------------------------------------------
increment    1.000     1.000     3         0         3         0        
double       0.200     0.400     Never     Never     Never     Never    
clear        1.000     1.000     0         1         0         1        

Tasks where motifs improved best fitness:
  • double: 0.200 → 0.400

Tasks with faster convergence using motifs:
  80% threshold: 1/3 tasks
  100% threshold: 1/3 tasks
```

## For Blog Post

This demonstrates:
1. **Section 2.1**: Motifs emerge from multi-task evolution
2. **Section 2.2**: Motif reuse speeds up learning (3 → 0 gens for increment)

Clean, reproducible results with no manual path management required!