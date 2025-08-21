"""
Updated version of 03_evolve_with_mined_motifs.py to use hierarchical mining.

This replaces the old single-pass motif mining with the new hierarchical system
that discovers both L1 (token-level) and L2 (symbol-level) motifs.

Usage:
    python 03_evolve_with_mined_motifs_updated.py --winners ../outputs/runs/cfg_best.json --base-grammar ../motifs/minimal_cfg.json --out-grammar ../motifs/hierarchical_cfg.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional

# Make blog_demos a module root for imports like core.* and tasks.*
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the new hierarchical mining system
from core.hierarchical_miner import HierarchicalMiner


def main():
    """Main function that replaces the old motif mining with hierarchical mining."""
    ap = argparse.ArgumentParser(description="Mine hierarchical motifs from winners and expand grammar")
    
    # Input/output arguments
    ap.add_argument('--winners', required=True, help='Path to winners JSON file')
    ap.add_argument('--base-grammar', required=True, help='Path to base grammar')
    ap.add_argument('--out-grammar', required=True, help='Path for output enriched grammar')
    ap.add_argument('--output-dir', help='Directory for intermediate outputs (default: same dir as out-grammar)')
    
    # Hierarchical mining parameters
    ap.add_argument('--max-iterations', type=int, default=3, help='Max hierarchical mining iterations')
    
    # L1 mining parameters
    ap.add_argument('--l1-min-support', type=int, default=3, help='L1 minimum support')
    ap.add_argument('--l1-mdl-thresh', type=float, default=3.0, help='L1 MDL threshold')
    ap.add_argument('--l1-motif-weight', type=float, default=0.1, help='L1 motif weight in grammar')
    ap.add_argument('--l1-step-limit', type=int, default=1000, help='L1 behavioral harness step limit')
    ap.add_argument('--l1-max-linear-len', type=int, default=6, help='L1 max linear segment length')
    ap.add_argument('--l1-min-len', type=int, default=3, help='L1 minimum motif length')
    
    # L2 mining parameters  
    ap.add_argument('--l2-min-support', type=int, default=3, help='L2 minimum support')
    ap.add_argument('--l2-mdl-thresh', type=float, default=2.0, help='L2 MDL threshold')
    ap.add_argument('--l2-motif-weight', type=float, default=0.1, help='L2 motif weight in grammar')
    ap.add_argument('--l2-min-n', type=int, default=2, help='L2 minimum n-gram size')
    ap.add_argument('--l2-max-n', type=int, default=5, help='L2 maximum n-gram size')
    
    args = ap.parse_args()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.out_grammar), 'hierarchical_mining_temp')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Hierarchical Motif Mining System")
    print("=" * 50)
    print(f"Winners file: {args.winners}")
    print(f"Base grammar: {args.base_grammar}")
    print(f"Output grammar: {args.out_grammar}")
    print(f"Output directory: {output_dir}")
    
    # Load winners data
    print(f"\nLoading winners from: {args.winners}")
    try:
        with open(args.winners, 'r') as f:
            winners_data = json.load(f)
        
        total_solutions = sum(len(task_data.get('solutions', [])) 
                            for task_data in winners_data.values())
        print(f"Tasks: {len(winners_data)}")
        print(f"Total solutions: {total_solutions}")
        
        if total_solutions == 0:
            print("WARNING: No solutions found in winners file!")
            return
            
    except Exception as e:
        print(f"Error loading winners: {e}")
        return
    
    # Configure hierarchical miner
    l1_config = {
        'step_limit': args.l1_step_limit,
        'min_support': args.l1_min_support,
        'mdl_lambda': 1.5,  # Fixed for L1
        'mdl_thresh': args.l1_mdl_thresh,
        'max_linear_len': args.l1_max_linear_len,
        'min_len': args.l1_min_len,
        'motif_weight': args.l1_motif_weight
    }
    
    l2_config = {
        'min_support': args.l2_min_support,
        'mdl_lambda': 1.0,  # Fixed for L2
        'mdl_thresh': args.l2_mdl_thresh,
        'min_n': args.l2_min_n,
        'max_n': args.l2_max_n,
        'motif_weight': args.l2_motif_weight
    }
    
    print(f"\nL1 Configuration: {l1_config}")
    print(f"L2 Configuration: {l2_config}")
    
    # Create and run hierarchical miner
    print(f"\nCreating hierarchical miner...")
    miner = HierarchicalMiner(
        base_grammar_path=args.base_grammar,
        output_dir=output_dir,
        l1_config=l1_config,
        l2_config=l2_config
    )
    
    # Run the hierarchical mining pipeline
    print(f"\nRunning hierarchical mining pipeline (max {args.max_iterations} iterations)...")
    enriched_grammar = miner.run_pipeline(winners_data, max_iterations=args.max_iterations)
    
    # Save the final enriched grammar
    print(f"\nSaving final enriched grammar to: {args.out_grammar}")
    with open(args.out_grammar, 'w') as f:
        json.dump(enriched_grammar, f, indent=2)
    
    # Print final summary
    print(f"\nHierarchical motif mining completed successfully!")
    print(f"Final enriched grammar: {args.out_grammar}")
    print(f"Intermediate files: {output_dir}")
    
    # Provide usage example
    print(f"\nTo use the enriched grammar in evolution:")
    print(f"  python your_evolution_script.py --grammar {args.out_grammar}")


if __name__ == '__main__':
    main()