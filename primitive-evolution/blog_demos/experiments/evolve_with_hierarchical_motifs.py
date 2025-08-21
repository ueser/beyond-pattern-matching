"""
Evolution experiment with hierarchical motif mining integration.

This experiment demonstrates the complete hierarchical motif mining pipeline:
1. Start with a base grammar and run evolution
2. Mine L1 and L2 motifs from winners
3. Use enriched grammar for next evolution round
4. Repeat until convergence

Usage:
    python 03_evolve_with_hierarchical_motifs.py --task-suite sequences --max-iterations 3
"""

import argparse
import json
import os
import sys
import shutil
import subprocess
from typing import Dict, Any, List, Optional

# Optional YAML support for config files
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Add blog_demos to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Global pipeline config (filled in main())
PIPELINE_CONFIG: Dict[str, Any] = {}

# Lazy imports inside functions to allow --help without requiring optional modules
# from core.hierarchical_miner import HierarchicalMiner
# from core.evo import run_evolution
# from tasks.sequence_suite import SequenceSuite


def load_task_suite(suite_name: str, tasks_yaml_override: Optional[str] = None) -> Dict[str, Any]:
    """Load task suite configuration and return tasks plus the YAML path used."""
    from tasks.sequence_suite import load_sequence_tasks
    repo_root = os.path.dirname(os.path.dirname(__file__))

    # Map task suite names to their YAML files
    suite_yaml_map = {
        "sequences": "sequences.yaml",
        "sequences_toy": "sequences_toy.yaml"
    }

    if suite_name not in suite_yaml_map:
        available_suites = list(suite_yaml_map.keys())
        raise ValueError(f"Unknown task suite: '{suite_name}'. Available suites: {available_suites}")

    # Determine YAML file path
    if tasks_yaml_override:
        tasks_yaml = tasks_yaml_override
    else:
        yaml_filename = suite_yaml_map[suite_name]
        tasks_yaml = os.path.join(repo_root, 'tasks', yaml_filename)

    # Check if YAML file exists
    if not os.path.exists(tasks_yaml):
        raise FileNotFoundError(f"Task suite YAML file not found: {tasks_yaml}")

    # Load tasks
    tasks_list = load_sequence_tasks(tasks_yaml)
    tasks_dict = {t.name: t for t in tasks_list}

    return {
        "suite_name": suite_name,
        "tasks": tasks_dict,
        "task_count": len(tasks_dict),
        "tasks_yaml": tasks_yaml,
    }


def run_evolution_round_progressive(grammar_path: str, output_dir: str, tasks: Dict[str, Any],
                                   round_num: int, evo_config: Dict[str, Any],
                                   prev_winners_file: Optional[str] = None,
                                   tasks_yaml: Optional[str] = None) -> str:
    """
    Run one round of evolution with progressive learning within the round.
    Tasks are processed sequentially, with each task using solutions from previous tasks in the same round.
    """
    print(f"\n{'='*60}")
    print(f"EVOLUTION ROUND {round_num}")
    print(f"{'='*60}")

    # Create output subdirectory for this round
    round_output_dir = os.path.join(output_dir, f"round_{round_num}")
    os.makedirs(round_output_dir, exist_ok=True)

    # Output files
    stats_file = os.path.join(round_output_dir, "evolution_stats.csv")
    winners_file = os.path.join(round_output_dir, "winners.json")

    print(f"Running progressive evolution with grammar: {grammar_path}")
    print(f"Tasks: {len(tasks)}")
    print(f"Config: {evo_config}")
    print(f"Output: {round_output_dir}")

    # Step 1: Initialize winners.json for this round from previous round
    if prev_winners_file and os.path.exists(prev_winners_file):
        print(f"[Progressive] Copying winners from previous round: {prev_winners_file}")
        shutil.copyfile(prev_winners_file, winners_file)
        with open(winners_file, 'r') as f:
            current_winners = json.load(f)
        print(f"[Progressive] Initialized with {len(current_winners)} task solutions from previous round")
    else:
        print(f"[Progressive] Starting with empty winners (first round)")
        current_winners = {}
        with open(winners_file, 'w') as f:
            json.dump(current_winners, f, indent=2)

    # Step 2: Process tasks sequentially, updating winners after each task
    repo_root = os.path.dirname(os.path.dirname(__file__))
    all_results = []

    # Load task definitions
    if tasks_yaml:
        sys.path.append(repo_root)
        from tasks.sequence_suite import load_sequence_tasks
        all_tasks = {t.name: t for t in load_sequence_tasks(tasks_yaml)}
    else:
        all_tasks = tasks

    task_names = list(tasks.keys()) if isinstance(tasks, dict) else list(all_tasks.keys())

    for i, task_name in enumerate(task_names):
        print(f"\n[Progressive] Processing task {i+1}/{len(task_names)}: {task_name}")

        # Run evolution for this single task
        task_winners_file = run_single_task_evolution(
            grammar_path=grammar_path,
            task_name=task_name,
            round_output_dir=round_output_dir,
            evo_config=evo_config,
            init_genomes=winners_file,  # Use current round's winners
            tasks_yaml=tasks_yaml,
            task_index=i
        )

        # Update the round's winners.json with new solutions
        if os.path.exists(task_winners_file):
            with open(task_winners_file, 'r') as f:
                task_results = json.load(f)

            # Merge new solutions into current winners
            for task, task_data in task_results.items():
                if task in current_winners:
                    # Merge solutions, avoiding duplicates
                    existing_solutions = set(current_winners[task].get('solutions', []))
                    new_solutions = task_data.get('solutions', [])
                    merged_solutions = list(existing_solutions.union(set(new_solutions)))
                    current_winners[task] = {
                        'solutions': merged_solutions,
                        'count': len(merged_solutions)
                    }
                    print(f"[Progressive] Updated {task}: {len(existing_solutions)} -> {len(merged_solutions)} solutions")
                else:
                    current_winners[task] = task_data
                    print(f"[Progressive] Added {task}: {len(task_data.get('solutions', []))} solutions")

            # Save updated winners
            with open(winners_file, 'w') as f:
                json.dump(current_winners, f, indent=2)

            print(f"[Progressive] Updated winners.json with solutions from {task_name}")

    print(f"\n[Progressive] Evolution round completed. Final winners saved to: {winners_file}")

    # Verify winners file exists and has content
    with open(winners_file, 'r') as f:
        winners = json.load(f)
        total_solutions = sum(len(task_data.get('solutions', [])) for task_data in winners.values())
        print(f"Total solutions found: {total_solutions}")

    return winners_file


def run_single_task_evolution(grammar_path: str, task_name: str, round_output_dir: str,
                             evo_config: Dict[str, Any], init_genomes: Optional[str] = None,
                             tasks_yaml: Optional[str] = None, task_index: int = 0) -> str:
    """
    Run evolution for a single task and return the winners file path.
    """
    print(f"  Running evolution for task: {task_name}")

    try:
        repo_root = os.path.dirname(os.path.dirname(__file__))
        evo_script = os.path.join(repo_root, 'experiments', 'evolve_with_cfg.py')

        # Create task-specific output files
        task_output_file = os.path.join(round_output_dir, f"task_{task_index}_{task_name}_best.json")

        # Map config to script args
        cmd = [sys.executable, evo_script, '--grammar', grammar_path]
        cmd += ['--tasks', task_name]  # Run only this specific task

        if 'population_size' in evo_config:
            cmd += ['--pop', str(int(evo_config['population_size']))]
        if 'generations' in evo_config:
            cmd += ['--gens', str(int(evo_config['generations']))]
        if 'seed' in evo_config:
            cmd += ['--seed', str(int(evo_config['seed']) + task_index)]  # Different seed per task
        if 'step_limit' in evo_config:
            cmd += ['--step-limit', str(int(evo_config['step_limit']))]
        if 'max_symbols' in evo_config:
            cmd += ['--max-len', str(int(evo_config['max_symbols']))]
        elif 'max_program_length' in evo_config:
            cmd += ['--max-len', str(int(evo_config['max_program_length']))]
        if 'elitism' in evo_config:
            cmd += ['--elitism', str(float(evo_config['elitism']))]
        if evo_config.get('no_parallel'):
            cmd += ['--no-parallel']
        if init_genomes and os.path.exists(init_genomes):
            cmd += ['--init-genomes', init_genomes]
        if tasks_yaml:
            cmd += ['--tasks_yaml', tasks_yaml]

        print(f"    Command: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=repo_root, check=True)

        # Locate outputs produced by evolve_with_cfg.py
        base_name = os.path.splitext(os.path.basename(grammar_path))[0]
        runs_dir = os.path.join(repo_root, 'outputs', 'runs')
        winners_src = os.path.join(runs_dir, f"{base_name}_best.json")

        if os.path.exists(winners_src):
            # Copy to task-specific file
            shutil.copyfile(winners_src, task_output_file)
            print(f"    Task {task_name} completed. Results: {task_output_file}")
            return task_output_file
        else:
            print(f"    Task {task_name} completed but no winners file found")
            return ""

    except Exception as e:
        print(f"    Task {task_name} failed: {e}")
        return ""


def run_evolution_round(grammar_path: str, output_dir: str, tasks: Dict[str, Any],
                       round_num: int, evo_config: Dict[str, Any], init_genomes: Optional[str] = None,
                       tasks_yaml: Optional[str] = None) -> str:
    """
    Backward compatibility wrapper for the old evolution round function.
    Now uses progressive learning by default.
    """
    return run_evolution_round_progressive(
        grammar_path=grammar_path,
        output_dir=output_dir,
        tasks=tasks,
        round_num=round_num,
        evo_config=evo_config,
        prev_winners_file=init_genomes,
        tasks_yaml=tasks_yaml
    )


def run_hierarchical_evolution_pipeline(cfg: Dict[str, Any]):
    """Run the hierarchical evolution pipeline using a config dict.
    Expected cfg keys:
      - task_suite: "sequences"
      - output_dir: path
      - max_iterations: int
      - evo: { population_size, generations, seed, step_limit, max_program_length, elitism, no_parallel }
      - base_grammar: path to initial grammar JSON
      - first_init_genomes: optional path to seed first round
      - mine: { enabled: bool, min_support, min_len } (optional) to run miner between rounds
    """
    print("Starting Hierarchical Evolution Pipeline")

    repo_root = cfg.get('base_path', os.path.dirname(os.path.dirname(__file__)))

    task_suite_name = cfg.get('task_suite', 'sequences')
    output_dir = cfg.get('output_dir', os.path.join(repo_root, 'outputs', 'hierarchical_evolution'))
    max_iterations = int(cfg.get('max_iterations', 1))
    evo_config = cfg.get('evo', {})  # expects keys: population_size, generations, seed, step_limit, max_symbols, elitism, no_parallel
    base_grammar_path = cfg.get('base_grammar', os.path.join(repo_root, 'motifs', 'minimal_cfg.json'))
    if not os.path.isabs(base_grammar_path):
        base_grammar_path = os.path.join(repo_root, base_grammar_path)
    mine_cfg = cfg.get('mine', {}) or {}

    print(f"Task suite: {task_suite_name}")
    print(f"Max iterations: {max_iterations}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repo_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load task suite
    task_suite = load_task_suite(task_suite_name, cfg.get('tasks_yaml'))

    # Initialize paths
    if not os.path.exists(base_grammar_path):
        raise FileNotFoundError(f"Base grammar not found: {base_grammar_path}")

    # Create round_0 with base grammar and initial winners seed
    round0_dir = os.path.join(output_dir, 'round_0')
    os.makedirs(round0_dir, exist_ok=True)
    round0_grammar = os.path.join(round0_dir, 'grammar.json')
    shutil.copyfile(base_grammar_path, round0_grammar)

    # Seed winners for round_0 from config.first_init_genomes if provided; else empty dict
    init_seed_path = cfg.get('first_init_genomes')
    if init_seed_path:
        if not os.path.isabs(init_seed_path):
            init_seed_path = os.path.join(repo_root, init_seed_path)
        shutil.copyfile(init_seed_path, os.path.join(round0_dir, 'winners.json'))
    else:
        with open(os.path.join(round0_dir, 'winners.json'), 'w') as f:
            json.dump({}, f)

    pipeline_history = []

    # Create hierarchical miner once for the entire session
    from core.hierarchical_miner import HierarchicalMiner, mine_lk_motifs, add_lk_motifs
    from core.miner_l1 import mine_l1
    from core.grammar_updater import add_l1_motifs, save_grammar, get_grammar_stats, remove_unused_motifs
    from core.miner_l2 import symbolize_corpus

    l1_config = {
        'min_support': mine_cfg.get('min_support', 3),
        'min_len': mine_cfg.get('min_len', 3),
        'motif_weight': mine_cfg.get('motif_weight', 0.1),
        'mdl_thresh': mine_cfg.get('l1_mdl_thresh', 2.0),
        'step_limit': mine_cfg.get('step_limit', 1000),
        'mdl_lambda': mine_cfg.get('mdl_lambda', 1.5),
        'max_linear_len': mine_cfg.get('max_linear_len', 6)
    }

    lk_config = {
        'min_support': mine_cfg.get('min_support', 3),
        'motif_weight': mine_cfg.get('motif_weight', 0.1),
        'mdl_thresh': mine_cfg.get('lk_mdl_thresh', 2.0),
        'min_n': mine_cfg.get('min_len', 3),  # Use min_len for consistency
        'max_n': mine_cfg.get('max_n', 5),
        'mdl_lambda': mine_cfg.get('mdl_lambda', 1.0)
    }

    # Create hierarchical miner with timestamped session directory
    miner = HierarchicalMiner(
        base_grammar_path=base_grammar_path,
        base_output_dir=output_dir,
        l1_config=l1_config,
        lk_config=lk_config,
        max_levels=mine_cfg.get('max_levels', 10)
    )

    print(f"[Pipeline] Created hierarchical mining session: {miner.session_dir}")

    # Create round_0 in the timestamped session directory with empty winners
    round_0_dir = os.path.join(miner.session_dir, 'round_0')
    os.makedirs(round_0_dir, exist_ok=True)

    # Copy base grammar to round_0
    round_0_grammar = os.path.join(round_0_dir, 'grammar.json')
    shutil.copyfile(base_grammar_path, round_0_grammar)

    # Create empty winners.json for round_0
    round_0_winners = os.path.join(round_0_dir, 'winners.json')
    with open(round_0_winners, 'w') as f:
        json.dump({}, f, indent=2)

    print(f"[Pipeline] Created round_0 in session: {round_0_dir}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*80}")
        print(f"PIPELINE ITERATION {iteration}/{max_iterations}")
        print(f"{'='*80}")

        # For iteration 1, use base grammar; for later iterations, use previous mining result
        if iteration == 1:
            grammar_in = base_grammar_path
            seed_winners = os.path.join(miner.session_dir, 'round_0', 'winners.json')
        else:
            # Use the grammar and winners from the previous round in the timestamped session
            prev_round = os.path.join(miner.session_dir, f'round_{iteration - 1}')
            grammar_in = os.path.join(prev_round, 'grammar_final.json')
            seed_winners = os.path.join(prev_round, 'winners.json')

        # Step 1: Run evolution with progressive learning within the round
        # Use the timestamped session directory for evolution output
        winners_file = run_evolution_round_progressive(
            grammar_path=grammar_in,
            output_dir=miner.session_dir,  # Use timestamped session directory
            tasks=task_suite['tasks'],
            round_num=iteration,
            evo_config=evo_config,
            prev_winners_file=seed_winners,
            tasks_yaml=task_suite.get('tasks_yaml'),
        )

        # Step 2: Multi-level hierarchical motif mining using the existing miner
        print(f"\n[Pipeline] Mining hierarchical motifs from winners (multi-level L1→L2→L3→...)...")

        # Load winners data
        with open(winners_file, 'r') as f:
            winners_data = json.load(f)

        # Update the miner's base grammar for this iteration
        with open(grammar_in, 'r') as f:
            miner.current_grammar = json.load(f)

        # Create round directory for this iteration
        miner.current_round_dir = os.path.join(miner.session_dir, f"round_{iteration}")
        os.makedirs(miner.current_round_dir, exist_ok=True)

        # Run single iteration of mining manually
        print(f"\n{'='*60}")
        print(f"MINING ITERATION {iteration}")
        print(f"Round directory: {miner.current_round_dir}")
        print(f"{'='*60}")

        iteration_start_stats = get_grammar_stats(miner.current_grammar)
        total_discovered = 0
        level_motifs_discovered = {}
        all_motif_info = {}

        # Phase 1: Iterative L1 motif mining
        print("\n[L1 Phase] Mining token-level motifs (iterative)...")
        l1_motifs, l1_motif_info = mine_l1_iterative(
            winners_data=winners_data,
            grammar=miner.current_grammar,
            l1_config=miner.l1_config
        )

        if l1_motifs:
            # Get current L1 motif count to determine names
            current_l1_count = len([k for k in miner.current_grammar.keys() if k.startswith('L1M')])

            # Add motifs to grammar
            miner.current_grammar = add_l1_motifs(
                grammar=miner.current_grammar,
                motifs=l1_motifs,
                motif_weight=miner.l1_config['motif_weight'],
                level=1
            )

            # Add motif names to the detailed info
            for i, motif_info in enumerate(l1_motif_info):
                motif_info['name'] = f"L1M{current_l1_count + i + 1}"

            total_discovered += len(l1_motifs)
            level_motifs_discovered[1] = len(l1_motifs)
            all_motif_info[1] = l1_motif_info
            print(f"[L1 Phase] Added {len(l1_motifs)} motifs")
        else:
            print("[L1 Phase] No L1 motifs discovered")
            level_motifs_discovered[1] = 0
            all_motif_info[1] = []

        # Always save L1 grammar and motifs
        l1_grammar_path = os.path.join(miner.current_round_dir, "grammar_L1.json")
        save_grammar(miner.current_grammar, l1_grammar_path)

        l1_motifs_path = os.path.join(miner.current_round_dir, "motifs_L1.json")
        with open(l1_motifs_path, 'w') as f:
            json.dump(all_motif_info[1], f, indent=2)

        print(f"[L1 Phase] Saved grammar: {l1_grammar_path}")
        print(f"[L1 Phase] Saved motifs: {l1_motifs_path}")

        # Phase 2+: Mine L2, L3, L4, ... motifs
        level = 2
        previous_motifs = set()  # Track motifs to detect duplicates
        while level <= miner.max_levels:
            print(f"\n[L{level} Phase] Mining level-{level} motifs...")

            lk_motifs, lk_motif_info = mine_lk_motifs(
                winners_data=winners_data,
                grammar=miner.current_grammar,
                level=level,
                min_support=miner.lk_config['min_support'],
                mdl_lambda=miner.lk_config['mdl_lambda'],
                mdl_thresh=miner.lk_config['mdl_thresh'],
                min_n=miner.lk_config['min_n'],
                max_n=miner.lk_config['max_n']
            )

            if lk_motifs:
                # Check for duplicate motifs (same patterns as previous levels)
                current_motifs = set(tuple(motif) for motif in lk_motifs)
                if current_motifs.issubset(previous_motifs):
                    print(f"[L{level} Phase] All motifs are duplicates from previous levels - stopping")
                    level_motifs_discovered[level] = 0
                    all_motif_info[level] = []
                    break

                # Get current motif count for this level to determine names
                current_level_count = len([k for k in miner.current_grammar.keys() if k.startswith(f'L{level}M')])

                miner.current_grammar = add_lk_motifs(
                    grammar=miner.current_grammar,
                    motifs=lk_motifs,
                    motif_weight=miner.lk_config['motif_weight'],
                    level=level
                )

                # Add motif names to the detailed info
                for i, motif_info in enumerate(lk_motif_info):
                    motif_info['name'] = f"L{level}M{current_level_count + i + 1}"

                total_discovered += len(lk_motifs)
                level_motifs_discovered[level] = len(lk_motifs)
                all_motif_info[level] = lk_motif_info
                print(f"[L{level} Phase] Added {len(lk_motifs)} motifs")

                # Update previous motifs for next iteration
                previous_motifs.update(current_motifs)
            else:
                print(f"[L{level} Phase] No L{level} motifs discovered - stopping level discovery")
                level_motifs_discovered[level] = 0
                all_motif_info[level] = lk_motif_info

            # Always save Lk grammar and motifs
            lk_grammar_path = os.path.join(miner.current_round_dir, f"grammar_L{level}.json")
            save_grammar(miner.current_grammar, lk_grammar_path)

            lk_motifs_path = os.path.join(miner.current_round_dir, f"motifs_L{level}.json")
            with open(lk_motifs_path, 'w') as f:
                json.dump(all_motif_info[level], f, indent=2)

            print(f"[L{level} Phase] Saved grammar: {lk_grammar_path}")
            print(f"[L{level} Phase] Saved motifs: {lk_motifs_path}")

            if lk_motifs:
                level += 1
            else:
                print(f"[L{level} Phase] No motifs found - stopping hierarchical discovery")
                break

        # Clean up unused motifs
        miner.current_grammar = remove_unused_motifs(miner.current_grammar)

        # Save final iteration grammar
        final_iteration_grammar_path = os.path.join(miner.current_round_dir, "grammar_final.json")
        save_grammar(miner.current_grammar, final_iteration_grammar_path)
        print(f"[Pipeline] Saved final iteration grammar: {final_iteration_grammar_path}")

        # Save all motifs summary
        all_motifs_path = os.path.join(miner.current_round_dir, "motifs_all.json")
        with open(all_motifs_path, 'w') as f:
            json.dump(all_motif_info, f, indent=2)
        print(f"[Pipeline] Saved all motifs summary: {all_motifs_path}")

        # Update miner history
        iteration_end_stats = get_grammar_stats(miner.current_grammar)
        history_entry = {
            'iteration': iteration,
            'l1_motifs': len(l1_motifs) if l1_motifs else 0,
            'total_motifs_before': iteration_start_stats['motif_count'],
            'total_motifs_after': iteration_end_stats['motif_count'],
            'motifs_added': iteration_end_stats['motif_count'] - iteration_start_stats['motif_count'],
            'max_level_reached': level - 1,
            'level_motifs': level_motifs_discovered.copy(),
            'round_dir': miner.current_round_dir
        }
        miner.history.append(history_entry)

        print(f"\n[Mining Iteration {iteration} Summary]")
        print(f"  L1 motifs: {len(l1_motifs) if l1_motifs else 0}")
        for lvl, count in level_motifs_discovered.items():
            if lvl > 1:
                print(f"  L{lvl} motifs: {count}")
        print(f"  Max level reached: L{level - 1}")
        print(f"  Total motifs: {iteration_start_stats['motif_count']} → {iteration_end_stats['motif_count']}")

        # Evolution results are already in the correct location (miner.current_round_dir)
        # No need to copy since both evolution and mining use the same round directory

        # The final grammar is in the miner's current round directory
        session_final_grammar = os.path.join(miner.current_round_dir, 'grammar_final.json')

        print(f"[Pipeline] Hierarchical mining completed")
        print(f"[Pipeline] Session directory: {miner.session_dir}")
        print(f"[Pipeline] Round directory: {miner.current_round_dir}")
        print(f"[Pipeline] Final grammar: {session_final_grammar}")

        current_grammar_path = session_final_grammar

        # Record iteration history
        iteration_history = {
            'iteration': iteration,
            'grammar_used': grammar_in,
            'winners_file': winners_file,
            'enriched_grammar': current_grammar_path,
            'hierarchical_mining_session': miner.session_dir,
            'hierarchical_mining_round': miner.current_round_dir,
            'mining_summary': miner.history[-1] if miner.history else None
        }
        pipeline_history.append(iteration_history)

    # Save pipeline summary
    summary_file = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'task_suite': task_suite_name,
            'max_iterations': max_iterations,
            'iterations_completed': len(pipeline_history),
            'final_grammar': current_grammar_path,
            'history': pipeline_history
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED")
    print(f"{'='*80}")
    print(f"Iterations completed: {len(pipeline_history)}")
    print(f"Final grammar: {current_grammar_path}")
    print(f"Pipeline summary: {summary_file}")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        if path.endswith(('.yml', '.yaml')):
            if yaml is None: raise RuntimeError("YAML config requested but PyYAML is not installed")
            return yaml.safe_load(f)
        return json.load(f)


def mine_l1_iterative(winners_data, grammar, l1_config):
    """
    Iteratively mine L1 motifs until complete primitive coverage.

    This ensures no primitive patterns leak into L2 mining by repeatedly:
    1. Mining L1 motifs from current primitive sequences
    2. Symbolizing programs with discovered L1 motifs
    3. Repeating until no new L1 motifs are found

    Returns:
        Tuple of (all_l1_motifs, all_l1_motif_info)
    """
    from core.miner_l1 import mine_l1
    from core.grammar_updater import add_l1_motifs
    from core.miner_l2 import symbolize_corpus

    print("[L1 Iterative] Starting iterative L1 motif mining...")

    all_l1_motifs = []
    all_l1_motif_info = []
    current_grammar = grammar.copy()
    round_num = 1
    max_rounds = 10  # Prevent infinite loops

    while round_num <= max_rounds:
        print(f"[L1 Iterative] Round {round_num}")

        # Mine L1 motifs from current state
        l1_motifs, l1_motif_info = mine_l1(
            winners_data=winners_data,
            step_limit=l1_config['step_limit'],
            min_support=l1_config['min_support'],
            mdl_lambda=l1_config['mdl_lambda'],
            mdl_thresh=l1_config['mdl_thresh'],
            max_linear_len=l1_config['max_linear_len'],
            min_len=l1_config['min_len']
        )

        if not l1_motifs:
            print(f"[L1 Iterative] No new L1 motifs found in round {round_num} - stopping")
            break

        print(f"[L1 Iterative] Round {round_num}: Found {len(l1_motifs)} new L1 motifs")

        # Add new motifs to grammar
        current_grammar = add_l1_motifs(
            grammar=current_grammar,
            motifs=l1_motifs,
            motif_weight=l1_config['motif_weight'],
            level=1
        )

        # Add to cumulative results
        all_l1_motifs.extend(l1_motifs)
        all_l1_motif_info.extend(l1_motif_info)

        # Update winners_data by symbolizing with current L1 motifs
        # This removes discovered patterns from future mining rounds
        symbolized_winners = {}
        for task_name, task_data in winners_data.items():
            solutions = task_data.get('solutions', [])

            # Symbolize each solution
            winners_formatted = {"temp": {"solutions": solutions}}
            symbol_sequences = symbolize_corpus(winners_formatted, current_grammar)

            # Convert back to string programs (only primitives remain)
            new_solutions = []
            for seq in symbol_sequences:
                # Only keep primitive tokens, skip L1 motifs
                primitives = [token for token in seq if token in ['+', '-', '>', '<', '[', ']']]
                if primitives:  # Only add if there are remaining primitives
                    new_solutions.append(''.join(primitives))

            if new_solutions:
                symbolized_winners[task_name] = {"solutions": new_solutions}

        # Update winners_data for next round
        winners_data = symbolized_winners

        # If no primitive sequences remain, we're done
        if not winners_data:
            print(f"[L1 Iterative] No primitive sequences remain - complete coverage achieved")
            break

        round_num += 1

    print(f"[L1 Iterative] Completed after {round_num-1} rounds")
    print(f"[L1 Iterative] Total L1 motifs discovered: {len(all_l1_motifs)}")

    return all_l1_motifs, all_l1_motif_info


def main():
    """Main entry point for the hierarchical evolution experiment."""
    parser = argparse.ArgumentParser(description="Hierarchical motif-based evolution pipeline")
    parser.add_argument('config', help='Path to JSON or YAML config file')
    args = parser.parse_args()
    cfg = _load_config(args.config)

    try:
        run_hierarchical_evolution_pipeline(cfg)

    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()