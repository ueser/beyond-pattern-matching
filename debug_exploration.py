#!/usr/bin/env python3
"""
Debug script to analyze why the RL agent stops exploring.
"""

import sys
sys.path.append('primitive-evolution/blog_demos')

import torch
import numpy as np
from bf_rl_trainer import BFPPOTrainer, TrainingConfig

def debug_exploration():
    """Debug exploration behavior."""
    print("🔍 DEBUGGING EXPLORATION ISSUES")
    print("=" * 50)
    
    config = TrainingConfig(
        max_length=20,
        eval_inputs=[2, 3, 5, 7],
        target_function="2*x",
        
        # Dynamic sampling
        input_pool=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_sample_inputs=4,
        
        # Training
        n_episodes_per_update=16,
        learning_rate=1e-3,
        entropy_coef=0.1,  # High entropy for exploration
        
        # Simple curriculum
        curriculum_phases=[
            {"max_length": 20, "updates": 20},
        ],
        
        log_interval=5,
        use_wandb=False
    )
    
    trainer = BFPPOTrainer(config)
    
    print("🎯 Testing exploration over 20 updates...")
    print(f"Entropy coefficient: {config.entropy_coef}")
    
    # Track exploration metrics
    unique_programs = set()
    reward_history = []
    program_history = []
    entropy_history = []
    
    for update in range(20):
        trainer.update_count = update
        
        # Sample new inputs
        trainer._sample_new_inputs()
        
        # Collect rollouts and analyze
        all_steps, rollout_metrics = trainer.collect_rollouts()
        
        # Track unique programs from this update
        update_programs = set()
        for step in all_steps:
            if step.done:
                program = trainer.env.tokens_to_program(trainer.env.state.tokens_prefix)
                update_programs.add(program)
                unique_programs.add(program)
        
        program_history.append(update_programs)
        reward_history.append(rollout_metrics.get('max_episode_reward', 0))
        
        # Analyze policy entropy by sampling
        policy_entropy = analyze_policy_entropy(trainer)
        entropy_history.append(policy_entropy)
        
        # Training step
        training_metrics = trainer.train_step(all_steps)
        
        # Track best solution
        if rollout_metrics.get('max_episode_reward', float('-inf')) > trainer.best_reward:
            trainer.best_reward = rollout_metrics['max_episode_reward']
            trainer.best_program = rollout_metrics.get('best_program', '')
        
        print(f"Update {update:2d}: "
              f"Reward={trainer.best_reward:.3f}, "
              f"Programs={len(update_programs)}, "
              f"Unique={len(unique_programs)}, "
              f"Entropy={policy_entropy:.3f}, "
              f"Best='{trainer.best_program}'")
    
    print(f"\n📊 EXPLORATION ANALYSIS:")
    print(f"Total unique programs generated: {len(unique_programs)}")
    print(f"Final entropy: {entropy_history[-1]:.3f}")
    print(f"Entropy trend: {entropy_history[0]:.3f} → {entropy_history[-1]:.3f}")
    
    # Check if exploration decreased over time
    recent_exploration = len(program_history[-5:][0]) if len(program_history) >= 5 else 0
    early_exploration = len(program_history[:5][0]) if len(program_history) >= 5 else 0
    
    print(f"Early exploration (programs per update): {early_exploration}")
    print(f"Recent exploration (programs per update): {recent_exploration}")
    
    if recent_exploration < early_exploration * 0.5:
        print("⚠️  EXPLORATION DECAY DETECTED!")
    
    if entropy_history[-1] < entropy_history[0] * 0.5:
        print("⚠️  POLICY ENTROPY COLLAPSE DETECTED!")
    
    # Analyze reward stagnation
    if len(reward_history) >= 10:
        recent_rewards = reward_history[-10:]
        if max(recent_rewards) - min(recent_rewards) < 0.01:
            print("⚠️  REWARD STAGNATION DETECTED!")
    
    print(f"\nUnique programs found:")
    for i, program in enumerate(sorted(unique_programs)[:10]):
        print(f"  {i+1:2d}. '{program}'")
    if len(unique_programs) > 10:
        print(f"  ... and {len(unique_programs)-10} more")
    
    return len(unique_programs) > 10  # Good exploration = many unique programs

def analyze_policy_entropy(trainer):
    """Measure policy entropy by sampling actions."""
    trainer.policy.eval()
    
    # Sample from initial state
    state = trainer.env.reset()
    tokens = torch.zeros(1, 1, dtype=torch.long, device=trainer.device)
    state_features = torch.tensor(
        [list(state.to_features(trainer.env.max_length).values())], 
        dtype=torch.float, device=trainer.device
    )
    action_mask = torch.tensor(
        np.array([trainer.env.get_action_mask()]), 
        dtype=torch.bool, device=trainer.device
    )
    
    with torch.no_grad():
        # Get action probabilities
        _, _, _, action_probs = trainer.policy.act_with_probs(
            tokens, state_features, action_mask, temperature=1.0
        )
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()
    
    trainer.policy.train()
    return entropy

if __name__ == "__main__":
    debug_exploration()