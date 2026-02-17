#!/usr/bin/env python3
"""
Example: How to use Stable Baselines3 models as experts in IQ-Learn

This shows you exactly how to generate expert demonstrations using
pre-trained models from rl-baselines3-zoo.
"""

def example_breakout():
    """Generate expert demos for Breakout using pre-trained DQN"""
    print("Example 1: Breakout with DQN expert")
    print("Command:")
    print("python expert_generation.py \\")
    print("  env=breakout \\")
    print("  eval.use_baselines=true \\")
    print("  expert.demos=20 \\")
    print("  eval.threshold=21")
    print()

def example_cartpole():
    """Generate expert demos for CartPole using pre-trained PPO"""
    print("Example 2: CartPole with PPO expert")  
    print("Command:")
    print("python expert_generation.py \\")
    print("  env=cartpole \\")
    print("  eval.use_baselines=true \\")
    print("  expert.demos=10 \\")
    print("  eval.threshold=195")
    print()

def example_custom():
    """Show how to specify a specific algorithm"""
    print("Example 3: Custom algorithm selection")
    print("If you want to use a specific algorithm (not auto-detected):")
    print()
    print("# Modify expert_generation.py line 34 to:")
    print("agent = BaselinesExpert(args.env.name, ")
    print("                       folder='rl-baselines3-zoo/rl-trained-agents',")
    print("                       algorithm='dqn')  # Force DQN")
    print()

def show_available_models():
    """Show what models are available"""
    print("Available models for common environments:")
    print("=" * 50)
    
    models = {
        'BreakoutNoFrameskip-v4': ['a2c', 'ppo', 'dqn', 'qrdqn'],
        'PongNoFrameskip-v4': ['a2c', 'ppo', 'dqn', 'qrdqn'],
        'CartPole-v1': ['a2c', 'trpo', 'ppo', 'dqn', 'ars', 'qrdqn']
    }
    
    for env, algos in models.items():
        print(f"{env}:")
        for algo in algos:
            print(f"  - {algo.upper()}")
        print()

def main():
    print("🚀 Using Stable Baselines3 Models as IQ-Learn Experts")
    print("=" * 60)
    print()
    
    show_available_models()
    
    print("Usage Examples:")
    print("=" * 20)
    example_breakout()
    example_cartpole() 
    example_custom()
    
    print("Benefits of using pre-trained experts:")
    print("✓ No need to train your own expert policies")
    print("✓ High-quality demonstrations from state-of-the-art algorithms")
    print("✓ Consistent performance across different environments")
    print("✓ Save time and computational resources")
    print()
    
    print("Next steps:")
    print("1. Install stable-baselines3: pip install stable-baselines3")
    print("2. Run one of the example commands above")
    print("3. The generated expert demos will be saved in experts/ folder")
    print("4. Use these demos to train your IQ-Learn agent!")

if __name__ == "__main__":
    main()