import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import DoublePendulumEnv

def train():
    parser = argparse.ArgumentParser(description='Train PPO agent on DoublePendulumEnv')
    parser.add_argument('--reward_type', type=str, default='shaped', choices=['baseline', 'shaped'],
                        help='Type of reward function to use')
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Total number of timesteps to train')
    parser.add_argument('--save_path', type=str, default='models/ppo_double_pendulum',
                        help='Path to save the trained model')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create environment
    env = DoublePendulumEnv(reward_type=args.reward_type)
    env = Monitor(env, os.path.join(args.log_dir, f'monitor_{args.reward_type}.csv'))
    
    # Initialize PPO agent
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=args.log_dir)
    
    print(f"Starting training with reward_type={args.reward_type} for {args.timesteps} steps...")
    
    # Train
    model.learn(total_timesteps=args.timesteps)
    
    # Save model
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")
    
    env.close()

if __name__ == '__main__':
    train()
