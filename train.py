import argparse
import csv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment import DoublePendulumEnv


class MetricsLoggerCallback(BaseCallback):
    def __init__(self, output_csv: str, log_every_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.output_csv = output_csv
        self.log_every_steps = log_every_steps
        self._last_logged_step = 0

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mean_reward"])

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_logged_step < self.log_every_steps:
            return True

        self._last_logged_step = self.num_timesteps
        if len(self.model.ep_info_buffer) == 0:
            return True

        mean_reward = sum(item["r"] for item in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, float(mean_reward)])

        return True

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
    parser.add_argument('--tensorboard_log', type=str, default=None,
                        help='Directory for TensorBoard logs (optional)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create environment
    env = DoublePendulumEnv(reward_type=args.reward_type, legacy_api=False)
    env = Monitor(env, os.path.join(args.log_dir, f'monitor_{args.reward_type}.csv'))
    metrics_csv = os.path.join(args.log_dir, f"metrics_{args.reward_type}.csv")
    
    # Initialize PPO agent
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=args.tensorboard_log)
    
    print(f"Starting training with reward_type={args.reward_type} for {args.timesteps} steps...")
    
    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=MetricsLoggerCallback(metrics_csv, log_every_steps=250)
    )
    
    # Save model
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")
    
    env.close()

if __name__ == '__main__':
    train()
