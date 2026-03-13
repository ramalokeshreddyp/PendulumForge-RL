import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_results():
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--output', type=str, default='reward_comparison.png', help='Output filename')
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    for reward_type in ['baseline', 'shaped']:
        log_file = os.path.join(args.log_dir, f'monitor_{reward_type}.csv')
        if os.path.exists(log_file):
            # Skip the first line (header comment)
            df = pd.read_csv(log_file, skiprows=1)
            
            # Calculate cumulative timesteps
            df['cumulative_timesteps'] = df['l'].cumsum()
            
            # Smooth the rewards
            df['smoothed_reward'] = df['r'].rolling(window=10).mean()
            
            plt.plot(df['cumulative_timesteps'], df['smoothed_reward'], label=f'{reward_type.capitalize()} Reward')

    plt.title('Learning Curves: Baseline vs Shaped Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == '__main__':
    plot_results()
