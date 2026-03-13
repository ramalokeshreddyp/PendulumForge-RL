import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from environment import DoublePendulumEnv
import imageio

def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file')
    parser.add_argument('--render_mode', type=str, default='human', choices=['human', 'rgb_array'],
                        help='Render mode')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--save_gif', type=str, default=None,
                        help='Path to save evaluation as GIF')
    
    args = parser.parse_args()
    
    # Create environment
    env = DoublePendulumEnv(render_mode=args.render_mode)
    
    # Load model
    model = PPO.load(args.model_path, env=env)
    
    frames = []
    
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            frame = env.render()
            if args.save_gif and args.render_mode == 'rgb_array':
                frames.append(frame)
                
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
        
    if args.save_gif and frames:
        print(f"Compiling GIF with {len(frames)} frames...")
        os.makedirs(os.path.dirname(args.save_gif), exist_ok=True)
        imageio.mimsave(args.save_gif, frames, fps=60)
        print(f"Saved evaluation to {args.save_gif}")
        
    env.close()

if __name__ == '__main__':
    evaluate()
