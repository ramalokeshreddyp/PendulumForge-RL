# Project Test Commands

Use these commands from the repository root to test the project end to end.

## 1) Build the Docker images

docker compose build

## 2) Run short baseline training smoke test

docker compose run --rm app python train.py --reward_type baseline --timesteps 10 --save_path models/ppo_baseline_test.zip

## 3) Run short shaped training smoke test

docker compose run --rm app python train.py --reward_type shaped --timesteps 10 --save_path models/ppo_shaped_test.zip

## 4) Evaluate baseline model and generate initial GIF

docker compose run --rm app python evaluate.py --model_path models/ppo_baseline_test.zip --render_mode rgb_array --episodes 1 --save_gif media/agent_initial.gif

## 5) Evaluate shaped model and generate final GIF

docker compose run --rm app python evaluate.py --model_path models/ppo_shaped_test.zip --render_mode rgb_array --episodes 2 --save_gif media/agent_final.gif

## 6) Generate reward comparison plot

docker compose run --rm plot

## 7) Verify required artifacts exist

Get-ChildItem models,logs,media,reward_comparison.png,README.md,.env.example,Dockerfile,docker-compose.yml | Select-Object FullName,Length | Format-Table -AutoSize

## 8) Verify metrics columns (timesteps, mean_reward)

python -c "import pandas as pd; df=pd.read_csv('logs/metrics_shaped.csv'); print(df.columns.tolist())"

## 9) Optional: quick environment contract check in Docker

docker compose run --rm app python -c "import numpy as np, pymunk; from environment import DoublePendulumEnv as E; env=E(); assert isinstance(env.space,pymunk.Space); assert env.observation_space.shape==(6,); assert env.action_space.shape==(1,); print('contract-ok')"
