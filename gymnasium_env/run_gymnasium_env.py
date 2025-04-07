# run_gymnasium_env.py

import gymnasium
import gymnasium_env
import time

env = gymnasium.make('gymnasium_env:gymnasium_env/GridWorld', render_mode="human")
#env = gymnasium.make('gymnasium_env:gymnasium_env/GridWorld', render_mode="rgb_array")

obs = env.reset()

obs, info = env.reset()

for _ in range(50):  # Run for 50 steps
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        obs, info = env.reset()
    
    time.sleep(0.1)  # Pause so you can see the moves

print(obs, env)

env.close()