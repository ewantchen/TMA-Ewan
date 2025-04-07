#import gymnasium
#import gymnasium_env
#import my_library   # contains "my_environment"

#import ale_py
#import shimmy

from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

#env =  gymnasium.make("gym_examples.my_library")

#obs = env.reset()
#print("obs = ", obs)

