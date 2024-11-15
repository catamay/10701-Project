import gymnasium as gym
import recorder as rc

env = gym.make("HalfCheetah-v5", render_mode="rgb_array",max_episode_steps=1000)

recorder = rc.recorder(env, "cheetah")

record_env = recorder.render_video(1)



print(f'Episode time taken: {record_env.time_queue}')
print(f'Episode total rewards: {record_env.return_queue}')
print(f'Episode lengths: {record_env.length_queue}')