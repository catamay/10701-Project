import gymnasium as gym
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

class Recorder:
    def __init__(self, env, name, trigger=None):
        self.env = env
        self.name = name
        self.trigger = trigger if trigger is not None else lambda x: True

    def render_video(self, num_eval_episodes, agent=None):
        env = RecordEpisodeStatistics(self.env, buffer_length=num_eval_episodes)
        env = RecordVideo(env, video_folder="video_renders", name_prefix=self.name, episode_trigger=self.trigger)

        for episode_num in range(num_eval_episodes):
            obs, info = env.reset()
            done = False

            while not done:
                if agent is not None:
                    state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    with torch.no_grad():
                        action = agent.select_action(state).cpu().numpy()
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        env.close()
        return env
