import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

class recorder:
    def __init__(self, env, name, trigger=None):
        self.name = name
        self.env = env
        if trigger is not None:
            self.trigger = trigger
        else: 
            self.trigger = lambda x:True

    def render_video(self, num_eval_episodes):
        env = RecordVideo(self.env, video_folder="video_renders", name_prefix=self.name, episode_trigger=self.trigger)
        env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

        for episode_num in range(num_eval_episodes):
            obs, info = env.reset()

            episode_over = False
            while not episode_over:
                action = env.action_space.sample()  # replace with actual agent
                obs, reward, terminated, truncated, info = env.step(action)

                episode_over = terminated or truncated
        env.close()
        return env