from gymnasium.envs.registration import register

register(
    id='MaritimeCheatAttention-v0',
    entry_point='gym_flipit.envs:MaritimeCheatAttentionEnv',
    max_episode_steps=500,
)
