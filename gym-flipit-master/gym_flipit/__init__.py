from gym.envs.registration import register

register(
    id='Flipit-v0',
    entry_point='gym_flipit.envs:FlipitEnv',
)

register(
    id='CheatFlipit-v0',
    entry_point='gym_flipit.envs:CheatFlipitEnv',
)

register(
    id='MaritimeNontraditional-v0',
    entry_point='gym_flipit.envs:MaritimeNontraditionalEnv',
)

register(
    id='ResourceConstraintFlipit-v0',
    entry_point='gym_flipit.envs:ResourceConstraintFlipitEnv',
)

register(
    id='MaritimeDRL-v0',
    entry_point='gym_flipit.envs:MaritimeDRLEnv',
    max_episode_steps=500,
)

register(
    id='MultiAgentMaritimeDRL-v0',
    entry_point='gym_flipit.envs:MultiAgentMaritimeDRLEnv',
    max_episode_steps=500,
)
