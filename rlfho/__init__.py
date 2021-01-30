from gym.envs.registration import register

register(
    id='rlfho-v0',
    entry_point='rlfho.envs:RlfhoEnv',
)
register(
    id='rlfho-extrahard-v0',
    entry_point='rlfho.envs:RlfhoExtraHardEnv',
)