import pandas as pd
from src.rl_env import FantasyEnv


def test_reset_and_step():
    df = pd.read_csv("tests/data/dummy_weekly.csv")
    env = FantasyEnv(df)

    obs, info = env.reset()
    assert obs.shape == (6,)

    # take one random valid action
    obs2, reward, term, trunc, info = env.step(0)
    assert obs2.shape == (6,)
    assert isinstance(reward, float)
    assert term is False
    assert trunc is False
