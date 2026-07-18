import ml_collections
from mujoco_playground._src import mjx_env
import numpy as np

ALL_ENVS = ("MyCustomEnv",)
class MjxEnv:
    def __init__(self, config=None):
        pass

class MyCustomEnv(MjxEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self.state = 0.0

    @property
    def action_size(self):
        return 1

    @property
    def mj_model(self):
        return None

    @property
    def mjx_model(self):
        return None

    @property
    def xml_path(self):
        return None

    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        self.state += float(action[0])
        obs = np.array([self.state], dtype=np.float32)
        reward = -abs(self.state)
        done = abs(self.state) > 10
        info = {}
        return obs, reward, done, info
def load(env_name: str, config=None, config_overrides=None) -> mjx_env.MjxEnv:
        if env_name != "MyCustomEnv":
            raise ValueError(f"Unknown env: {env_name}")

        if config is None:
            config = get_default_config(env_name)
        if config_overrides:
            config.update(config_overrides)

        return MyCustomEnv()
    
def get_domain_randomizer(env_name: str):
    if env_name == "MyCustomEnv":
        return None
    return None
    
def get_default_config(env_name: str) -> ml_collections.ConfigDict:
    if env_name == "MyCustomEnv":
        cfg = ml_collections.ConfigDict()
        cfg.obs_dim = 1
        cfg.action_dim = 1
        return cfg
    raise ValueError(f"Unknown env: {env_name}")

if __name__ == "__main__":
    # Instantiate environment
    env = MyCustomEnv()

    # Reset environment and print initial observation
    obs = env.reset()
    print("Initial observation:", obs)

    # Step through the environment
    for i in range(15):
        action = [1.0]  # test action
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1} | obs: {obs}, reward: {reward}, done: {done}")
        if done:
            print("Environment reached terminal state")
            break

    # Test loader function
    try:
        loaded_env = load("MyCustomEnv")
        obs = loaded_env.reset()
        print("Loaded environment initial obs:", obs)
        print("Loader test passed!")
    except Exception as e:
        print("Loader test failed:", e)

    # Simple assertions for sanity check
    assert isinstance(obs, (list, np.ndarray)), "Observation must be array-like"
    assert hasattr(env, "step"), "Environment must have step() method"
    print("Sanity checks passed!")
