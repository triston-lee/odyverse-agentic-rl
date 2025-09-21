import argparse, torch
from verl_playground.envs.cartpole import make_env
from verl_playground.algos.dqn import DQN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    from verl_playground.utils.config import load_config
    cfg = load_config(args.config)

    env = make_env(cfg["env"]["name"], seed=cfg["seed"] + 7)
    algo = DQN(env.observation_space, env.action_space, cfg)
    algo.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    ret = algo.evaluate(env, episodes=10)
    print(f"Average return over 10 episodes: {ret:.2f}")


if __name__ == "__main__":
    main()
