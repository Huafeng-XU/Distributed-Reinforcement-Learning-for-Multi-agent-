from ic3net_envs.traffic_junction_env import TrafficJunctionEnv
import argparse
import sys
import signal

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Example GCCNet environment random agent')
    # parser.add_argument('--nagents', type=int, default=2, help="Number of agents")
    parser.add_argument('--nagents', type=int, default=10, help="Number of agents")
    parser.add_argument('--display', action="store_true", default=False,
                        help="Use to display environment")

    env = TrafficJunctionEnv()
    #env.init_curses()
    env.init_args(parser)

    args = parser.parse_args()

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.exit_render()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    env.multi_agent_init(args)

    agent = RandomAgent(env.action_space)
    episodes = 0

    while episodes < 50:
        print('Im here')
        obs = env.reset()
        print("222222222222222")
        done = False
        while not done:
            actions = []

            print("333333333333333333")
            for _ in range(args.nagents):
                action = agent.act()
                actions.append(action)
            print("44444444444444444444")
            obs, reward, done, info = env.step(actions)
            print("555555555555555555555555555")

            if args.display:
                env.render()
        episodes += 1
        print(reward)

    env.close()
