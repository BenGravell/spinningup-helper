import gym

env = gym.make('CartPole-v0')  # change "CartPole-V0" to test different Gym environments
env.reset()
for _ in range(200):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
