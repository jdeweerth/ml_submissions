import hiive.mdptoolbox.mdp
import hiive.mdptoolbox as mdptoolbox
import gym
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os


root_path = os.getcwd()

actions = {
    0: '\u2190',  # LEFT
    1: '\u2193',  # DOWN
    2: '\u2192',  # RIGHT
    3: '\u2191'  # UP
}


def policy_eval(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                # print('Reached the goal after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("Fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    print('Reached goal in average of {:.0f} steps'.format(np.mean(steps_list)))
    print('Fell in the hole {:.2f} % of the time'.format((misses/episodes) * 100))
    print('----------------------------------------------')


def policy_viz(policy):
    viz_policy = list(policy)
    states = len(viz_policy)
    side = int(np.sqrt(states))
    for i in range(states):
        viz_policy[i] = actions[viz_policy[i]]

    for i in range(states):
        if i % side == 0 and i != 0:
            print(" ".join(viz_policy[i - side:i]))
        elif i == states - 1:
            print(" ".join(viz_policy[i - (side - 1):i + 1]))


def get_fl():
    env = gym.make('FrozenLake-v0')
    env.render()
    old_P = env.P
    states = len(old_P)
    P = np.zeros((4, states, states))
    R = np.zeros((states, 4))
    for action in range(4):
        for state in range(len(old_P)):
            # print(old_P[state][action][2])
            # R[state, action] = old_P[state][action][2]
            state_options = ((old_P[state])[action])
            for opt in state_options:
                next_prob = opt[0]
                next_state = opt[1]
                reward = opt[2]
                P[action, state, next_state] += next_prob
                R[state, action] += reward * next_prob
    return P, R, env



print("----------------------------------------------")
print("Value Iteration")
print("----------------------------------------------")
t = time()
P, R, env = get_fl()
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9999, max_iter=100000)
run_stats = vi.run()
policy = vi.policy

policy_viz(policy)
policy_eval(env, policy)
print("Value Iteration time: {}".format(time() - t))
print("Total Iterations: {}".format(run_stats[len(run_stats) - 1]["Iteration"]))
print()

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Mean Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("mean value", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
mean_val = [i["Mean V"] for i in run_stats]

ax.plot(iterations, mean_val, color='r')

fig.savefig(root_path + "/plots/frozenlake/vi_mean_val.png")

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Max Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("max value", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
max_val = [i["Max V"] for i in run_stats]

ax.plot(iterations, max_val, color='r')

fig.savefig(root_path + "/plots/frozenlake/vi_max_val.png")

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Error vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
errors = [i["Error"] for i in run_stats]

ax.plot(iterations, errors, color='r')

fig.savefig(root_path + "/plots/frozenlake/vi_error.png")

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Error vs Time")
ax.set_xlabel("time", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

times = [i["Time"] for i in run_stats]
errors = [i["Error"] for i in run_stats]

ax.plot(times, errors, color='r')

fig.savefig(root_path + "/plots/frozenlake/vi_time.png")



print("----------------------------------------------")
print("Policy Iteration")
print("----------------------------------------------")
t = time()
P, R, env = get_fl()
pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.999, max_iter=10000, eval_type=0)
run_stats = pi.run()
policy = pi.policy

policy_viz(policy)
policy_eval(env, policy)
print("Policy Iteration time: {}".format(time() - t))
print("Total Iterations: {}".format(run_stats[len(run_stats) - 1]["Iteration"]))
print()

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Mean Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("mean value", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
mean_val = [i["Mean V"] for i in run_stats]

ax.plot(iterations, mean_val, color='r')

fig.savefig(root_path + "/plots/frozenlake/pi_mean_val.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Max Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("max value", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
max_val = [i["Max V"] for i in run_stats]

ax.plot(iterations, max_val, color='r')

fig.savefig(root_path + "/plots/frozenlake/pi_max_val.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Error vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
errors = [i["Error"] for i in run_stats]

ax.plot(iterations, errors, color='r')

fig.savefig(root_path + "/plots/frozenlake/pi_error.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Policy Change vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("policy change", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
p_change = [i["nd"] for i in run_stats]

ax.plot(iterations, p_change, color='r')

ax.legend()

fig.savefig(root_path + "/plots/frozenlake/pi_policy_change.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Error vs Time")
ax.set_xlabel("time", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

times = [i["Time"] for i in run_stats]
errors = [i["Error"] for i in run_stats]

ax.plot(times, errors, color='r')

fig.savefig(root_path + "/plots/frozenlake/pi_time.png")



print("----------------------------------------------")
print("Q Learning")
print("----------------------------------------------")
t = time()
P, R, env = get_fl()
ql = mdptoolbox.mdp.QLearning(P, R, 0.9999,
                              n_iter=100000000,
                              alpha_decay=1.0,
                              alpha=0.1,
                              epsilon_decay=0.99999)
run_stats = ql.run(init_state=0, episode_len=1000)

print(ql.Q)

state_count = np.zeros(64)
for run in run_stats:
    state_count[run["State"]] += 1
np.set_printoptions(suppress=True)
print(state_count)
np.set_printoptions(suppress=False)

policy = ql.policy

policy_viz(policy)
policy_eval(env, policy)
print("Q Learning time: {}".format(time() - t))
print("Total Iterations: {}".format(run_stats[len(run_stats) - 1]["Iteration"]))
print("alpha: {}".format(run_stats[len(run_stats) - 1]["Alpha"]))
print("epsilon: {}".format(run_stats[len(run_stats) - 1]["Epsilon"]))
print()

fig, ax = plt.subplots()

ax.set_title("QLearning - Mean Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("mean value", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
mean_val = [i["Mean V"] for i in run_stats]

ax.plot(iterations, mean_val, color='r')

fig.savefig(root_path + "/plots/frozenlake/ql_mean_val.png")

fig, ax = plt.subplots()

ax.set_title("QLearning - Max Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("max value", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
max_val = [i["Max V"] for i in run_stats]

ax.plot(iterations, max_val, color='r')

fig.savefig(root_path + "/plots/frozenlake/ql_max_val.png")

fig, ax = plt.subplots()

ax.set_title("QLearning - Error vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

iterations = [i["Iteration"] for i in run_stats]
errors = [i["Error"] for i in run_stats]

ax.plot(iterations, errors, color='r')

fig.savefig(root_path + "/plots/frozenlake/ql_error.png")

fig, ax = plt.subplots()

ax.set_title("QLearning - Error vs Time")
ax.set_xlabel("time", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

times = [i["Time"] for i in run_stats]
errors = [i["Error"] for i in run_stats]

ax.plot(times, errors, color='r')

fig.savefig(root_path + "/plots/frozenlake/ql_time.png")
