import hiive.mdptoolbox.example, hiive.mdptoolbox.mdp
import hiive.mdptoolbox as mdptoolbox
from time import time
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

root_path = os.getcwd()

file_name = "forest_stats" + str(datetime.now()) + ".txt"

with open(root_path + "/plots/forest/" + file_name, "w") as text_file:
        text_file.write("")

print("----------------------------------------------")
print("Value Iteration")
print("----------------------------------------------")

runs = []
state_opts = [200, 400, 800, 1600]
for states in state_opts:
    t = time()
    P, R = mdptoolbox.example.forest(S=states, r1=states/10, r2=states/20, p=0.02)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9999, max_iter=states)
    run_stats = vi.run()
    policy = vi.policy

    with open(root_path + "/plots/forest/" + file_name, "a") as text_file:
        text_file.write("\n-----------------------------------------------------------------------\n")
        text_file.write("Value iteration time for forest with {} states: {}\n".format(states, (time() - t)))
        text_file.write(str(policy))

    # if states == 400:
    #     print(policy)
    # print("Total Iterations: {}".format(run_stats[len(run_stats) - 1]["Iteration"]))
    # print("Value Iteration time: {}".format(time() - t))
    # print()
    runs.append(run_stats)

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Mean Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("mean value", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    mean_val = [i["Mean V"] for i in runs[i]]

    ax.plot(iterations, mean_val, label=state_opts[i])

ax.legend()
fig.savefig(root_path + "/plots/forest/vi_mean_val.png")

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Max Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("max value", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    max_val = [i["Max V"] for i in runs[i]]

    ax.plot(iterations, max_val, label=state_opts[i])

ax.legend()
fig.savefig(root_path + "/plots/forest/vi_max_val.png")

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Error vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    errors = [i["Error"] for i in runs[i]]

    ax.plot(iterations, errors, label=state_opts[i])

ax.legend()
fig.savefig(root_path + "/plots/forest/vi_error.png")

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Error vs Time")
ax.set_xlabel("time", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

for i in range(len(runs)):
    times = [i["Time"] for i in runs[i]]
    errors = [i["Error"] for i in runs[i]]

    ax.plot(times, errors, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/vi_error_time.png")

fig, ax = plt.subplots()

ax.set_title("Value Iteration - Time vs States")
ax.set_xlabel("number of states", fontweight='bold')
ax.set_ylabel("convergence time", fontweight='bold')

times = []
for r in runs:
    times.append(r[len(r) - 1]["Time"])

ax.plot(state_opts, times)

fig.savefig(root_path + "/plots/forest/vi_time.png")


print("----------------------------------------------")
print("Policy Iteration")
print("----------------------------------------------")

runs = []
state_opts = [200, 400, 800, 1600]
for states in state_opts:
    t = time()
    P, R = mdptoolbox.example.forest(S=states, r1=states/10, r2=states/20, p=0.02)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9999, max_iter=states, eval_type=0)
    run_stats = pi.run()
    policy = pi.policy

    with open(root_path + "/plots/forest/" + file_name, "a") as text_file:
        text_file.write("\n-----------------------------------------------------------------------\n")
        text_file.write("Policy iteration time for forest with {} states: {}\n".format(states, (time() - t)))
        text_file.write(str(policy))

    # if states == 400:
    #     print(policy)
    # print("Total Iterations: {}".format(run_stats[len(run_stats) - 1]["Iteration"]))
    # print("Value Iteration time: {}".format(time() - t))
    # print()
    runs.append(run_stats)


fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Mean Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("mean value", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    mean_val = [i["Mean V"] for i in runs[i]]

    ax.plot(iterations, mean_val, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/pi_mean_val.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Max Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("max value", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    max_val = [i["Max V"] for i in runs[i]]

    ax.plot(iterations, max_val, label=state_opts[i])

ax.legend()
fig.savefig(root_path + "/plots/forest/pi_max_val.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Error vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    errors = [i["Error"] for i in runs[i]]

    ax.plot(iterations, errors, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/pi_error.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Policy Change vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("policy change", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    p_change = [i["nd"] for i in runs[i]]

    ax.plot(iterations, p_change, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/pi_policy_change.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Error vs Time")
ax.set_xlabel("time", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

for i in range(len(runs)):
    times = [i["Time"] for i in runs[i]]
    errors = [i["Error"] for i in runs[i]]

    ax.plot(times, errors, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/pi_error_time.png")

fig, ax = plt.subplots()

ax.set_title("Policy Iteration - Time vs States")
ax.set_xlabel("number of states", fontweight='bold')
ax.set_ylabel("convergence time", fontweight='bold')

times = []
for r in runs:
    times.append(r[len(r) - 1]["Time"])

ax.plot(state_opts, times)

fig.savefig(root_path + "/plots/forest/pi_time.png")


print("----------------------------------------------")
print("Q Learning")
print("----------------------------------------------")

runs = []
state_opts = [200, 400, 800, 1600]
for states in state_opts:
    t = time()
    P, R = mdptoolbox.example.forest(S=states, r1=states/10, r2=states/20, p=0.05)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.99999,
                                  n_iter=10000000,
                                  alpha_decay=1.0,
                                  alpha=0.1,
                                  epsilon_decay=0.99999)
    run_stats = ql.run(init_state=0, episode_len=1000)
    policy = ql.policy


    
    state_count = np.zeros(states)
    for run in run_stats:
        state_count[run["State"]] += 1

    iterations = run_stats[len(run_stats) - 1]["Iteration"]

    with open(root_path + "/plots/forest/" + file_name, "a") as text_file:
        text_file.write("\n-----------------------------------------------------------------------\n")
        text_file.write("QLearning time for forest with {} states: {}\n".format(states, (time() - t)))
        text_file.write(str(policy))
        text_file.write("\nQLearning - {} state counter after {} iterations\n".format(states, iterations))
        text_file.write(str(state_count))

    # if states == 400:
    #     print(policy)
    # print("Total Iterations: {}".format(run_stats[len(run_stats) - 1]["Iteration"]))
    # print("Value Iteration time: {}".format(time() - t))
    # print()
    runs.append(run_stats)

# print(policy)
print("Total Iterations: {}".format(run_stats[len(run_stats) - 1]["Iteration"]))
print("Q Learning time: {}".format(time() - t))
print()

fig, ax = plt.subplots()

ax.set_title("QLearning - Mean Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("mean value", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    mean_val = [i["Mean V"] for i in runs[i]]

    ax.plot(iterations, mean_val, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/ql_mean_val.png")

fig, ax = plt.subplots()

ax.set_title("QLearning - Max Value vs Iterations")
ax.set_xlabel("iteration", fontweight='bold')
ax.set_ylabel("max value", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    max_val = [i["Max V"] for i in runs[i]]

    ax.plot(iterations, max_val, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/ql_max_val.png")

fig, ax = plt.subplots()

ax.set_title("QLearning - Error vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

for i in range(len(runs)):
    iterations = [i["Iteration"] for i in runs[i]]
    errors = [i["Error"] for i in runs[i]]

    ax.plot(iterations, errors, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/ql_error.png")

fig, ax = plt.subplots()

ax.set_title("QLearning - Error vs Time")
ax.set_xlabel("time", fontweight='bold')
ax.set_ylabel("error", fontweight='bold')

for i in range(len(runs)):
    times = [i["Time"] for i in runs[i]]
    errors = [i["Error"] for i in runs[i]]

    ax.plot(times, errors, label=state_opts[i])

ax.legend()

fig.savefig(root_path + "/plots/forest/ql_error_time.png")

fig, ax = plt.subplots()

ax.set_title("QLearning - Time vs States")
ax.set_xlabel("number of states", fontweight='bold')
ax.set_ylabel("convergence time", fontweight='bold')

times = []
for r in runs:
    times.append(r[len(r) - 1]["Time"])

ax.plot(state_opts, times)

fig.savefig(root_path + "/plots/forest/ql_time.png")
