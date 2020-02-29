import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import mlrose_hiive as mlrose
from mlrose_hiive.runners import GARunner, SARunner, RHCRunner, MIMICRunner
from mlrose_hiive import ArithDecay


root_path = os.getcwd()

global eval_count
eval_count = 0

rhc_repeats = 1
ga_repeats = 200
sa_repeats = 1
mimic_repeats = 1


def fitness_counter(state):
    global eval_count
    fitness = mlrose.FourPeaks(t_pct=0.25)
    eval_count += 1
    return fitness.evaluate(state)


fitness = mlrose.CustomFitness(fitness_counter)
problem = mlrose.DiscreteOpt(length=40,
                             fitness_fn=fitness,
                             maximize=True,
                             max_val=2)

ga_evals_list = []
df_ga_stats_list = []
df_ga_curves_list = []
for r in range(ga_repeats):
    ga = GARunner(problem=problem,
                  experiment_name="ga_test",
                  output_directory="./results/",
                  seed=r,
                  iteration_list=2 ** np.arange(18),
                  max_attempts=50,
                  population_sizes=[300],
                  mutation_rates=[0.2])

    df_ga_stats, df_ga_curves = ga.run()
    df_ga_stats_list.append(df_ga_stats)
    df_ga_curves_list.append(df_ga_curves)

    print("fitness evaluations: {}".format(eval_count))
    ga_evals_list.append(eval_count)
    eval_count = 0


sa_evals_list = []
df_sa_stats_list = []
df_sa_curves_list = []
for r in range(sa_repeats):
    sa = SARunner(problem=problem,
                  experiment_name="sa_test",
                  output_directory="./results/",
                  seed=r,
                  iteration_list=2 ** np.arange(18),
                  max_attempts=50,
                  temperature_list=[1],
                  decay_list=[ArithDecay])

    df_sa_stats, df_sa_curves = sa.run()
    df_sa_stats_list.append(df_sa_stats)
    df_sa_curves_list.append(df_sa_curves)

    print("fitness evaluations: {}".format(eval_count))
    sa_evals_list.append(eval_count)
    eval_count = 0


rhc_evals_list = []
df_rhc_stats_list = []
df_rhc_curves_list = []
for r in range(rhc_repeats):
    rhc = RHCRunner(problem=problem,
                    experiment_name="rhc_test",
                    output_directory="./results/",
                    seed=r,
                    iteration_list=2 ** np.arange(18),
                    max_attempts=1000,
                    restart_list=[0])

    df_rhc_stats, df_rhc_curves = rhc.run()
    df_rhc_stats_list.append(df_rhc_stats)
    df_rhc_curves_list.append(df_rhc_curves)

    print("fitness evaluations: {}".format(eval_count))
    rhc_evals_list.append(eval_count)
    eval_count = 0


mimic_evals_list = []
df_mimic_stats_list = []
df_mimic_curves_list = []
for r in range(mimic_repeats):
    problem.set_mimic_fast_mode(True)
    mimic = MIMICRunner(problem=problem,
                        experiment_name="mimic_test",
                        output_directory="./results/",
                        seed=r,
                        iteration_list=2 ** np.arange(18),
                        max_attempts=200,
                        use_fast_mimic=True,
                        keep_percent_list=[0.05],
                        population_sizes=[200])

    df_mimic_stats, df_mimic_curves = mimic.run()
    df_mimic_stats_list.append(df_mimic_stats)
    df_mimic_curves_list.append(df_mimic_curves)

    print("fitness evaluations: {}".format(eval_count))
    mimic_evals_list.append(eval_count)
    eval_count = 0

#
# fitness vs iterations
#
fig, ax = plt.subplots()

ax.set_title("Four Peaks - Fitness vs Iterations")
ax.set_xlabel("iterations", fontweight='bold')
ax.set_ylabel("fitness", fontweight='bold')

for r in range(ga_repeats):
    ga_iterations_df = df_ga_curves_list[r]['Iteration']
    ga_fitness_df = df_ga_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(ga_iterations_df, ga_fitness_df, color='r', label='GA')

for r in range(sa_repeats):
    sa_iterations_df = df_sa_curves_list[r]['Iteration']
    sa_fitness_df = df_sa_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(sa_iterations_df, sa_fitness_df, color='b', label='SA')

for r in range(rhc_repeats):
    rhc_iterations_df = df_rhc_curves_list[r]['Iteration']
    rhc_fitness_df = df_rhc_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(rhc_iterations_df, rhc_fitness_df, color='g', label='RHC')

for r in range(mimic_repeats):
    mimic_iterations_df = df_mimic_curves_list[r]['Iteration']
    mimic_fitness_df = df_mimic_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(mimic_iterations_df, mimic_fitness_df, color='k', label='MIMIC')

custom_lines = [Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='k', lw=2)]

ax.legend(custom_lines, ['GA', 'SA', 'RHC', 'MIMIC'], loc='best')

fig.savefig(root_path + "/plots/peaks/iterations.png")

#
# fitness vs evaluations
#
fig, ax = plt.subplots()

ax.set_title("Four Peaks - Max Fitness vs Fitness Evaluations")
ax.set_xlabel("fitness evaluations", fontweight='bold')
ax.set_ylabel("max fitness", fontweight='bold')

for r in range(ga_repeats):
    ga_max_fitness = (df_ga_curves_list[r]['Fitness']).max()
    # Make the plot
    ax.scatter(ga_evals_list[r], ga_max_fitness, color='r', s=100, label='GA')

for r in range(sa_repeats):
    sa_max_fitness = (df_sa_curves_list[r]['Fitness']).max()
    # Make the plot
    ax.scatter(sa_evals_list[r], sa_max_fitness, color='b', s=100, label='SA')

for r in range(rhc_repeats):
    rhc_max_fitness = (df_rhc_curves_list[r]['Fitness']).max()
    # Make the plot
    ax.scatter(rhc_evals_list[r], rhc_max_fitness, color='g', s=100, label='RHC')

for r in range(mimic_repeats):
    mimic_max_fitness = (df_mimic_curves_list[r]['Fitness']).max()
    # Make the plot
    ax.scatter(mimic_evals_list[r], mimic_max_fitness, color='k', s=100, label='MIMIC')

custom_lines = [Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='k', lw=2)]

ax.legend(custom_lines, ['GA', 'SA', 'RHC', 'MIMIC'], loc='best')

fig.savefig(root_path + "/plots/peaks/evaluations.png")

#
# fitness vs time
#
fig, ax = plt.subplots()

ax.set_title("Four Peaks - Fitness vs Time")
ax.set_xlabel("time", fontweight='bold')
ax.set_ylabel("fitness", fontweight='bold')

for r in range(ga_repeats):
    ga_time_df = df_ga_curves_list[r]['Time']
    ga_fitness_df = df_ga_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(ga_time_df, ga_fitness_df, color='r', label='GA')

for r in range(sa_repeats):
    sa_time_df = df_sa_curves_list[r]['Time']
    sa_fitness_df = df_sa_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(sa_time_df, sa_fitness_df, color='b', label='SA')

for r in range(rhc_repeats):
    rhc_time_df = df_rhc_curves_list[r]['Time']
    rhc_fitness_df = df_rhc_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(rhc_time_df, rhc_fitness_df, color='g', label='RHC')

for r in range(mimic_repeats):
    mimic_time_df = df_mimic_curves_list[r]['Time']
    mimic_fitness_df = df_mimic_curves_list[r]['Fitness']
    # Make the plot
    ax.plot(mimic_time_df, mimic_fitness_df, color='k', label='MIMIC')

custom_lines = [Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='k', lw=2)]

ax.legend(custom_lines, ['GA', 'SA', 'RHC', 'MIMIC'], loc='best')

fig.savefig(root_path + "/plots/peaks/time.png")

print("ga evaluations: {}".format(ga_evals_list))
print("sa evaluations: {}".format(sa_evals_list))
print("rhc evaluations: {}".format(rhc_evals_list))
print("mimic evaluations: {}".format(mimic_evals_list))
