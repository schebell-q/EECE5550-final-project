import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.algorithm import algorithm
from src.geometry import Environment
from src.plotting import plot_environment, plot_graph, save_fig
from tests.test_environment import make_long_toy_env


OUTPUT_DIR = Path("main_plots")


def sweep_ks(name: str, env_maker: Callable[[], Environment]):
    ks = [2, 3, 4, 5, 6, np.inf]
    alpha = 1.5

    for k in ks:
        env = env_maker()

        gen_start_time = time.process_time()
        g_final = algorithm(env, k, alpha)
        gen_end_time = time.process_time()

        plan_start_time = time.process_time()
        path_length = g_final.path_distance(env.start, env.goal)
        plan_end_time = time.process_time()

        n_vertices = len(g_final.vertices)
        n_edges = len(g_final.edges)
        gen_time = gen_end_time - gen_start_time
        plan_time = plan_end_time - plan_start_time

        print(name, "k =", k)
        print("\tTgen", gen_time)
        print("\t|V|", n_vertices)
        print("\t|E|", n_edges)
        print("\tPath", path_length)
        print("\tTplan", plan_time)
        print()

        fig = plot_environment(env)
        fig = plot_graph(env, g_final, fig=fig)
        file = OUTPUT_DIR / f"{name}_k_{k}.html"
        save_fig(fig, file)


def main():
    sweep_ks("long_toy", make_long_toy_env)


if __name__ == '__main__':
    main()
