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
    
    report_file = open(OUTPUT_DIR / "report.txt", "w")
    data_file = open(OUTPUT_DIR / "data.csv", "w")
    data_file.write("k,Tgen,|V|,|E|,Path cost,Tplan\n")

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

        report_file.write(f"""{name} k = {k}
\tTgen {gen_time}
\t|V| {n_vertices}
\t|E| {n_edges}
\tPath {path_length}
\tTplan {plan_time}

""")
        data_file.write(f"{k},{gen_time},{n_vertices},{n_edges},{path_length},{plan_time}\n")

        fig = plot_environment(env)
        fig = plot_graph(env, g_final, fig=fig)
        file = OUTPUT_DIR / f"{name}_k_{k}.html"
        save_fig(fig, file)


def main():
    sweep_ks("long_toy", make_long_toy_env)


if __name__ == '__main__':
    main()
