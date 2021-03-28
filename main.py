import pandas as pd
from sys import argv
from loader import iterate_instances
from time import time
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

from optimizers import Route, RandomOptimizer, LocalInnerEdgeOptimizer, GlobalInnerEdgeOptimizer, \
    LocalInnerVertexOptimizer, GlobalInnerVertexOptimizer
from optimizers.base import Solution

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def visualize_route(route: Route, points, path: str, filename: str):
    fig, ax = plt.subplots()

    x, y = [points[v_id][1] for v_id in route], [points[v_id][2] for v_id in route]

    x.append(x[0])
    y.append(y[0])

    ax.scatter(x, y)
    ax.plot(x, y)

    x, y = [v_id[1] for v_id in points if v_id not in route], [v_id[2] for v_id in points if v_id not in route]
    ax.scatter(x, y)
    fig.savefig(path + '/' + filename)


def main(instances_path: str, repeat: int, output_path: str):
    print('[STARTED]')

    optimizers = [
        GlobalInnerEdgeOptimizer, GlobalInnerVertexOptimizer,
        LocalInnerEdgeOptimizer, LocalInnerVertexOptimizer
    ]

    for x in iterate_instances(instances_path):
        df_time = pd.DataFrame(columns=[o.__name__ for o in optimizers])
        df_cost = pd.DataFrame(columns=[o.__name__ for o in optimizers])
        fname, distance_matrix, points = x.name, x.distance_matrix, x.points

        routes = []
        print('[INSTANCE]', fname)
        inform.write(f'[INSTANCE] {fname}\n')

        best_solutions_routes = [Solution(np.inf, Route([])) for _ in optimizers]
        for i in range(repeat):
            inform.write(f'[PROGRESS {i+1}/{repeat}]\n')
            vertices = distance_matrix.shape[0]
            route = Route([*range(vertices//2)])
            random.shuffle(route)
            routes.append(route)

            t, c = {}, {}
            for oid, Optimizer in enumerate(optimizers):
                begin = time()

                opt = Optimizer(distance_matrix, route)
                solution = opt()

                end = time()
                t[Optimizer.__name__] = end - begin
                c[Optimizer.__name__] = solution.cost

                if solution.cost < best_solutions_routes[oid].cost:
                    best_solutions_routes[oid] = solution

            df_time = df_time.append(t, ignore_index=True)
            df_cost = df_cost.append(c, ignore_index=True)

        df_time = df_time.describe().loc[['min', 'mean', 'max']]
        df_cost = df_cost.describe().loc[['min', 'mean', 'max']]

        print('[TIME]')
        print(df_time)

        print('[COST]')
        print(df_cost)

        longest_mean_time = df_time.loc['mean'].max()
        df_random_opt = pd.DataFrame(columns=('time', 'cost'))

        random_solution = Solution(np.inf, Route([]))
        for r in routes:
            begin = time()
            solution = RandomOptimizer(distance_matrix, r, longest_mean_time)()
            end = time()

            df_random_opt = df_random_opt.append({'time': end - begin, 'cost': solution.cost}, ignore_index=True)
            if solution.cost < random_solution.cost:
                random_solution = solution

        df_random_opt = df_random_opt.describe().loc[['min', 'mean', 'max']]

        print('[COST_RANDOM]')
        print(df_random_opt)

        inform.write('[VISUALISING]\n')
        for method, bs in zip([o.__name__ for o in optimizers], best_solutions_routes):
            visualize_route(bs.route, points, output_path, f'{fname}_{method}')

        visualize_route(random_solution.route, points, output_path, f'{fname}_{RandomOptimizer.__name__}')

    inform.write('[FINISHED]\n')


if __name__ == '__main__':
    stdout = argv[4]
    inform = sys.stdout
    with open(stdout, 'w') as stdout:
        sys.stdout = stdout
        main(argv[1], int(argv[2]), argv[3])
