from src.algorithm import algorithm
from src.geometry import Environment


def main():
    # TODO: get environment and args from somewhere
    # raise NotImplementedError
    algorithm(Environment(0, 1, [], set()), 2.0, 2.0)


if __name__ == '__main__':
    main()
