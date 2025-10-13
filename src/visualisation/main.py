import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import dataInput as i


def loadAndParse(file_path, dimensions):
    data = np.loadtxt(file_path)

    # make it 2D (1 row, n columns)
    if data.ndim == 1:
        data = data.reshape(-1, dimensions + 1)

    x_inputs = data[:, :-1]
    y_output = data[:, -1]
    return x_inputs, y_output


def run(filepath: str):
    pass


if __name__ == "__main__":
    run("../../Ackley-1000.txt")
