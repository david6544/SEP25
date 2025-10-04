import numpy as np

def load_data(file_path, d):

    data = np.loadtxt(file_path)  

    # make it 2D (1 row, n columns)
    if data.ndim == 1:
        data = data.reshape(-1, d + 1)
     
    x_inputs = data[:, :-1]
    y_output = data[:, -1]
    return x_inputs, y_output
