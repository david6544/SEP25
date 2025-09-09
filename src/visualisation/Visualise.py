import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# create a random dataset
n = 500   
d = 1
data = np.random.rand(n, d + 1)

x_inputs = data[:, :d]   
y_output = data[:, d]    

# choose a colour map for showing the output
colour_map = plt.cm.Greys

# slider width
slice_tolerance = 0.1


# calc mask
def makeMask(slider_values):
    mask = np.ones(len(x_inputs), dtype=bool)
    for i, val in enumerate(slider_values, start=3):  
        mask &= np.abs(x_inputs[:, i] - val) < slice_tolerance
    return mask


# init slice
init_vals = [0.5] * max(0, d - 3)
mask = makeMask(init_vals)


# make the figure and choose how to plot depending on dimensions
fig = plt.figure(figsize=(7, 7))

if d == 1:
    ax = fig.add_subplot()
    sc = ax.scatter(x_inputs[mask, 0], y_output[mask])
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title("1D n Elements")

elif d == 2:
    ax = fig.add_subplot(projection = "3d")
    sc = ax.scatter(x_inputs[mask, 0], x_inputs[mask, 1], 
                    y_output[mask])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D n Elements")

#else:  # three or more input dimensions
#    ax = fig.add_subplot(111, projection= "3d" )
#    sc = ax.scatter(x_inputs[mask, 0], x_inputs[mask, 1], x_inputs[mask, 2], 
#                    c=y_output[mask], cmap=colour_map)
#    ax.set_xlabel("X1")
#    ax.set_ylabel("X2")
#    ax.set_zlabel("X3")
#    ax.set_title(f"{d} dimensional state space (output is colour)")
#    plt.colorbar(sc, ax=ax, label="Output")


# make sliders for every input dimension beyond the first three
#sliders = []
#if d > 3:
#    plt.subplots_adjust(bottom=0.05 * (d - 3) + 0.15)
#    for i in range(3, d):
#        ax_slider = plt.axes([0.2, 0.05 * (i - 3) + 0.05, 0.6, 0.03])
#        slider = Slider(ax_slider, f"X{i+1}", 0.0, 1.0, 
#                        valinit=0.5, valstep=0.01)
#        sliders.append(slider)


# update the picture whenever a slider is moved
#def updatePlot(val):
#    slider_vals = [s.val for s in sliders] if sliders else []
#    mask = makeMask(slider_vals)

#    if d == 1:
#        sc.set_offsets(np.c_[range(np.sum(mask)), x_inputs[mask, 0]])
#        sc.set_array(y_output[mask])
#    elif d == 2:
#        sc.set_offsets(x_inputs[mask, :2])
#        sc.set_array(y_output[mask])
#    else:
#        sc._offsets3d = (x_inputs[mask, 0], x_inputs[mask, 1], x_inputs[mask, 2])
#        sc.set_array(y_output[mask])

#    fig.canvas.draw_idle()


# connect each slider to the update function
#for s in sliders:
#    s.on_changed(updatePlot)

# show the finished plot
plt.show()
