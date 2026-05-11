# SimPEG functionality
import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps
from simpeg.utils import plot_1d_layer_model

# Common Python functionality
import os
import numpy as np
from scipy.constants import mu_0
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

mpl.rcParams.update({"font.size": 14})

def forwardModel(height, res_max, chi=False, ordered=False):
    # Source properties
    s = 4.4
    a = s / 2                      # 2.2
    b = (np.sqrt(3) / 2) * s       # ~3.809

    source_vertices = np.array([
        [ s,  0,   height],
        [ a,  b,   height],
        [-a,  b,   height],
        [-s,  0,   height],
        [-a, -b,   height],
        [ a, -b,   height],
        [ s,  0,   height],
    ])

    # Receiver properties
    receiver_locations = np.array([0.0, 0.0, height])  # or (N, 3) numpy.ndarray
    receiver_orientation = "z"  # "x", "y" or "z"
    times = np.logspace(-5, -2, 31)  # time channels (s)

    stepoff_waveform = tdem.sources.StepOffWaveform()

    # Define receiver list. In our case, we have only a single receiver for each source.
    # When simulating the response for multiple data types for the same source,
    # the list consists of multiple receiver objects.
    receiver_list = []
    receiver_list.append(
        tdem.receivers.PointMagneticFluxDensity(
            receiver_locations, times, orientation=receiver_orientation
        )
    )

    # Define source list. In our case, we have only a single source.
    source_list = [
        tdem.sources.LineCurrent(
            receiver_list=receiver_list,
            location=source_vertices,
            waveform=stepoff_waveform,
        )
    ]

    # Define the survey
    survey = tdem.Survey(source_list)

    if chi == False:
        min_res = 1
        max_res = int(res_max)

        res1 = random.randint(min_res,max_res)
        res2 = random.randint(min_res,max_res)
        res3 = random.randint(min_res,max_res)
        res4 = random.randint(min_res,max_res)
        res5 = random.randint(min_res,max_res)
    else:
        res_list = chiSquareRand(res_max)

        res1 = res_list[0]
        res2 = res_list[1]
        res3 = res_list[2]
        res4 = res_list[3]
        res5 = res_list[4]

    if (ordered):
        res_list = [res1, res2, res3, res4, res5]
        res_list.sort()
        res_list.reverse()

        res1 = res_list[0]
        res2 = res_list[1]
        res3 = res_list[2]
        res4 = res_list[3]
        res5 = res_list[4]

    # Layer conductivities
    layer_conductivities = np.r_[1/res1, 1/res2, 1/res3, 1/res4, 1/res5]

    # Layer thicknesses
    layer_thicknesses = np.r_[20.0,20.0,20.0,20.0]

    # Number of layers
    n_layers = len(layer_conductivities)

    """
    fig = plt.figure(figsize=(4, 5))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1 = plot_1d_layer_model(layer_thicknesses, layer_conductivities, scale="log", ax=ax1)
    ax1.grid(which="both")
    ax1.set_xlabel(r"Conductivity ($S/m$)")
    plt.show()
    """

    # Define model and mapping for a conductivity model.
    conductivity_model = layer_conductivities.copy()
    conductivity_map = maps.IdentityMap(nP=n_layers)

    # Define model and mappings for the parametric model.
    # Note the ordering in which you defined the model parameters and the
    # order in which you defined the wire mappings matters!!!
    parametric_model = np.r_[layer_thicknesses, np.log(1 / layer_conductivities)]
    wire_map = maps.Wires(("thicknesses", n_layers - 1), ("log_resistivity", n_layers))
    thicknesses_map = wire_map.thicknesses
    log_resistivity_map = maps.ExpMap() * wire_map.log_resistivity

    simulation_conductivity = tdem.simulation_1d.Simulation1DLayered(
        survey=survey,
        sigmaMap=conductivity_map,
        thicknesses=layer_thicknesses,
    )

    simulation_parametric = tdem.simulation_1d.Simulation1DLayered(
        survey=survey,
        rhoMap=log_resistivity_map,
        thicknessesMap=thicknesses_map,
    )

    dpred_conductivity = simulation_conductivity.dpred(conductivity_model)

    """
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.78])
    ax.loglog(times, dpred_conductivity, "b-", lw=3)
    ax.loglog(times, dpred_parametric, "r--", lw=3)
    ax.set_xlim([times.min(), times.max()])
    ax.grid()
    ax.set_xlabel("Times (s)")
    ax.set_ylabel("B (T)")
    ax.set_title("Magnetic Flux Density")
    ax.legend(["Conductivity model", "Parametric model"])
    plt.show()
    """

    return [times,dpred_conductivity]

def chiSquareRand(res_max):
    df = 2.0  # degrees of freedom
    res_min = 1.0
    n_layers = 5

    # sample chi-square
    chi_samples = np.random.chisquare(df=df, size=n_layers)

    # normalize to [0, 1]
    chi_norm = chi_samples / np.percentile(chi_samples, 99)

    # map to resistivity range
    resistivities = res_min + chi_norm * (res_max - res_min)

    # clip extreme values
    resistivities = np.clip(resistivities, res_min, res_max)

    return resistivities

simulationArr = []
for i in range(1000):
    simulationArr.append(forwardModel(10.0,500.0,chi=True,ordered=True))

data_sets = simulationArr

"""
plt.figure()

for i, (x, y) in enumerate(data_sets):
    plt.plot(x, y, label=f"Dataset {i+1}")

plt.xscale("log") 
plt.yscale("log")  
plt.xlabel("Time")   
plt.ylabel("Magnetic Flux")
plt.title("Multiple Datasets")
#plt.legend()
plt.grid(True)

plt.show()
"""

def plot_decay_density(
    data_sets,
    n_bins=80,
    cmap="inferno",
    show_mean=True
):
    times = data_sets[0][0]
    responses = np.array([y for _, y in data_sets])

    # Log-space
    log_t = np.log10(times)
    log_r = np.log10(responses)

    # Bin edges for responses
    r_min, r_max = log_r.min(), log_r.max()
    r_bins = np.linspace(r_min, r_max, n_bins + 1)

    # Build density map
    density = np.zeros((n_bins, log_t.size))

    for i in range(log_t.size):
        hist, _ = np.histogram(log_r[:, i], bins=r_bins)
        density[:, i] = hist

    # Plot
    plt.figure(figsize=(8, 6))

    plt.imshow(
        density,
        aspect="auto",
        origin="lower",
        extent=[log_t.min(), log_t.max(), r_min, r_max],
        cmap=cmap
    )

    if show_mean:
        mean_curve = log_r.mean(axis=0)
        plt.plot(log_t, mean_curve, color="cyan", lw=2)

    plt.xlabel("log(Time (s))")
    plt.ylabel("log10(Response)")
    plt.title("Decay Curve Density")
    plt.colorbar(label="Count")

    plt.tight_layout()
    plt.show()

plot_decay_density(data_sets, n_bins=30)