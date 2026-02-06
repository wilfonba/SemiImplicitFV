#!/usr/bin/env python3
"""Visualize initial and final conditions for the Sod shock tube problem."""

import numpy as np
import matplotlib.pyplot as plt

# Load data
t0 = np.loadtxt("sod_t0.dat")
tf = np.loadtxt("sod_final.dat")

fields = ["Density", "Velocity", "Pressure", "Sigma"]
cols = [1, 2, 3, 4]  # column indices in data

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("Sod Shock Tube — Semi-Implicit FV with IGR", fontsize=14)

for ax, name, col in zip(axes.flat, fields, cols):
    ax.plot(t0[:, 0], t0[:, col], "k--", label="t = 0")
    ax.plot(tf[:, 0], tf[:, col], "b-", linewidth=1.2, label="t = 0.2")
    ax.set_xlabel("x")
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
# plt.savefig("sod_shock.png", dpi=150)
plt.show()
