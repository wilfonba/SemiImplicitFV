#!/usr/bin/env python3
"""Visualize the 1D subsonic advection (entropy wave) results."""

import numpy as np
import matplotlib.pyplot as plt

# Load data
t0    = np.loadtxt("advection_t0.dat")      # x  rho  u  p
tf    = np.loadtxt("advection_final.dat")    # x  rho  u  p
exact = np.loadtxt("advection_exact.dat")    # x  rho_exact

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("1D Subsonic Advection (Entropy Wave)", fontsize=14)

# --- Density ---
ax = axes[0, 0]
ax.plot(t0[:, 0], t0[:, 1], "k--", label="Initial")
ax.plot(exact[:, 0], exact[:, 1], "g-", linewidth=1.5, label="Exact")
ax.plot(tf[:, 0], tf[:, 1], "b.", markersize=2, label="Computed")
ax.set_xlabel("x")
ax.set_ylabel("Density [kg/m³]")
ax.set_title("Density")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Density error ---
ax = axes[0, 1]
error = tf[:, 1] - exact[:, 1]
ax.plot(tf[:, 0], error, "r-", linewidth=0.8)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("rho_computed - rho_exact")
ax.set_title(f"Density Error  (Linf = {np.max(np.abs(error)):.2e})")
ax.grid(True, alpha=0.3)

# --- Velocity ---
ax = axes[1, 0]
ax.plot(t0[:, 0], t0[:, 2], "k--", label="Initial")
ax.plot(tf[:, 0], tf[:, 2], "b-", linewidth=1.2, label="Final")
ax.set_xlabel("x")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Velocity")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Pressure ---
ax = axes[1, 1]
ax.plot(t0[:, 0], t0[:, 3], "k--", label="Initial")
ax.plot(tf[:, 0], tf[:, 3], "b-", linewidth=1.2, label="Final")
ax.set_xlabel("x")
ax.set_ylabel("Pressure [Pa]")
ax.set_title("Pressure")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
# plt.savefig("advection_results.png", dpi=150)
plt.show()
