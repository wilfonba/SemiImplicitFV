#!/usr/bin/env python3
"""
Compute and plot volume-averaged kinetic energy, enstrophy, and KE dissipation
rate over time from VTK output of the 3D Taylor-Green vortex.

Usage:
    python3 plot_ke_enstrophy.py VTK_DIR1 [VTK_DIR2] [-l LABEL1 LABEL2] [-o OUTPUT]

Each VTK_DIR should contain a .pvd file and snapshot_*/ subdirectories.
When two directories are given, both datasets are plotted on the same axes
for comparison.

Options:
    -l LABEL1 [LABEL2]   Legend labels (defaults to directory names)
    -o OUTPUT             Output filename prefix (default: ke_enstrophy)
"""

import sys
import os
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_pvd(pvd_path):
    """Parse a .pvd file and return sorted list of (time, pvtr_path) tuples."""
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    base_dir = os.path.dirname(pvd_path)
    entries = []
    for ds in root.iter("DataSet"):
        t = float(ds.get("timestep"))
        fpath = os.path.join(base_dir, ds.get("file"))
        entries.append((t, fpath))
    entries.sort(key=lambda x: x[0])
    return entries


def load_snapshot(pvtr_path):
    """Load a .pvtr file and return the vtkRectilinearGrid."""
    reader = vtk.vtkXMLPRectilinearGridReader()
    reader.SetFileName(pvtr_path)
    reader.Update()
    return reader.GetOutput()


def compute_ke_enstrophy(grid):
    """
    Compute volume-averaged kinetic energy and enstrophy from a snapshot.

    Kinetic energy:  KE = (1/V) * sum_cells 0.5 * rho * |u|^2 * dV
    Enstrophy:       Omega = (1/V) * sum_cells 0.5 * rho * |omega|^2 * dV

    where omega = curl(u) is computed with 2nd-order central differences
    (periodic boundary handling).
    """
    cd = grid.GetCellData()

    # Extract fields
    vel_vtk = cd.GetArray("Velocity")
    rho_vtk = cd.GetArray("Density")
    vel = vtk_to_numpy(vel_vtk)  # (ncells, 3)
    rho = vtk_to_numpy(rho_vtk)  # (ncells,)

    # Grid dimensions (node counts) -> cell counts
    dims = grid.GetDimensions()
    nx, ny, nz = dims[0] - 1, dims[1] - 1, dims[2] - 1

    # Cell spacings
    xc = vtk_to_numpy(grid.GetXCoordinates())
    yc = vtk_to_numpy(grid.GetYCoordinates())
    zc = vtk_to_numpy(grid.GetZCoordinates())
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]
    dz = zc[1] - zc[0]
    dV = dx * dy * dz

    # Reshape to 3D (VTK uses Fortran ordering: x fastest)
    u = vel[:, 0].reshape((nz, ny, nx))
    v = vel[:, 1].reshape((nz, ny, nx))
    w = vel[:, 2].reshape((nz, ny, nx))
    rho3d = rho.reshape((nz, ny, nx))

    # Vorticity via central differences (periodic BCs via np.roll)
    # omega_x = dw/dy - dv/dz
    # omega_y = du/dz - dw/dx
    # omega_z = dv/dx - du/dy
    dwdy = (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2.0 * dy)
    dvdz = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dz)
    dudz = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dz)
    dwdx = (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2.0 * dx)
    dvdx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2.0 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)

    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy

    omega_sq = omega_x**2 + omega_y**2 + omega_z**2
    vel_sq = u**2 + v**2 + w**2

    # Volume-averaged quantities
    total_volume = nx * ny * nz * dV
    ke = np.sum(0.5 * rho3d * vel_sq * dV) / total_volume
    enstrophy = np.sum(0.5 * rho3d * omega_sq * dV) / total_volume

    return ke, enstrophy


def process_vtk_dir(vtk_dir):
    """Process a VTK directory and return (times, KE, enstrophy) arrays."""
    vtk_dir = os.path.abspath(vtk_dir)

    pvd_files = glob.glob(os.path.join(vtk_dir, "*.pvd"))
    if not pvd_files:
        print(f"Error: no .pvd file found in {vtk_dir}")
        sys.exit(1)
    pvd_path = pvd_files[0]
    print(f"Using PVD file: {pvd_path}")

    entries = parse_pvd(pvd_path)
    print(f"Found {len(entries)} snapshots")

    times = []
    kinetic_energies = []
    enstrophies = []

    for i, (t, pvtr_path) in enumerate(entries):
        print(f"  [{i+1}/{len(entries)}] t = {t:.4f} ... ", end="", flush=True)
        grid = load_snapshot(pvtr_path)
        ke, enst = compute_ke_enstrophy(grid)
        times.append(t)
        kinetic_energies.append(ke)
        enstrophies.append(enst)
        print(f"KE = {ke:.6e}, Enstrophy = {enst:.6e}")

    return np.array(times), np.array(kinetic_energies), np.array(enstrophies)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot KE and enstrophy from VTK snapshots.")
    parser.add_argument("vtk_dirs", nargs="+",
                        help="VTK directories containing .pvd files")
    parser.add_argument("-l", "--labels", nargs="+", default=None,
                        help="Legend labels (defaults to directory names)")
    parser.add_argument("-o", "--output", default="ke_enstrophy",
                        help="Output filename prefix (default: ke_enstrophy)")
    args = parser.parse_args()

    vtk_dirs = args.vtk_dirs
    labels = args.labels or [os.path.basename(os.path.abspath(d)) for d in vtk_dirs]
    if len(labels) < len(vtk_dirs):
        labels.extend([os.path.basename(os.path.abspath(d))
                       for d in vtk_dirs[len(labels):]])

    # Process each directory
    results = []
    for vtk_dir, label in zip(vtk_dirs, labels):
        print(f"\n=== Processing: {label} ({vtk_dir}) ===")
        times, ke, enst = process_vtk_dir(vtk_dir)

        # KE dissipation rate: -dKE/dt (central differences, one-sided at endpoints)
        dke_dt = np.gradient(ke, times)
        dissipation = -dke_dt
        results.append((label, times, ke, enst, dissipation))

        # Save per-directory CSV
        csv_path = args.output + f"_{label}.csv"
        np.savetxt(csv_path,
                   np.column_stack([times, ke, enst, dissipation]),
                   header="time, kinetic_energy, enstrophy, ke_dissipation",
                   delimiter=",", fmt="%.10e")
        print(f"Data saved to {csv_path}")

    # Plot
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    styles = ["-o", "--s", "-.^", ":D"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for i, (label, times, ke, enst, diss) in enumerate(results):
        c = colors[i % len(colors)]
        s = styles[i % len(styles)]
        ax1.plot(times, ke, s, color=c, markersize=3, label=label)
        ax2.plot(times, diss, s, color=c, markersize=3, label=label)
        ax3.plot(times, enst, s, color=c, markersize=3, label=label)

    ax1.set_ylabel("Kinetic Energy")
    ax1.set_title("3D Taylor-Green Vortex")
    ax1.grid(True)
    ax1.legend()

    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"KE Dissipation Rate ($-dKE/dt$)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, 0.016)
    ax2.set_xlim(0,20)

    ax3.set_ylabel("Enstrophy")
    ax3.grid(True)
    ax3.legend()
    ax3.set_ylim(0, 13)
    ax3.set_xlim(0,20)

    plt.tight_layout()
    plot_path = args.output + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
