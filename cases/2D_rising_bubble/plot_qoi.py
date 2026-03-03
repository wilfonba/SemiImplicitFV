#!/usr/bin/env python3
"""
Plot the four Hysing benchmark quantities of interest for the 2D rising bubble.

Reads VTK output from a .pvd file and computes:
  1. Bubble area (mass conservation check)
  2. Circularity = 2*sqrt(pi*A) / perimeter
  3. Center of mass (y-coordinate)
  4. Rise velocity (y-component, alpha-weighted)

Optionally overlays reference data from Hysing et al. (2009) text files.
Reference data format: 5 columns (time, mass, circularity, center_of_mass, rise_velocity).

Usage:
  python3 plot_qoi.py                           # defaults: VTK/ directory
  python3 plot_qoi.py --vtk-dir path/to/VTK
  python3 plot_qoi.py --ref c2g1l8.txt --ref-label "TP2D L8"
  python3 plot_qoi.py --ref c2g1l8.txt c2g2l3.txt --ref-label "TP2D L8" "FreeLIFE L3"
"""

import argparse
import glob
import multiprocessing
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def parse_pvd(pvd_path):
    """Parse a .pvd file and return sorted (time, pvtr_path) pairs."""
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    base_dir = os.path.dirname(pvd_path)
    entries = []
    for ds in root.iter("DataSet"):
        t = float(ds.attrib["timestep"])
        fpath = os.path.join(base_dir, ds.attrib["file"])
        entries.append((t, fpath))
    entries.sort(key=lambda x: x[0])
    return entries


def compute_qoi(mesh, alpha_field="Alpha_1"):
    """
    Compute the four quantities of interest from a single pyvista mesh.

    Returns (area, circularity, center_of_mass_y, rise_velocity_y).
    """
    alpha = mesh.cell_data[alpha_field]
    velocity = mesh.cell_data["Velocity"]

    # Cell volumes (areas in 2D — pyvista gives volume, which for 2D
    # extruded grids is area * dz; we normalize by dividing out later)
    sizes = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    vol = sizes.cell_data["Volume"]

    # Normalize volumes: for 2D meshes extruded by dz, divide out the z-extent
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    dz = bounds[5] - bounds[4]
    if dz > 0:
        cell_area = vol / dz
    else:
        cell_area = vol

    # Bubble area: integral of alpha over domain
    bubble_area = np.sum(alpha * cell_area)

    # Center of mass (y): integral(y * alpha * dA) / integral(alpha * dA)
    centers = mesh.cell_centers().points
    y_coords = centers[:, 1]
    alpha_area = alpha * cell_area
    total_alpha_area = np.sum(alpha_area)
    if total_alpha_area > 0:
        com_y = np.sum(y_coords * alpha_area) / total_alpha_area
    else:
        com_y = 0.0

    # Rise velocity (y): integral(v_y * alpha * dA) / integral(alpha * dA)
    vy = velocity[:, 1]
    if total_alpha_area > 0:
        rise_vel = np.sum(vy * alpha_area) / total_alpha_area
    else:
        rise_vel = 0.0

    # Circularity: 2*sqrt(pi*A) / perimeter
    # Slice at z=mid to get a true 2D plane, then contour for line segments
    point_mesh = mesh.cell_data_to_point_data()
    z_mid = 0.5 * (bounds[4] + bounds[5])
    perimeter = 0.0
    try:
        sliced = point_mesh.slice(normal="z", origin=[0, 0, z_mid])
        contour = sliced.contour(isosurfaces=[0.5], scalars=alpha_field)
        if contour.n_cells > 0:
            pts = contour.points
            lines = contour.lines
            i = 0
            while i < len(lines):
                n = lines[i]
                if n == 2:
                    p0 = pts[lines[i + 1]]
                    p1 = pts[lines[i + 2]]
                    perimeter += np.linalg.norm(p1 - p0)
                i += n + 1
    except Exception:
        pass

    if perimeter > 0:
        circularity = 2.0 * np.sqrt(np.pi * bubble_area) / perimeter
    else:
        circularity = 1.0

    # Collect contour points (x, y) for optional bubble shape plotting
    contour_xy = None
    try:
        if contour.n_cells > 0:
            contour_xy = contour.points[:, :2].copy()
    except Exception:
        pass

    return bubble_area, circularity, com_y, rise_vel, contour_xy


def _process_snapshot(args_tuple):
    """Worker function for parallel QoI computation. Reads one snapshot."""
    idx, t, pvtr_path, alpha_field = args_tuple
    mesh = pv.read(pvtr_path)
    a, c, y, v, contour_xy = compute_qoi(mesh, alpha_field=alpha_field)
    return idx, t, a, c, y, v, contour_xy


def load_reference(filepath):
    """
    Load Hysing reference data file.
    Format: 5 columns — time, mass, circularity, center_of_mass, rise_velocity.
    Lines starting with '#' or '%' are skipped.
    """
    data = np.loadtxt(filepath, comments=("#", "%"))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        "time": data[:, 0],
        "mass": data[:, 1],
        "circularity": data[:, 2],
        "center_of_mass": data[:, 3],
        "rise_velocity": data[:, 4],
    }


def load_reference_shape(filepath):
    """
    Load Hysing reference shape file (*s.txt).
    Format: consecutive row pairs define line segments (x, y).
    Returns an (N, 2, 2) array of N segments, each with two (x, y) endpoints.
    """
    data = np.loadtxt(filepath, comments=("#", "%"))
    n_segments = len(data) // 2
    return data[:2 * n_segments].reshape(n_segments, 2, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Hysing benchmark QoIs for the 2D rising bubble."
    )
    parser.add_argument(
        "--vtk-dir", default="VTK",
        help="Path to VTK output directory containing the .pvd file (default: VTK)"
    )
    parser.add_argument(
        "--ref", nargs="+", default=[],
        help="Reference data file(s) in Hysing format (5 columns)"
    )
    parser.add_argument(
        "--ref-label", nargs="+", default=[],
        help="Labels for reference data (one per --ref file)"
    )
    parser.add_argument(
        "--ref-shape", nargs="+", default=[],
        help="Reference shape file(s) in Hysing *s.txt format (paired row segments)"
    )
    parser.add_argument(
        "--ref-shape-label", nargs="+", default=[],
        help="Labels for reference shape files (one per --ref-shape file)"
    )
    parser.add_argument(
        "--output", "-o", default="qoi.png",
        help="Output plot filename (default: qoi.png)"
    )
    parser.add_argument(
        "--alpha-field", default="Alpha_1",
        help="Name of the bubble volume fraction field (default: Alpha_1)"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of parallel workers for reading snapshots (default: 1)"
    )
    parser.add_argument(
        "--ref-only", action="store_true",
        help="Plot only reference data (skip VTK reading)"
    )
    args = parser.parse_args()

    # Compute QoIs from VTK data (unless --ref-only)
    times = np.array([])
    circularities = np.array([])
    com_y = np.array([])
    rise_vel = np.array([])
    last_contour = None
    last_time = 0.0

    if not args.ref_only:
        # Find .pvd file
        pvd_files = glob.glob(os.path.join(args.vtk_dir, "*.pvd"))
        if not pvd_files:
            print(f"Error: no .pvd file found in {args.vtk_dir}", file=sys.stderr)
            sys.exit(1)
        pvd_path = pvd_files[0]
        print(f"Reading {pvd_path}")

        entries = parse_pvd(pvd_path)
        n_snapshots = len(entries)
        print(f"Found {n_snapshots} snapshots")

        times = np.zeros(n_snapshots)
        areas = np.zeros(n_snapshots)
        circularities = np.zeros(n_snapshots)
        com_y = np.zeros(n_snapshots)
        rise_vel = np.zeros(n_snapshots)

        pv.set_plot_theme("document")

        work_items = [(i, t, path, args.alpha_field) for i, (t, path) in enumerate(entries)]
        n_workers = min(args.jobs, n_snapshots)

        def _collect_result(result):
            nonlocal last_contour, last_time
            i, t, a, c, y, v, contour_xy = result
            times[i] = t
            areas[i] = a
            circularities[i] = c
            com_y[i] = y
            rise_vel[i] = v
            if contour_xy is not None and t >= last_time:
                last_contour = contour_xy
                last_time = t
            print(f"  [{i+1}/{n_snapshots}] t = {t:.4f}  "
                  f"area={a:.6f}  circ={c:.4f}  CoM_y={y:.4f}  V_rise={v:.6f}")

        if n_workers > 1:
            print(f"Processing with {n_workers} workers ...")
            with multiprocessing.Pool(n_workers) as pool:
                for result in pool.imap_unordered(_process_snapshot, work_items):
                    _collect_result(result)
        else:
            for item in work_items:
                _collect_result(_process_snapshot(item))

    # Load reference data
    ref_data = []
    for j, ref_file in enumerate(args.ref):
        label = args.ref_label[j] if j < len(args.ref_label) else os.path.basename(ref_file)
        print(f"Loading reference: {ref_file} ({label})")
        ref_data.append((label, load_reference(ref_file)))

    # Plot 2x2: top-left = bubble shape, others = QoI time histories
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Hysing Rising Bubble — Quantities of Interest", fontsize=14)

    ref_colors = ["tab:red", "tab:green", "tab:purple", "tab:orange"]

    # Load reference shape data
    ref_shapes = []
    for j, shape_file in enumerate(args.ref_shape):
        label = args.ref_shape_label[j] if j < len(args.ref_shape_label) else os.path.basename(shape_file)
        print(f"Loading reference shape: {shape_file} ({label})")
        ref_shapes.append((label, load_reference_shape(shape_file)))

    # Top-left: bubble contour at final time
    ax_shape = axes[0, 0]
    for j, (label, segments) in enumerate(ref_shapes):
        color = ref_colors[j % len(ref_colors)]
        for seg in segments:
            ax_shape.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.0)
        # Invisible plot for legend entry
        ax_shape.plot([], [], color=color, linewidth=1.5, label=label)
    if last_contour is not None:
        ax_shape.plot(last_contour[:, 0], last_contour[:, 1], "k.", markersize=1, label="MFC")
    ax_shape.set_xlabel("$x$")
    ax_shape.set_ylabel("$y$")
    ax_shape.set_title(f"Bubble Shape ($t = {last_time:.2f}$)" if last_contour is not None
                       else "Bubble Shape")
    ax_shape.set_aspect("equal")
    ax_shape.grid(True, alpha=0.3)
    if ref_shapes or last_contour is not None:
        ax_shape.legend(fontsize=8)

    # Remaining three panels: circularity, center of mass, rise velocity
    qoi_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    qoi_keys = ["circularity", "center_of_mass", "rise_velocity"]
    qoi_labels = ["Circularity", "Center of Mass ($y_c$)", "Rise Velocity ($V_c$)"]
    sim_data = [circularities, com_y, rise_vel]

    for idx, ax in enumerate(qoi_axes):
        for j, (label, rd) in enumerate(ref_data):
            color = ref_colors[j % len(ref_colors)]
            ax.plot(rd["time"], rd[qoi_keys[idx]], "-", color=color,
                    linewidth=1, label=label)
        if len(times) > 0:
            ax.plot(times, sim_data[idx], "k-", linewidth=1.5, label="MFC")
        ax.set_xlabel("Time")
        ax.set_ylabel(qoi_labels[idx])
        ax.set_title(qoi_labels[idx])
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 3])
        if ref_data or len(times) > 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
