"""Draw a 3D tetrahedron and tangent/normal frames on its subsimplices.

Run from the repository root, for example:

    python test/visualization/draw_subsimplex_frames.py \
        --output presentations/hu_zhang_metaprogramming/figures/frame3d.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#1565c0"  # tangent directions
RED = "#c62828"   # normal directions
BLACK = "#111111"


def draw_tetrahedron(ax):
    """A genuine 3D regular tetrahedron and frames on its subsimplices."""

    # These four points have identical pairwise distances: a regular tetrahedron.
    vertices = np.array([
        [0.0, 0.0, 1.0],
        [2.0 * np.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0],
        [-np.sqrt(2.0) / 3.0, np.sqrt(2.0 / 3.0), -1.0 / 3.0],
        [-np.sqrt(2.0) / 3.0, -np.sqrt(2.0 / 3.0), -1.0 / 3.0],
    ])
    v0, v1, v2, v3 = vertices

    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ax.plot(
            *zip(vertices[i], vertices[j]), color=BLACK, lw=1.7,
            ls="--" if (i, j) == (1, 2) else "-",
        )
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=BLACK, s=20)
    for i, v in enumerate(vertices):
        ax.text(*(v + 0.08), rf"$v_{i}$", fontsize=9)

    def arrow3(p, direction, text, color):
        ax.quiver(*p, *direction, color=color, linewidth=1.8,
                  arrow_length_ratio=.18, normalize=False)
        ax.text(*(p + 1.12 * direction), text, color=color, fontsize=9)

    # Vertex v0: all three directions are normal directions.
    for direction, text in (([.42, 0, 0], r"$n_1$"),
                            ([0, .42, 0], r"$n_2$"),
                            ([0, 0, .42], r"$n_3$")):
        arrow3(v0, np.asarray(direction), text, RED)

    # Edge e01: tangent t and an orthogonal normal pair n1, n2.
    e = (v0 + v1) / 2
    t = (v1 - v0) / np.linalg.norm(v1 - v0)
    n1 = np.cross(t, v2 - v0); n1 /= np.linalg.norm(n1)
    n2 = np.cross(t, n1); n2 /= np.linalg.norm(n2)
    arrow3(e, .42 * t, r"$t$", BLUE)
    arrow3(e, .42 * n1, r"$n_1$", RED)
    arrow3(e, .42 * n2, r"$n_2$", RED)

    # Face F1=[v0,v2,v3], opposite to v1: two tangential directions and one normal.
    f = (v0 + v2 + v3) / 3
    tf1 = (v2 - v0) / np.linalg.norm(v2 - v0)
    face_n = np.cross(v2 - v0, v3 - v0); face_n /= np.linalg.norm(face_n)
    tf2 = np.cross(face_n, tf1); tf2 /= np.linalg.norm(tf2)
    arrow3(f, .42 * tf1, r"$t_1$", BLUE)
    arrow3(f, .42 * tf2, r"$t_2$", BLUE)
    arrow3(f, .42 * face_n, r"$n$", RED)
    ax.text(*(f + .10 * face_n), r"$F_1$", fontsize=9)

    # Cell interior: every direction is tangential to the 3-simplex.
    k = vertices.mean(axis=0)
    for direction, text in (([.42, 0, 0], r"$t_1$"),
                            ([0, .42, 0], r"$t_2$"),
                            ([0, 0, .42], r"$t_3$")):
        arrow3(k, np.asarray(direction), text, BLUE)
    ax.text(*(k - .20), r"$K$", fontsize=10)

    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-115)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("frame3d.pdf"))
    parser.add_argument("--show", action="store_true", help="Open an interactive Matplotlib window")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.2, 5.2))
    ax3 = fig.add_subplot(1, 1, 1, projection="3d")
    draw_tetrahedron(ax3)
    fig.savefig(args.output, dpi=250, bbox_inches="tight", facecolor="white")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
