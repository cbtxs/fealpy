"""Generate cubic Lagrange-basis figures used by the Hu--Zhang slides."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


OUT = Path("presentations/hu_zhang_metaprogramming/figures")
COLORS = ["#4c78a8", "#59a14f", "#e15759"]


def ell(m, x):
    value = np.ones_like(x, dtype=float)
    for j in range(m):
        value *= 3 * x - j
    return value / math.factorial(m)


def phi(alpha, bary):
    value = np.ones(bary.shape[:-1])
    for i, m in enumerate(alpha):
        value *= ell(m, bary[..., i])
    return value


def sample_triangle(vertices, n=44):
    bary = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            bary.append((i / n, j / n, 1 - (i + j) / n))
    bary = np.asarray(bary)
    xy = bary @ vertices
    return bary, xy, mtri.Triangulation(xy[:, 0], xy[:, 1])


def p3_nodes(vertices):
    bary = np.asarray([(i / 3, j / 3, 1 - (i + j) / 3)
                       for i in range(4) for j in range(4 - i)])
    return bary @ vertices


def style_3d(ax, zoom=1.30):
    ax.view_init(elev=25, azim=-55)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, .9), zoom=zoom)
    ax.set_axis_off()
    ax.set_zlim(-0.55, 1.15)


def draw_base(ax, vertices):
    closed = np.vstack((vertices, vertices[0]))
    ax.plot(closed[:, 0], closed[:, 1], np.zeros(4), color="#222222", lw=1.1)
    ax.plot_trisurf(*sample_triangle(vertices, 2)[1].T, np.zeros(6),
                    color="#d8d8d8", alpha=.25, linewidth=0)
    nodes = p3_nodes(vertices)
    ax.scatter(nodes[:, 0], nodes[:, 1], np.zeros(len(nodes)), color="red", s=13)


def plot_surface(ax, vertices, alpha, color, opacity=.55):
    bary, xy, tri = sample_triangle(vertices)
    z = phi(alpha, bary)
    ax.plot_trisurf(tri, z, color=color, alpha=opacity, linewidth=.12,
                    edgecolor="white", antialiased=True)
    draw_base(ax, vertices)


def basis_figure():
    vertices = np.asarray([[0., 0.], [1., 0.], [.5, np.sqrt(3) / 2]])
    data = [((3, 0, 0), r"$\phi_{300}$"),
            ((2, 1, 0), r"$\phi_{210}$"),
            ((1, 1, 1), r"$\phi_{111}$")]
    fig = plt.figure(figsize=(11.2, 3.8))
    for i, ((alpha, title), color) in enumerate(zip(data, COLORS), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        plot_surface(ax, vertices, alpha, color)
        ax.set_title(title, fontsize=16, pad=4)
        style_3d(ax)
    fig.tight_layout()
    fig.savefig(OUT / "lagrange_p3_basis.pdf", bbox_inches="tight")
    plt.close(fig)


def edge_gluing():
    left = np.asarray([[-1.55, 0.], [0., 0.], [0., 1.35]])
    right = np.asarray([[.15, 0.], [1.70, 0.], [.15, 1.35]])
    fig = plt.figure(figsize=(5.0, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    plot_surface(ax, left, (0, 2, 1), COLORS[0], opacity=.80)
    plot_surface(ax, right, (2, 0, 1), COLORS[0], opacity=.80)
    ax.set_title(r"$P_3$ edge basis: $\phi^-_{021}\ \leftrightarrow\ \phi^+_{201}$",
                 fontsize=13, pad=2)
    style_3d(ax, zoom=1.48)
    fig.savefig(OUT / "lagrange_gluing_edge.pdf", bbox_inches="tight")
    plt.close(fig)


def node_gluing():
    fig = plt.figure(figsize=(5.0, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    for theta in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        direction = np.array([np.cos(theta), np.sin(theta)])
        left = np.array([np.cos(theta + .58), np.sin(theta + .58)])
        right = np.array([np.cos(theta - .58), np.sin(theta - .58)])
        vertices = np.asarray([.14 * direction, 1.28 * left, 1.28 * right])
        plot_surface(ax, vertices, (3, 0, 0), COLORS[2], opacity=.80)
    ax.set_title(r"$P_3$ vertex basis on separated cells", fontsize=13, pad=2)
    style_3d(ax, zoom=1.48)
    fig.savefig(OUT / "lagrange_gluing_node.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    basis_figure()
    edge_gluing()
    node_gluing()
