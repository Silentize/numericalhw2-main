import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def plot_iterations(program_type: str,inner_obj_values: list[float],outer_obj_values: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(range(len(inner_obj_values)), inner_obj_values, label="Inner Values", color="blue", linestyle="-",
            marker="o")
    ax.plot(range(len(outer_obj_values)), outer_obj_values, label="Outer Values", color="green", linestyle="--",
            marker="x")

    ax.legend()
    ax.set_title(f"Objective Function Values of {program_type} Program", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Iterations", fontsize=12)
    ax.set_ylabel("Objective Function Value", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def plot_feasible_region_qp(path_points: list[NDArray]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    path_array = np.array(path_points)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color="lightblue", alpha=0.7)
    ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], color="red", marker="^", linestyle="--", label="Path")
    ax.scatter(path_array[-1][0], path_array[-1][1], path_array[-1][2], s=60, c="purple", label="Final Candidate")

    ax.set_title("Feasible Region and Path of Quadratic Program", fontsize=14, fontweight='bold')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    ax.legend()
    ax.view_init(30, 120)
    plt.show()


def plot_feasible_region_lp(path_points: list[NDArray]) -> None:
    x_values = np.linspace(-2, 4, 300)
    x_mesh, y_mesh = np.meshgrid(x_values, x_values)
    plt.imshow(
        ((y_mesh >= -x_mesh + 1) & (y_mesh <= 1) & (x_mesh <= 2) & (y_mesh >= 0)).astype(int),
        extent=(x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()),
        origin="lower",
        cmap="Wistia",
        alpha=0.3,
    )

    x_plot = np.linspace(0, 4, 2000)
    y1 = -x_plot + 1
    y2 = np.ones(x_plot.size)
    y3 = np.zeros(x_plot.size)

    x_path = [point[0] for point in path_points]
    y_path = [point[1] for point in path_points]
    plt.plot(x_path, y_path, label="Path", color="blue", marker="s", linestyle="--")
    plt.scatter(x_path[-1], y_path[-1], label="Final Candidate", color="orange", s=60, zorder=3)

    plt.plot(x_plot, y1, color='brown', label="y = -x + 1")
    plt.plot(x_plot, y2, color='olive', label="y = 1")
    plt.plot(x_plot, y3, color='teal', label="y = 0")
    plt.plot(np.ones(x_plot.size) * 2, x_plot, color='violet', label="x = 2")

    plt.xlim(0, 3.1)
    plt.ylim(0, 2)
    plt.legend()
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title("Feasible Region and Path of Linear Program", fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
