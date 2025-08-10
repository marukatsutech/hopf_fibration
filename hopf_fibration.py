""" Hopf fibration """
import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt

""" Global variables """
num_points = 48
colors = [plt.cm.gist_rainbow(i/(num_points - 1)) for i in range(num_points)]
# colors = colors[::-1]
plt_fibers = []
plot_points = []

""" Animation control """
is_play = False
is_tilt = False

""" Axis vectors """

""" Create figure and axes """
title_ax0 = "Stereographic projection of S^3"
title_ax1 = "Point on S^2"
title_tk = "Hopf fibration"

x_min = -4.
x_max = 4.
y_min = -4.
y_max = 4.
z_min = -4.
z_max = 4.

fig = Figure(facecolor='black')
ax0 = fig.add_subplot(121, projection="3d")
ax0.set_box_aspect((4, 4, 4))
ax0.grid()
ax0.set_title(title_ax0, color="white")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")
ax0.set_xlim(x_min, x_max)
ax0.set_ylim(y_min, y_max)
ax0.set_zlim(z_min, z_max)

ax0.set_facecolor("black")
ax0.axis('off')

x_min = -2.
x_max = 2.
y_min = -2.
y_max = 2.
z_min = -2.
z_max = 2.

ax1 = fig.add_subplot(122, projection="3d")
ax1.set_box_aspect((4, 4, 4))
ax1.grid()
ax1.set_title(title_ax1, color="white")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_zlim(z_min, z_max)

ax1.set_facecolor("black")
ax1.axis('off')

""" Embed in Tkinter """
root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill="both")

toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()


def on_move(event):
    if event.inaxes == ax0:
        ax1.view_init(elev=ax0.elev, azim=ax0.azim)
    elif event.inaxes == ax1:
        ax1.view_init(elev=ax1.elev, azim=ax1.azim)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect('motion_notify_event', on_move)

""" Global objects of Tkinter """

""" Classes and functions """


class Counter:
    def __init__(self, is3d=None, ax=None, xy=None, z=None, label=""):
        self.is3d = is3d if is3d is not None else False
        self.ax = ax
        self.x, self.y = xy[0], xy[1]
        self.z = z if z is not None else 0
        self.label = label

        self.count = 0

        if not is3d:
            self.txt_step = self.ax.text(self.x, self.y, self.label + str(self.count))
        else:
            self.txt_step = self.ax.text2D(self.x, self.y, self.label + str(self.count))
            self.xz, self.yz, _ = proj3d.proj_transform(self.x, self.y, self.z, self.ax.get_proj())
            self.txt_step.set_position((self.xz, self.yz))

    def count_up(self):
        self.count += 1
        self.txt_step.set_text(self.label + str(self.count))

    def reset(self):
        self.count = 0
        self.txt_step.set_text(self.label + str(self.count))

    def get(self):
        return self.count


class UnitSphere:
    def __init__(self, ax, cmap, edge_color, alpha):
        self.ax = ax
        self.cmap = cmap
        self.edge_color = edge_color
        self.alpha = alpha

        # Define the range of parameters
        phi = np.linspace(0, np.pi, 50)  # Elevation angle
        theta = np.linspace(0, 2 * np.pi, 50)  # Azimuth angle
        PHI, THETA = np.meshgrid(phi, theta)

        # Unit sphere
        X = np.sin(PHI) * np.cos(THETA)  # X-coordinate
        Y = np.sin(PHI) * np.sin(THETA)  # Y-coordinate
        Z = np.cos(PHI)  # T-coordinate

        # Create plot
        surface = self.ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor=self.edge_color, alpha=self.alpha)


def get_base_points_circle_vertical():
    points = []
    ph = 0

    for i in range(num_points):
        th = i / num_points * np.pi * 2
        points.append((th, ph))

    return points


def get_base_points_small_circle(num_points, center_theta=np.pi/2, center_phi=0, angle_radius=np.pi/6):
    points = []
    for i in range(num_points):
        angle = i / num_points * 2 * np.pi  # Angle around the circle

        # Compute the polar angle (theta) of the point on the small circle
        theta = np.arccos(np.cos(angle_radius) * np.cos(center_theta) +
                          np.sin(angle_radius) * np.sin(center_theta) * np.cos(angle))
        phi = center_phi + np.arctan2(np.sin(angle) * np.sin(angle_radius),
                                      np.cos(angle_radius) * np.sin(center_theta) -
                                      np.sin(angle_radius) * np.cos(center_theta) * np.cos(angle))
        points.append((theta, phi))
    return points


def get_base_points_circle_horizontal(theta):
    points = []
    th = theta

    for i in range(num_points):
        ph = i / num_points * np.pi * 2
        points.append((th, ph))

    return points


def get_base_points_spiral():
    points = []

    for i in range(num_points):
        th = i / num_points * np.pi
        ph = i / num_points * np.pi * 2 * 4
        points.append((th, ph))

    return points


def get_base_points_fibonacci_sphere(num_points):
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))  # ≈ 2.39996

    for i in range(num_points):
        z = 1 - 2 * i / (num_points - 1)  # z ∈ [-1, 1]
        theta = np.arccos(z)  # θ: polar angle
        phi = golden_angle * i  # φ: azimuthal angle
        points.append((theta, phi))

    return points


def hopf_preimage(theta, phi, t):
    """
    Given a point (theta, phi) on S^2, return the stereographic projection
    coordinates in R^3 of a point on the corresponding Hopf fiber (a circle in S^3).

    Construction:
      - Represent a point on S^3 as (z1, z2) in C^2.
      - For the chosen base point on S^2, construct one preimage (z1_base, z2_base).
      - Multiply by a common phase e^{i t} to trace the entire fiber.
      - Apply stereographic projection from the north pole (x4 = 1) to R^3.
    """
    alpha = theta / 2.0
    z1_base = np.cos(alpha) * np.exp(1j * 0.0)
    z2_base = np.sin(alpha) * np.exp(-1j * phi)

    # Rotate along the fiber direction by phase t
    z1 = z1_base * np.exp(1j * t)
    z2 = z2_base * np.exp(1j * t)

    # Coordinates in R^4: (x1, x2, x3, x4)
    x1 = np.real(z1)
    x2 = np.imag(z1)
    x3 = np.real(z2)
    x4 = np.imag(z2)

    # Stereographic projection from north pole (x4 = 1) to R^3
    denom = 1.0 - x4
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)

    return np.array([x1 / denom, x2 / denom, x3 / denom])


def plot_hopf_fibers():
    global plt_fibers, plot_points
    for (theta, phi), color in zip(base_points, colors):
        # Plot each fiber
        pts = np.array([hopf_preimage(theta, phi, t) for t in t_vals])
        fiber, = ax0.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.5, color=color)
        plt_fibers.append(fiber)

        # Plot the base points on S^2 (radius=1)
        x_s2 = np.sin(theta) * np.cos(phi)
        y_s2 = np.sin(theta) * np.sin(phi)
        z_s2 = np.cos(theta)
        point = ax1.scatter(x_s2, y_s2, z_s2, s=50, edgecolor='white', color=color)
        plot_points.append(point)

        """
        ax1.text(x_s2, y_s2, z_s2 + 0.1,f"θ={theta:.2f}, φ={phi:.2f}",
                 color=color, fontsize=8, ha='center')
        """


def remove_hopf_fibers():
    global plt_fibers, plot_points
    for plt_fiber in plt_fibers:
        plt_fiber.remove()
        plt_fibers = []
    for plt_point in plot_points:
        plt_point.remove()
        plot_points = []


def option_base_points_selected(event):
    global base_points
    if combo_options.get() == option_base_points[0]:
        remove_hopf_fibers()
        base_points = get_base_points_circle_vertical()
        plot_hopf_fibers()
    elif combo_options.get() == option_base_points[1]:
        remove_hopf_fibers()
        base_points = get_base_points_circle_horizontal(np.pi / 2)
        plot_hopf_fibers()
    elif combo_options.get() == option_base_points[2]:
        remove_hopf_fibers()
        base_points = get_base_points_spiral()
        plot_hopf_fibers()
    elif combo_options.get() == option_base_points[3]:
        remove_hopf_fibers()
        base_points = get_base_points_fibonacci_sphere(num_points)
        plot_hopf_fibers()
    else:
        pass


def create_parameter_setter():
    pass


def create_animation_control():
    frm_anim = ttk.Labelframe(root, relief="ridge", text="Animation", labelanchor="n")
    frm_anim.pack(side="left", fill=tk.Y)
    btn_play = tk.Button(frm_anim, text="Play/Pause", command=switch)
    btn_play.pack(side="left")
    btn_reset = tk.Button(frm_anim, text="Reset", command=reset)
    btn_reset.pack(side="left")
    # btn_clear = tk.Button(frm_anim, text="Clear path", command=lambda: aaa())
    # btn_clear.pack(side="left")


def create_center_lines():
    ln_axis_x = art3d.Line3D([x_min, x_max], [0., 0.], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(ln_axis_x)
    ln_axis_y = art3d.Line3D([0., 0.], [y_min, y_max], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(ln_axis_y)
    ln_axis_z = art3d.Line3D([0., 0.], [0., 0.], [z_min, z_max], color="gray", ls="-.", linewidth=1)
    ax0.add_line(ln_axis_z)

    ln_axis_x1 = art3d.Line3D([x_min, x_max], [0., 0.], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax1.add_line(ln_axis_x1)
    ln_axis_y1 = art3d.Line3D([0., 0.], [y_min, y_max], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax1.add_line(ln_axis_y1)
    ln_axis_z1 = art3d.Line3D([0., 0.], [0., 0.], [z_min, z_max], color="gray", ls="-.", linewidth=1)
    ax1.add_line(ln_axis_z1)


def draw_static_diagrams():
    create_center_lines()


def update_diagrams():
    pass


def reset():
    global is_play
    # cnt.reset()
    if is_play:
        is_play = not is_play


def switch():
    global is_play
    is_play = not is_play


def update(f):
    if is_play:
        # cnt.count_up()
        update_diagrams()


""" main loop """
if __name__ == "__main__":
    # cnt = Counter(ax=ax0, is3d=True, xy=np.array([x_min, y_max]), z=z_max, label="Step=")
    draw_static_diagrams()
    # create_animation_control()
    create_parameter_setter()

    unit_sphere0 = UnitSphere(ax0, "plasma", "none", 0.3)
    unit_sphere1 = UnitSphere(ax1, "plasma", "none", 0.3)

    # Plot Hopf fibration and points
    t_vals = np.linspace(0, 2 * np.pi, 800)
    base_points = get_base_points_circle_vertical()
    plot_hopf_fibers()

    # ax0.legend(loc='lower right', fontsize=8)

    # Select base points
    frm_options = ttk.Labelframe(root, relief="ridge", text="Base points option", labelanchor="n")
    frm_options.pack(side="left", fill=tk.Y)
    option_base_points = ["Vertical circle", "horizontal circle", "Spiral", "fibonacci"]
    variable_base_points_option = tk.StringVar(root)
    combo_options = ttk.Combobox(frm_options, values=option_base_points, textvariable=variable_base_points_option,
                                 width=18)
    combo_options.set(option_base_points[0])
    combo_options.bind("<<ComboboxSelected>>", option_base_points_selected)
    combo_options.pack()

    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()