#!/usr/bin/env python3
'''
Author: Jimn gaoqianmain@gmail.com
Date: 2026-05-22 14:43:52
LastEditors: Jimn gaoqianmain@gmail.com
LastEditTime: 2026-05-22 15:24:14
FilePath: /calib_mono_cam_manual/visualize/visualize_cam_poses.py
Description: 可视化位姿3D
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def read_camera_poses(file_path):
    R_list = []
    T_list = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split()
            if len(parts) != 13:
                continue

            R = np.array([float(x) for x in parts[1:10]]).reshape(3, 3)
            
            t = np.array([float(x) for x in parts[10:13]])

            R_list.append(R)
            T_list.append(t)

    return R_list, T_list

def compute_camera_position(R, t):
    return -R.T @ t


def draw_camera_axes(ax, position, R, scale=0.3, label=None):
    R_t = R.T
    x_axis = R_t[:, 0] * scale
    y_axis = R_t[:, 1] * scale
    z_axis = R_t[:, 2] * scale

    ax.quiver(position[0], position[1], position[2], x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.05, linewidth=2)
    ax.quiver(position[0], position[1], position[2], y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.05, linewidth=2)
    ax.quiver(position[0], position[1], position[2], z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.05, linewidth=2)

    ax.scatter(position[0], position[1], position[2], 'red', s=50, marker='o')
    if label is not None:
        ax.text(position[0], position[1], position[2] + 0.1, label, fontsize=8, ha='center')


def draw_calibration_board(ax:Axes3D, board_size=1.0):
    xx, yy = np.meshgrid(np.linspace(-board_size / 2, board_size / 2, 10),
                         np.linspace(-board_size / 2, board_size / 2, 10))
    
    zz = np.zeros_like(xx)

    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    corners = np.array([
        [-board_size/2, -board_size/2, 0],
        [board_size/2, -board_size/2, 0],
        [board_size/2, board_size/2, 0],
        [-board_size/2, board_size/2, 0],
        [-board_size/2, -board_size/2, 0],
    ])

    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'k-', linewidth=2)

    ax.scatter(0,0,0,c='black',s=100, marker='x', linewidth=3)
    ax.text(0.1,0.1,0,"origin", fontsize=10)

def visualize_3d(R_list, T_list, board_size=0.2):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    camera_position = []
    for i, (R, t) in enumerate(zip(R_list, T_list)):
        pos = compute_camera_position(R, t)
        camera_position.append(pos)
    
    camera_position = np.array(camera_position)

    if len(camera_position) > 0:
        max_range = np.max(np.abs(camera_position)) * 1.5
        max_range = max(max_range, board_size)
    else:
        max_range = board_size

    draw_calibration_board(ax, board_size)

    for i, (R, t) in enumerate(zip(R_list, T_list)):
        position = compute_camera_position(R, t)
        draw_camera_axes(ax, position, R, scale=max_range*0.15, label=f"Cam {i}")

    ax.quiver(0, 0, 0, max_range * 0.3, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=3)
    ax.quiver(0, 0, 0, 0, max_range * 0.3, 0, color='g', arrow_length_ratio=0.1, linewidth=3)
    ax.quiver(0, 0, 0, 0, 0, max_range * 0.3, color='b', arrow_length_ratio=0.1, linewidth=3)

    ax.text(max_range * 0.3, 0, 0, "X", fontsize=12, color="r")
    ax.text(0, max_range * 0.3, 0, "Y", fontsize=12, color="g")
    ax.text(0, 0, max_range * 0.3, "Z", fontsize=12, color="b")

    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    return fig, ax


def create_interactive_visualization(R_list, T_list, board_size=1.0):
    fig, ax = visualize_3d(R_list, T_list, board_size)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="相机位姿3D可视化工具")
    parser.add_argument('pose_file', type=str, help="相机位姿文件路径")
    parser.add_argument('--board_size', type=float, default=1.0, help="板子的尺寸")
    parser.add_argument('--save', type=str, default=None)

    arg = parser.parse_args()

    print(f"读取相机位姿文件 {arg.pose_file}")
    R_list, T_list = read_camera_poses(arg.pose_file)
    print(f"读取到{len(R_list)}个数据")

    fig, ax = visualize_3d(R_list, T_list, arg.board_size)

    if arg.save:
        plt.savefig(arg.save, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
if __name__ == "__main__":
    main()
    

