"""Spatial transforms explanations."""

import time
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import argparse
from scipy.spatial.transform import Rotation
from rich.console import Console

console = Console()


def transform_multiple_frames(
    n_frames: int,
    frame_size: float = 0.1,
    seed: int = 42,
    label: bool = False,
    label_scale: float = 5.0,
) -> None:
    np.random.seed(seed)

    console.print(f"Creating {n_frames} frames!")
    transforms = []

    # Start with the origin to ground each frame.
    geometries = [
        (
            "origin",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 1.5),
        )
    ]
    label_specs = [("origin", np.zeros(3))]

    for n in range(n_frames):
        tf_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        tf = np.eye(4)
        tf[:3, :3] = Rotation.from_euler("XYZ", np.random.rand(3)).as_matrix()
        tf[:3, 3] = np.random.rand(3)

        tf_new = tf.copy()
        if len(transforms) > 0:
            tf_new = transforms[-1] @ tf_new

        console.print(f"Transform {n} has \n\tR={tf[:3, :3]}\n\tt={tf[:3, 3]}")

        transforms.append(tf)

        tf_geom.transform(tf_new)
        geometries.append((f"transform_{n}", tf_geom))
        label_specs.append((f"T{n}", tf_new[:3, 3].copy()))

    # Label3D is part of Open3D's GUI scene API, so labels must be created through
    # the singleton GUI application and a GUI-backed visualizer instead of draw_geometries().
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Spatial Transforms", 1024, 768)

    for name, geometry in geometries:
        vis.add_geometry(name, geometry)

    if label:
        for text, position in label_specs:
            label = vis.add_3d_label(position, text)
            if isinstance(label, gui.Label3D):
                label.scale = label_scale

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def transform_batch_einsum(
    normal: np.ndarray,
    point: np.ndarray,
    min_distance: float = 0.2,
    max_distance: float = 1.0,
    n_samples: int = 10,
    n_standoff_samples: int = 10,
    cone_angle: float = 0.15,
    vis: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    start_time = time.time()

    # 1. We start by sampling points on the unit hemisphere
    phi = np.random.uniform(0, 2 * np.pi, size=n_samples)
    theta = np.random.uniform(np.pi / 2 - cone_angle, np.pi / 2, size=n_samples)
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    approach = -np.array([x, y, z]).T

    # 2. Create coordinate frames that correspond to the sampled approach directions
    axis_z = approach
    one = np.ones(axis_z.shape[0])
    zero = np.zeros_like(one)

    # 2.1 Create the Y axis as the cross product between the normal and
    # some default vector
    axis_y = np.cross(axis_z, np.stack([zero, zero, one], axis=-1))
    mask_y = np.linalg.norm(axis_y, axis=1) < 1e-6
    axis_y[mask_y] = np.array([0, 1, 0])

    # 2.2 Normalize the new axis
    axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)

    # 2.3 Create the X axis via cross product of the y-axis and the z-axis
    axis_x = np.cross(axis_y, axis_z)
    axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)

    # 2.4 Stack the axes
    rotations = np.stack([axis_x, axis_y, axis_z], axis=-1)

    # 2.5 Assemble into a tensor of size (num_samples, 4, 4)
    frames = np.tile(np.eye(4), reps=(n_samples, 1, 1))
    frames[:, :3, :3] = rotations

    # 3. Create the normal's corresponding coordinate frame
    normal_axis_z = normal
    normal_axis_y = np.cross(normal_axis_z, np.array([0, 0, 1]))
    if np.linalg.norm(normal_axis_y) < 1e-6:
        normal_axis_y = np.array([0, 1, 0])

    normal_axis_y = normal_axis_y / np.linalg.norm(normal_axis_y)

    normal_axis_x = np.cross(normal_axis_y, normal_axis_z)
    normal_axis_x = normal_axis_x / np.linalg.norm(normal_axis_x)

    rotations = np.stack([normal_axis_x, normal_axis_y, normal_axis_z]).T
    normal_frame = np.eye(4)
    normal_frame[:3, :3] = rotations
    normal_frame[:3, 3] = point

    # 4. Move all of the transforms to a new frame in batch.
    frames = np.einsum("ij,bjk->bik", normal_frame, frames, optimize="path")

    # 5. Apply standoff
    standoffs = np.linspace(min_distance, max_distance, n_standoff_samples)

    translations = np.tile(np.eye(4), reps=(len(standoffs), 1, 1))
    translations[:, 2, 3] = -standoffs
    tiled_poses = np.kron(frames, np.ones((n_standoff_samples, 1, 1)))
    tiled_standoff_poses = np.tile(translations, reps=(len(frames), 1, 1))
    standoff_poses = np.einsum(
        "bij,bjk->bik", tiled_poses, tiled_standoff_poses, optimize="path"
    )

    console.print(
        f"Transformed approach frames of shape {frames.shape} to the normal frame in {1000 * (time.time() - start_time)}ms"
    )

    if vis:
        start_time = time.time()
        origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        normal_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        normal_mesh.transform(normal_frame)
        approach_frames = []
        for f in standoff_poses:
            fm = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            fm.transform(f)
            approach_frames.append(fm)
        geometries = [origin_mesh, *approach_frames, normal_mesh]

        o3d.visualization.draw_geometries(geometries)
        console.print(f"Prepared vis in {time.time() - start_time}s")


def transform_inverse() -> None:
    rotation = Rotation.from_euler("XYZ", np.random.rand(3)).as_matrix()
    translation = np.random.rand(3)

    tf = np.eye(4)
    tf[:3, :3] = rotation
    tf[:3, 3] = translation

    tf_inv = np.eye(4)
    tf_inv[:3, :3] = rotation.T
    tf_inv[:3, 3] = -rotation.T @ translation

    console.print(f"R={rotation}\nR^T={rotation.T}")
    console.print(f"t={translation}\n-R^T t={-rotation.T @ translation}")

    console.print(
        f"[green]Inverses: \n{np.linalg.inv(tf)}\n{tf_inv}\nMatch: {np.allclose(np.linalg.inv(tf), tf_inv)}"
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    frames_parser = subparsers.add_parser(
        "frames", help="Visualize chained transform frames."
    )
    frames_parser.add_argument(
        "n_frames",
        type=int,
        default=4,
        help="Number of frames to display in the viewer",
    )
    frames_parser.add_argument(
        "-f",
        "--frame-size",
        type=float,
        default=0.2,
        help="The scale to apply to the transform frame geometry",
    )
    frames_parser.add_argument(
        "-l",
        "--label",
        action="store_true",
        help="Display labels for the generated transforms.",
    )
    frames_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for random transform generation.",
    )
    frames_parser.add_argument(
        "--label-scale",
        type=float,
        default=5.0,
        help="Scale factor applied to Label3D text.",
    )
    frames_parser.set_defaults(
        func=lambda args: transform_multiple_frames(
            args.n_frames,
            frame_size=args.frame_size,
            seed=args.seed,
            label=args.label,
            label_scale=args.label_scale,
        )
    )

    einsum_parser = subparsers.add_parser(
        "einsum", help="Run the batch transform einsum example."
    )
    einsum_parser.add_argument(
        "-n",
        "--normal",
        type=float,
        nargs=3,
        default=[0, 0, 1],
        help="The normal direction in the global frame.",
    )
    einsum_parser.add_argument(
        "-p",
        "--point",
        type=float,
        default=[0, 0, 0],
        help="The location of the point in the world frame",
    )
    einsum_parser.add_argument(
        "-md",
        "--min_distance",
        type=float,
        default=0.2,
        help="The minimum standoff distance of the approach frames",
    )
    einsum_parser.add_argument(
        "-xd",
        "--max_distance",
        type=float,
        default=1.0,
        help="The maximum standoff distance of the approach frames",
    )
    einsum_parser.add_argument(
        "-s",
        "--num_samples",
        type=int,
        default=10,
        help="The total number of random approach directions.",
    )
    einsum_parser.add_argument(
        "-ss",
        "--num_standoff_samples",
        type=int,
        default=10,
        help="The total number of standoff samples",
    )
    einsum_parser.add_argument(
        "-c",
        "--cone_angle",
        type=float,
        default=0.15,
        help="The cone angle used when calculating the hemispherical point samples",
    )
    einsum_parser.add_argument(
        "-nv",
        "--no_vis",
        action="store_false",
        help="Turn off the visualization with Open3D.",
    )
    einsum_parser.set_defaults(
        func=lambda args: transform_batch_einsum(
            normal=args.normal,
            point=args.point,
            n_samples=args.num_samples,
            n_standoff_samples=args.num_standoff_samples,
            cone_angle=args.cone_angle,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            vis=args.no_vis,
        )
    )

    inverse_parser = subparsers.add_parser(
        "inverse", help="Run the transform inverse example."
    )
    inverse_parser.set_defaults(func=lambda args: transform_inverse())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
