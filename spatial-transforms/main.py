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


def transform_batch_einsum() -> None:
    pass


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
    einsum_parser.set_defaults(func=lambda args: transform_batch_einsum())

    inverse_parser = subparsers.add_parser(
        "inverse", help="Run the transform inverse example."
    )
    inverse_parser.set_defaults(func=lambda args: transform_inverse())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
