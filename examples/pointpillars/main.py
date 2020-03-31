import numpy as np
import pyvista as pv
import torch.utils.data as data
import torch3d.datasets as datasets


if __name__ == "__main__":
    dataset = datasets.KITTIDetection("/Volumes/SS1TBN/data/KITTI")
    print(len(dataset))

    dataloader = data.DataLoader(
        dataset, batch_size=2, collate_fn=datasets.KITTIDetection.collate_fn
    )
    print(len(dataloader))
    for inputs, target in dataloader:
        print(target)

    pv.set_plot_theme("ParaView")
    for inputs, target in dataset:
        lidar = inputs["lidar"]
        plt = pv.Plotter()
        mesh = pv.PolyData(lidar[:, 0:3])
        mesh["intensity"] = lidar[:, 3]
        plt.add_mesh(mesh, render_points_as_spheres=True)
        plt.add_axes_at_origin()

        def roty(angle):
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        for i in range(len(target["class"])):
            print(target["class"][i])
            R = roty(target["yaw"][i])
            h, w, l = target["size"][i, 0:3]
            x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y = [0, 0, 0, 0, -h, -h, -h, -h]
            z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            obb = np.vstack([x, y, z])
            obb = np.dot(R, np.vstack([x, y, z]))
            obb = obb.T + target["center"][i]

            bounds = np.c_[np.amin(obb, axis=0), np.amax(obb, axis=0)]
            bounds = np.ravel(bounds, order="C")
            box = pv.Box(bounds)
            plt.add_mesh(box, style="wireframe", line_width=8.0)

        plt.enable_eye_dome_lighting()
        plt.show()
        plt.close()
