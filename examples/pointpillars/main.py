import pyvista as pv


if __name__ == "__main__":
    dataset = KITTIDetection("data/KITTI", rectified=True)

    pv.set_plot_theme("night")
    for inputs, target in dataset:
        lidar = inputs["lidar"]
        plt = pv.Plotter()
        mesh = pv.PolyData(lidar[:, 0:3])
        mesh["intensity"] = lidar[:, 3]
        plt.add_mesh(mesh, render_points_as_spheres=True, cmap="viridis")
        plt.add_axes_at_origin()

        def roty(angle):
            c = np.cos(angle)
            s = np.sin(angle)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        for i in range(len(target["class"])):
            if target["class"][i] == "DontCare":
                continue

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

        break
