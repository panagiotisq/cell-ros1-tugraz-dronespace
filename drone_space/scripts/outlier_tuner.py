import argparse

import numpy as np, open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Placeholder, overridden by --ply (e.g. the raw COLMAP dense reconstruction before
# alignment -- see drone_space/DATASET.md step 3).
PLY = "/path/to/colmap/dense/fused.ply"

class Tuner:
    def __init__(self, ply=PLY):
        self.base = o3d.io.read_point_cloud(ply)
        self.N = len(self.base.points)
        self.orig_cols = np.asarray(self.base.colors).copy() if self.base.has_colors() else \
                         np.full((self.N, 3), 0.7)

        gui.Application.instance.initialize()
        self.w = gui.Application.instance.create_window("Outlier tuner", 1600, 980)
        em = self.w.theme.font_size

        # 3D scene
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.w.renderer)
        self.scene.scene.set_background([0.05, 0.05, 0.07, 1.0])
        self.mat = rendering.MaterialRecord(); self.mat.shader = "defaultUnlit"; self.mat.point_size = 2.0

        # control panel
        panel = gui.Vert(0.5 * em, gui.Margins(0.5*em, 0.5*em, 0.5*em, 0.5*em))
        self.method = gui.Combobox()
        self.method.add_item("Radius (ROR): min nn within radius")
        self.method.add_item("Statistical (SOR): nb_neighbors + std_ratio")
        panel.add_child(gui.Label("Method")); panel.add_child(self.method)

        # ROR controls
        self.s_radius = gui.Slider(gui.Slider.DOUBLE); self.s_radius.set_limits(0.005, 0.10); self.s_radius.double_value = 0.030
        self.s_minpts = gui.Slider(gui.Slider.INT);    self.s_minpts.set_limits(1, 40);        self.s_minpts.int_value = 8
        panel.add_child(gui.Label("ROR  radius (units; median nn≈0.0038)")); panel.add_child(self.s_radius)
        panel.add_child(gui.Label("ROR  min points in radius")); panel.add_child(self.s_minpts)

        # SOR controls
        self.s_knn = gui.Slider(gui.Slider.INT);    self.s_knn.set_limits(5, 80);  self.s_knn.int_value = 20
        self.s_std = gui.Slider(gui.Slider.DOUBLE); self.s_std.set_limits(0.3, 4.0); self.s_std.double_value = 2.0
        panel.add_child(gui.Label("SOR  nb_neighbors")); panel.add_child(self.s_knn)
        panel.add_child(gui.Label("SOR  std_ratio (lower = stricter)")); panel.add_child(self.s_std)

        self.cb_removed = gui.Checkbox("show removed points in RED"); self.cb_removed.checked = True
        panel.add_child(self.cb_removed)

        btn = gui.Button("PREVIEW (recompute)"); btn.set_on_clicked(self.preview)
        panel.add_child(btn)
        self.status = gui.Label("Press PREVIEW to compute.\n")
        panel.add_child(self.status)
        panel.add_child(gui.Label("Mouse: L-drag orbit · wheel zoom · R-drag pan"))

        self.w.add_child(self.scene); self.w.add_child(panel)
        self.panel = panel
        self.w.set_on_layout(self._layout)

        self._show(np.ones(self.N, bool), None)   # start: all points
        bb = self.base.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bb, bb.get_center())

    def _layout(self, ctx):
        r = self.w.content_rect
        pw = 22 * self.w.theme.font_size
        self.scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def _show(self, keep_mask, removed_red):
        pts = np.asarray(self.base.points)
        if removed_red is None or not self.cb_removed.checked:
            idx = np.where(keep_mask)[0]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts[idx])
            cloud.colors = o3d.utility.Vector3dVector(self.orig_cols[idx])
        else:
            cols = self.orig_cols.copy()
            cols[~keep_mask] = [1.0, 0.0, 0.0]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud.colors = o3d.utility.Vector3dVector(cols)
        self.scene.scene.remove_geometry("pc")
        self.scene.scene.add_geometry("pc", cloud, self.mat)

    def preview(self):
        self.status.text = "computing...\n"
        if self.method.selected_index == 0:
            r = self.s_radius.double_value; mp = self.s_minpts.int_value
            _, ind = self.base.remove_radius_outlier(nb_points=mp, radius=r)
            desc = "ROR radius=%.4f min_pts=%d" % (r, mp)
        else:
            k = self.s_knn.int_value; sr = self.s_std.double_value
            _, ind = self.base.remove_statistical_outlier(nb_neighbors=k, std_ratio=sr)
            desc = "SOR nb_neighbors=%d std_ratio=%.2f" % (k, sr)
        keep = np.zeros(self.N, bool); keep[np.asarray(ind)] = True
        nrem = self.N - keep.sum()
        self._show(keep, True)
        msg = "%s\nkept %d / removed %d  (%.3f%%)\n" % (desc, keep.sum(), nrem, 100*nrem/self.N)
        self.status.text = msg
        print(msg, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=PLY, help="raw dense point cloud to clean (.ply)")
    args = ap.parse_args()
    Tuner(args.ply)
    gui.Application.instance.run()
