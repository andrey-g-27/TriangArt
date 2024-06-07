import math as m

import numpy as np
import numpy.random as rand

import scipy as sp

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

def smoothstep(x):
    return (x * x * (3.0 - 2.0 * x))

def is_point_inside_triang(v01, v12, v20, r0, r1, r2):
    cross_v01_r0 = v01[:, :, 0] * r0[:, :, 1] - v01[:, :, 1] * r0[:, :, 0]
    cross_v12_r1 = v12[:, :, 0] * r1[:, :, 1] - v12[:, :, 1] * r1[:, :, 0]
    cross_v20_r2 = v20[:, :, 0] * r2[:, :, 1] - v20[:, :, 1] * r2[:, :, 0]

    sign_mults = np.stack((cross_v01_r0 * cross_v12_r1, cross_v12_r1 * cross_v20_r2), axis = 2)
    min_sign_mults = np.min(sign_mults, axis = 2)
    return (min_sign_mults >= 0.0)

def get_point_segm_dist(v, r):
    dot_v_r = v[:, :, 0] * r[:, :, 0] + v[:, :, 1] * r[:, :, 1]
    len_sq_v = v[:, :, 0] * v[:, :, 0] + v[:, :, 1] * v[:, :, 1]
    k_v = dot_v_r / len_sq_v
    k_v[k_v < 0.0] = 0.0
    k_v[k_v > 1.0] = 1.0
    dist_vec_v = r - v * np.tile(np.expand_dims(k_v, axis = 2), reps = (1, 1, 2))
    return np.hypot(dist_vec_v[:, :, 0], dist_vec_v[:, :, 1])

class TriangArtProcessor(QObject):
    def __init__(self):
        super().__init__()
        self._img_src = None
        self._img_res = None

        self._grad_n = None
        self._points = None

    @pyqtSlot(np.ndarray)
    def set_source_image(self, img_data):
        if (img_data is None):
            self._img_src = None
            self._img_h, self._img_w = None, None
        else:
            self._img_src = img_data.copy()
            img_sz = self._img_src.shape
            self._img_h, self._img_w = img_sz[0], img_sz[1]

    @pyqtSlot(float, float)
    def generate_with_params(self, triang_size, triang_size_range):
        if (self._img_src is None):
            self._img_res = np.array(np.NaN)
            self.generated_result.emit(self._img_res.copy())
            return

        self.report_progress.emit(0) # progress values are rough estimates
        self._set_params(triang_size, triang_size_range) # progress: 0 - 100, not subdivided
        self.report_progress.emit(100)
        self._preproc_img_src() # progress: 101 - 3500, not subdivided
        self.report_progress.emit(3500)
        self._create_points() # progress: 3501 - 3600, not subdivided
        self.report_progress.emit(3600)
        self._progress_before = 3601
        self._progress_after = 9000
        self._move_points() # progress: 3601 - 9000
        self.report_progress.emit(9000)
        self._progress_before = 9001
        self._progress_after = 10000
        self._triangulate_with_colors() # progress: 9001 - 10000
        self.report_progress.emit(10000)
        self.generated_result.emit(self._img_res.copy())

    def get_result(self):
        return self._img_res.copy()

    def _set_params(self, triang_size, triang_size_range):
        # both input params are 0 - 1
        img_h, img_w = self._img_h, self._img_w # image already set here
        self._n_points = int(round(200.0 * m.pow(10.0, (1.0 - triang_size)))) # 200 - 2000
        self._char_len = (img_w + img_h + m.sqrt(img_w * img_w + img_h * img_h + 
            (2.0 + 8.0 * self._n_points / m.sqrt(3.0)) * img_w * img_h)) / (2.0 * self._n_points)
        self._blur_sigma = 1.0 * self._char_len
        self._n_iters = 1 + self._n_points // 3
        self._min_grad_sobel_at_pts = 1.0 / self._blur_sigma
        range_coeff = m.pow(triang_size_range, 0.82)
        self._displ_weight_grad = 0.10 + 0.05 * 2.0 * (range_coeff - 0.5)
        self._displ_weight_triang = 0.55 - 0.15 * 2.0 * (range_coeff - 0.5)

    def _point_displ_damp_coeff(self, cur_iter):
        return (0.001 ** (cur_iter / (self._n_iters - 1.0)))

    def _preproc_img_src(self):
        grad_rgb = sp.ndimage.gaussian_gradient_magnitude(self._img_src, self._blur_sigma)
        grad_n = sum((grad_rgb[:, :, i] * grad_rgb[:, :, i]) for i in range(3))
        grad_n = np.sqrt(grad_n / 3.0)

        grad_n_max = np.amax(grad_n)
        if (grad_n_max > 0.0):
            grad_n /= grad_n_max
        self._grad_n = grad_n

    def _create_points(self):
        rand_gen = rand.default_rng(None)
        points_idx_1d = rand_gen.integers(low = 0, high = self._img_w * self._img_h, size = self._n_points)

        # initial coords are pixel centers assuming top left corner is at (-0.5, -0.5)
        # two other columns are "changing x/y allowed"
        points_y, points_x = np.divmod(points_idx_1d, self._img_w, dtype = np.float64)
        points_x += rand_gen.uniform(low = -0.5, high = 0.5, size = points_x.shape)
        points_y += rand_gen.uniform(low = -0.5, high = 0.5, size = points_y.shape)
        points_coord = np.vstack([points_x, points_y, np.ones(self._n_points), np.ones(self._n_points)]).T

        point_num_w = m.ceil(self._img_w / self._char_len)
        point_num_h = m.ceil(self._img_h / self._char_len)

        points_w_coord_x = np.linspace(0.0, self._img_w, point_num_w + 1) - 0.5
        points_w_coord_x = points_w_coord_x[1 : -1]
        points_w_coord_y = np.array([-0.5, self._img_h - 0.5])
        points_h_coord_y = np.linspace(0.0, self._img_h, point_num_h + 1) - 0.5
        points_h_coord_y = points_h_coord_y[1 : -1]
        points_h_coord_x = np.array([-0.5, self._img_w - 0.5])

        points_w_coord = np.empty((0, 4))
        for y_idx in range(points_w_coord_y.shape[0]):
            point_w_coord_y_and_flags_curr = np.array([points_w_coord_y[y_idx], 1.0, 0.0]).T
            point_w_coord_y_and_flags_curr_rep = np.tile(point_w_coord_y_and_flags_curr, (points_w_coord_x.shape[0], 1))
            points_w_coord_curr = np.hstack([np.atleast_2d(points_w_coord_x).T, point_w_coord_y_and_flags_curr_rep])
            points_w_coord = np.vstack([points_w_coord, points_w_coord_curr])

        points_h_coord = np.empty((0, 4))
        for x_idx in range(points_h_coord_x.shape[0]):
            point_h_coord_x_and_flags_curr = np.array([points_h_coord_x[x_idx], 0.0, 1.0]).T
            point_h_coord_x_and_flags_curr_rep = np.tile(point_h_coord_x_and_flags_curr, (points_h_coord_y.shape[0], 1))
            points_h_coord_curr = np.hstack(\
                [np.atleast_2d(point_h_coord_x_and_flags_curr_rep[:, 0]).T, 
                np.atleast_2d(points_h_coord_y).T, 
                point_h_coord_x_and_flags_curr_rep[:, 1 : 3]])
            points_h_coord = np.vstack([points_h_coord, points_h_coord_curr])

        points_corner_coord = np.array(\
            [
                [-0.5,              -0.5,              0.0, 0.0], 
                [self._img_w - 0.5, -0.5,              0.0, 0.0], 
                [-0.5,              self._img_h - 0.5, 0.0, 0.0], 
                [self._img_w - 0.5, self._img_h - 0.5, 0.0, 0.0]
            ])

        self._points = np.vstack([points_coord, points_w_coord, points_h_coord, points_corner_coord])

    def _move_points(self):
        grad_sobel_x = sp.ndimage.sobel(self._grad_n, axis = 1) / 4.0 # -1 to +1
        grad_sobel_y = sp.ndimage.sobel(self._grad_n, axis = 0) / 4.0 # -1 to +1

        for cur_iter in range(self._n_iters):
            delaunay_triang = sp.spatial.Delaunay(self._points[:, 0 : 2], qhull_options = "QJ") # keep all points in output

            triang_edges = set()
            triang_simplices = delaunay_triang.simplices
            for simpl_idx in range(triang_simplices.shape[0]):
                cur_simpl = triang_simplices[simpl_idx, :]
                for point_idx in range(-1, cur_simpl.shape[0] - 1):
                    point_from = cur_simpl[point_idx]
                    point_to = cur_simpl[point_idx + 1]
                    if (point_from < point_to):
                        triang_edges.add((point_from, point_to))
                    elif (point_from > point_to):
                        triang_edges.add((point_to, point_from))

            gsx_at_pts = sp.ndimage.map_coordinates(grad_sobel_x, self._points[:, 1 :: -1].T, mode = "reflect")
            gsy_at_pts = sp.ndimage.map_coordinates(grad_sobel_y, self._points[:, 1 :: -1].T, mode = "reflect")
            gsn_at_pts = np.hypot(gsx_at_pts, gsy_at_pts)
            gsa_at_pts = np.arctan2(gsy_at_pts, gsx_at_pts)

            gsn_at_pts[gsn_at_pts < self._min_grad_sobel_at_pts] = self._min_grad_sobel_at_pts
            dist_to_go_at_pts = (1.0 / gsn_at_pts)
            displ_data = self._displ_weight_grad * \
                np.atleast_2d(dist_to_go_at_pts).T * np.hstack([np.atleast_2d(np.cos(gsa_at_pts)).T, np.atleast_2d(np.sin(gsa_at_pts)).T])

            for edge in triang_edges:
                point_from, point_to = edge[0], edge[1]
                delta_coord = self._points[point_to, 0 : 2] - self._points[point_from, 0 : 2]
                dist = m.sqrt(delta_coord[0] ** 2 + delta_coord[1] ** 2)
                if (dist > 0.0):
                    norm_coord = delta_coord / dist
                else:
                    norm_coord = delta_coord.copy()

                triang_displ_magn = min(dist - self._char_len, 0.0)
                cur_displ = self._displ_weight_triang * triang_displ_magn * norm_coord
                displ_data[point_from, :] += cur_displ
                displ_data[point_to, :] -= cur_displ

            self._points[:, 0 : 2] += \
                self._point_displ_damp_coeff(cur_iter) * displ_data * self._points[:, 2 : 4]
            self._points[self._points[:, 0] < -0.5, 0] = -0.5
            self._points[self._points[:, 0] > self._img_w - 0.5, 0] = self._img_w - 0.5
            self._points[self._points[:, 1] < -0.5, 1] = -0.5
            self._points[self._points[:, 1] > self._img_h - 0.5, 1] = self._img_h - 0.5

            self.report_progress.emit(int(round(self._progress_before + \
                (self._progress_after - self._progress_before) * cur_iter / self._n_iters)))

    def _triangulate_with_colors(self):
        dest_color = np.zeros_like(self._img_src)
        dest_weight_sum = np.zeros(shape = (self._img_h, self._img_w))
        base_dist = m.sqrt(2.0) + 1.0e-10

        delaunay_triang = sp.spatial.Delaunay(self._points[:, 0 : 2]) # now "extra" points are removed
        triang_simplices = delaunay_triang.simplices
        n_simplices = triang_simplices.shape[0]
        for simpl_idx in range(n_simplices):
            cur_simpl = triang_simplices[simpl_idx, :]

            cur_verts = self._points[cur_simpl, 0 : 2]
            
            x_min = np.min(cur_verts[:, 0])
            x_max = np.max(cur_verts[:, 0])
            y_min = np.min(cur_verts[:, 1])
            y_max = np.max(cur_verts[:, 1])

            x_min_cell = min(max(int(m.ceil(x_min) - 1.0), 0), self._img_w - 1)
            x_max_cell = min(max(int(m.floor(x_max) + 1.0), 0), self._img_w - 1)
            y_min_cell = min(max(int(m.ceil(y_min) - 1.0), 0), self._img_h - 1)
            y_max_cell = min(max(int(m.floor(y_max) + 1.0), 0), self._img_h - 1)

            x_range = np.array(range(x_min_cell, x_max_cell + 1))
            y_range = np.array(range(y_min_cell, y_max_cell + 1))

            y_mesh, x_mesh = np.meshgrid(x_range, y_range)
            src_rect = self._img_src[x_mesh, y_mesh, :]

            point_coords = np.stack((y_mesh, x_mesh), axis = 2)

            vert0_coords = np.tile(np.expand_dims(cur_verts[0, :], axis = (0, 1)), reps = (*x_mesh.shape, 1))
            vert1_coords = np.tile(np.expand_dims(cur_verts[1, :], axis = (0, 1)), reps = (*x_mesh.shape, 1))
            vert2_coords = np.tile(np.expand_dims(cur_verts[2, :], axis = (0, 1)), reps = (*x_mesh.shape, 1))

            v01 = vert1_coords - vert0_coords
            v12 = vert2_coords - vert1_coords
            v20 = vert0_coords - vert2_coords

            r0 = point_coords - vert0_coords
            r1 = point_coords - vert1_coords
            r2 = point_coords - vert2_coords

            is_inside = is_point_inside_triang(v01, v12, v20, r0, r1, r2)

            dist_v01 = get_point_segm_dist(v01, r0)
            dist_v12 = get_point_segm_dist(v12, r1)
            dist_v20 = get_point_segm_dist(v20, r2)

            dists = np.stack((dist_v01, dist_v12, dist_v20), axis = 2)
            min_dist = np.min(dists, axis = 2)

            min_dist[is_inside] = 0.0
            min_dist /= base_dist
            min_dist[min_dist > 1.0] = 1.0
            cur_weights = 1.0 - smoothstep(min_dist)
            
            avg_color = np.sum(src_rect * np.expand_dims(cur_weights, axis = 2), axis = (0, 1)) / np.sum(cur_weights)

            dest_color[x_mesh, y_mesh, :] += np.expand_dims(avg_color, axis = (0, 1)) * \
                np.expand_dims(cur_weights, axis = 2)
            dest_weight_sum[x_mesh, y_mesh] += cur_weights

            self.report_progress.emit(int(round(self._progress_before + \
                (self._progress_after - self._progress_before) * simpl_idx / n_simplices)))

        dest_color /= np.expand_dims(dest_weight_sum, axis = 2)
        dest_color[~np.isfinite(dest_color)] = 0.0

        self._img_res = dest_color
    
    generated_result = pyqtSignal(np.ndarray)
    report_progress = pyqtSignal(int) # 0 - 10000
