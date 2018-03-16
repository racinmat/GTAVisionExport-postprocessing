import numpy as np


def pixel_to_ndc(pixel, size):
    p_y, p_x = pixel
    s_y, s_x = size
    return (((- 2 / s_y) * p_y + 1), (2 / s_x) * p_x - 1)


def ndc_to_pixel(ndc, size):
    ndc_y, ndc_x = ndc
    s_y, s_x = size
    return (-(s_y / 2) * ndc_y + (s_y / 2), (s_x / 2) * ndc_x + (s_x / 2))


def generate_points(width, height):
    x_range = range(0, width)
    y_range = range(0, height)
    points = np.transpose([np.tile(y_range, len(x_range)), np.repeat(x_range, len(y_range))])
    return points


def points_to_homo(points, res, depth, tresholding=True):
    width = res['width']
    height = res['height']
    size = (height, width)
    proj_matrix = res['proj_matrix']

    if tresholding:
        max_depth = res['cam_far_clip']
        # max_depth = 60 # just for testing
        vec = proj_matrix @ np.array([[1], [1], [-max_depth], [1]])
        # print(vec)
        vec /= vec[3]
        treshold = vec[2]
    else:
        treshold = - np.inf

    # vecs = np.zeros((4, points.shape[0]))
    vecs = np.zeros((4, len(np.where(depth[points[:, 0], points[:, 1]] > treshold)[
                                0])))  # this one is used when ommiting 0 depth (point behind the far clip)
    print("vecs.shape")
    print(vecs.shape)
    i = 0
    arr = points
    for y, x in arr:
        if depth[(y, x)] <= treshold:
            continue
        ndc = pixel_to_ndc((y, x), size)
        vec = [ndc[1], -ndc[0], depth[(y, x)], 1]
        vec = np.array(vec)
        vecs[:, i] = vec
        i += 1

    return vecs


def ndc_to_view(vecs, proj_matrix):
    vecs_p = np.linalg.inv(proj_matrix) @ vecs
    vecs_p /= vecs_p[3, :]
    return vecs_p


def view_to_world(vecs_p, view_matrix):
    vecs_p = np.linalg.inv(view_matrix) @ vecs_p
    vecs_p /= vecs_p[3, :]
    return vecs_p


def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6


def is_rigid(M):
    return is_rotation_matrix(M[0:3, 0:3]) and np.linalg.norm(M[3, :] - np.array([0, 0, 0, 1], dtype=M.dtype)) < 1e-6


def inv_rigid(M):
    # if we have rigid transformation matrix, we can calculate its inversion analytically, with bigger precision
    assert is_rigid(M)
    Mt = np.zeros_like(M)
    Mt[0:3, 0:3] = np.transpose(M[0:3, 0:3])
    Mt[0:3, 3] = - Mt[0:3, 0:3] @ M[0:3, 3]
    Mt[3, 3] = 1
    return Mt


def ndc_to_real(depth, proj_matrix):
    width = depth.shape[1]
    height = depth.shape[0]
    points = generate_points(width, height)
    params = {
        'width': width,
        'height': height,
        'proj_matrix': proj_matrix,
    }
    vecs = points_to_homo(points, params, depth, tresholding=False)
    vecs_p = ndc_to_view(vecs, proj_matrix)
    new_depth = np.copy(depth)
    for i, (y, x) in points:
        new_depth[y, x] = vecs_p[i, 2]
    return new_depth
