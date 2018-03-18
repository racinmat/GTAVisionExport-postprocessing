import numpy as np
from math import tan, atan, radians, degrees
import time

THRESHOLD = 1000
MAXIMUM = np.iinfo(np.uint16).max


def pixel_to_ndc(pixel, size):
    p_y, p_x = pixel
    s_y, s_x = size
    return (((- 2 / s_y) * p_y + 1), (2 / s_x) * p_x - 1)


def pixels_to_ndcs(pixels, size):
    # vectorized version, of above function
    # pixels are in shape <pixels, 2>
    pixels = np.copy(pixels)
    p_y = pixels[:, 0]
    p_x = pixels[:, 1]
    s_y, s_x = size
    pixels[:, 0] = (-2 / s_y) * p_y + 1
    pixels[:, 1] = (2 / s_x) * p_x - 1
    return pixels


def ndc_to_pixel(ndc, size):
    ndc_y, ndc_x = ndc
    s_y, s_x = size
    return (-(s_y / 2) * ndc_y + (s_y / 2), (s_x / 2) * ndc_x + (s_x / 2))


def generate_points(width, height):
    x_range = range(0, width)
    y_range = range(0, height)
    points = np.transpose([np.tile(y_range, len(x_range)), np.repeat(x_range, len(y_range))])
    return points

# time to generate_points:  0.017998456954956055
# time to prepare treshold:  0.043000221252441406
# time to transfer all pixels to homo:  15.504000663757324
# time to points_to_homo:  15.547000885009766
# time to ndc_to_view:  0.06600069999694824
# time to extract to new depth:  1.988011121749878
# time to convert coords:  17.630000114440918
# depth calculated

# time to generate_points:  0.01800084114074707
# time to prepare treshold:  0.044997215270996094
# time to transfer all points to ndcs:  0.03500080108642578
# time to transfer all pixels to homo:  7.98600959777832
# time to points_to_homo:  8.066999673843384
# time to ndc_to_view:  0.06299757957458496
# time to extract to new depth:  2.01800274848938
# time to convert coords:  10.177000761032104
# depth calculated

def points_to_homo(points, res, depth, tresholding=True):
    start = time.time()

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
        threshold = vec[2]
    else:
        threshold = - np.inf

    # vecs = np.zeros((4, points.shape[0]))
    vecs = np.zeros((4, len(np.where(depth[points[:, 0], points[:, 1]] > threshold)[
                                0])))  # this one is used when ommiting 0 depth (point behind the far clip)

    valid_points = np.array(np.where(depth > threshold))

    end = time.time()
    print('time to prepare treshold: ', end - start)
    start = time.time()

    ndcs = pixels_to_ndcs(valid_points, size)

    end = time.time()
    print('time to transfer all points to ndcs: ', end - start)
    start = time.time()

    i = 0
    vecs[3, :] = 1  # last, homogenous coordinate
    arr = points
    vecs[0:2, :] = valid_points
    vecs[2, :] = depth[valid_points]
    for j, (y, x) in enumerate(arr):
        if depth[(y, x)] <= threshold:
            continue
        # vec = [ndcs[j, 1], ndcs[j, 0], depth[(y, x)], 1]
        # vec = np.array(vec)
        vecs[:, i] = np.array([ndcs[j, 1], ndcs[j, 0], depth[(y, x)], 1])
        i += 1

    end = time.time()
    print('time to transfer all pixels to homo: ', end - start)

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

    start = time.time()
    points = generate_points(width, height)
    end = time.time()
    print('time to generate_points: ', end - start)

    params = {
        'width': width,
        'height': height,
        'proj_matrix': proj_matrix,
    }

    start = time.time()
    vecs = points_to_homo(points, params, depth, tresholding=False)
    end = time.time()
    print('time to points_to_homo: ', end - start)

    start = time.time()
    vecs_p = ndc_to_view(vecs, proj_matrix).T
    end = time.time()
    print('time to ndc_to_view: ', end - start)

    start = time.time()
    new_depth = np.copy(depth)
    for i, (y, x) in enumerate(points):
        new_depth[y, x] = vecs_p[i, 2]
    end = time.time()
    print('time to extract to new depth: ', end - start)

    return new_depth


def construct_proj_matrix(H=1080, W=1914, fov=50.0, near_clip=1.5):
    # for z coord
    f = near_clip  # the near clip, but f in the book
    n = 10003.815  # the far clip, rounded value of median, after very weird values were discarded
    # x coord
    r = W * n * tan(radians(fov) / 2) / H
    l = -r
    # y coord
    t = n * tan(radians(fov) / 2)
    b = -t
    # x00 = 2*n/(r-l)
    x00 = H / (tan(radians(fov) / 2) * W)
    # x11 = 2*n/(t-b)
    x11 = 1 / tan(radians(fov) / 2)
    return np.array([
        [x00, 0, -(r + l) / (r - l), 0],
        [0, x11, -(t + b) / (t - b), 0],
        [0, 0, -f / (f - n), -f * n / (f - n)],
        [0, 0, -1, 0],
    ])


def depth_crop_and_positive(depth):
    depth = np.copy(depth)
    # first we reverse values, so they are in positive values
    depth *= -1
    # then we treshold the far clip so when we scale to integer range
    depth[depth > THRESHOLD] = THRESHOLD
    return depth


def depth_to_integer_range(depth):
    depth = np.copy(depth)
    # then we rescale to as big value as file format allows us
    ratio = MAXIMUM / THRESHOLD
    depth *= ratio
    return depth.astype(np.int32)


def depth_from_integer_range(depth):
    depth = np.copy(depth)
    depth = depth.astype(np.float32)
    # then we rescale to integer32
    ratio = THRESHOLD / MAXIMUM
    depth *= ratio
    return depth
