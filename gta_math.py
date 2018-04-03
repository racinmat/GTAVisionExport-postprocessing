import numpy as np
from math import tan, atan, radians, degrees, cos, sin
import time

THRESHOLD = 1000
MAXIMUM = np.iinfo(np.uint16).max


def pixel_to_ndc(pixel, size):
    p_y, p_x = pixel
    s_y, s_x = size
    return (((- 2 / s_y) * p_y + 1), (2 / s_x) * p_x - 1)


def pixels_to_ndcs(pixels, size):
    # vectorized version, of above function
    pixels = np.copy(pixels).astype(np.float32)
    if pixels.shape[1] == 2 and pixels.shape[0] != 2:
        pixels = pixels.T
    # pixels are in shape <pixels, 2>
    p_y = pixels[0, :]
    p_x = pixels[1, :]
    s_y, s_x = size
    pixels[0, :] = (-2 / s_y) * p_y + 1
    pixels[1, :] = (2 / s_x) * p_x - 1
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


def points_to_homo(res, depth, tresholding=True):
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
    valid_points = np.where(depth > threshold)
    valid_y, valid_x = valid_points

    vecs = np.zeros((4, len(valid_y)))

    ndcs = pixels_to_ndcs(np.array(valid_points), size)

    vecs[0, :] = ndcs[1, :]
    vecs[1, :] = ndcs[0, :]
    vecs[2, :] = depth[valid_y, valid_x]
    vecs[3, :] = 1  # last, homogenous coordinate
    return vecs, valid_points


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

    params = {
        'width': width,
        'height': height,
        'proj_matrix': proj_matrix,
    }

    vecs, transformed_points = points_to_homo(params, depth, tresholding=False)
    vec_y, vec_x = transformed_points

    vecs_p = ndc_to_view(vecs, proj_matrix).T

    new_depth = np.copy(depth)
    new_depth[vec_y, vec_x] = vecs_p[:, 2]

    return new_depth


def construct_proj_matrix(H=1080, W=1914, fov=50.0, near_clip=1.5):
    # for z coord
    f = near_clip  # the near clip, but f in the book
    n = 10003.815  # the far clip, rounded value of median, after very weird values were discarded
    # x coord
    # r = W * n * tan(radians(fov) / 2) / H
    # l = -r
    # y coord
    # t = n * tan(radians(fov) / 2)
    # b = -t
    # x00 = 2*n/(r-l)
    x00 = H / (tan(radians(fov) / 2) * W)
    # x11 = 2*n/(t-b)
    x11 = 1 / tan(radians(fov) / 2)
    # x02 = -(r + l) / (r - l)
    # x12 = -(t + b) / (t - b)
    x02 = 0  # since r == -l, the numerator is 0 and thus whole value is 0
    x12 = 0  # since b == -t, the numerator is 0 and thus whole value is 0
    return np.array([
        [x00, 0, x02, 0],
        [0, x11, x12, 0],
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


def construct_view_matrix(camera_pos, camera_rotation):
    view_matrix = np.zeros((4, 4))
    # view_matrix[0:3, 3] = camera_pos
    view_matrix[0:3, 0:3] = create_rot_matrix(camera_rotation)
    view_matrix[3, 3] = 1

    trans_matrix = np.eye(4)
    trans_matrix[0:3, 3] = -camera_pos

    # return view_matrix
    return view_matrix @ trans_matrix


def create_rot_matrix(euler):
    x = np.radians(euler[0])
    y = np.radians(euler[1])
    z = np.radians(euler[2])

    Rx = np.array([
        [1, 0, 0],
        [0, sin(x), cos(x)],
        [0, cos(x), -sin(x)]
    ], dtype=np.float)
    Ry = np.array([
        [cos(y), 0, -sin(y)],
        [0, 1, 0],
        [sin(y), 0, cos(y)]
    ], dtype=np.float)
    Rz = np.array([
        [cos(z), sin(z), 0],
        [sin(z), -cos(z), 0],
        [0, 0, 1]
    ], dtype=np.float)
    result = Rx @ Ry @ Rz
    return result


def homo_world_coords_to_pixel(point_homo, view_matrix, proj_matrix, width, height):
    viewed = view_matrix @ point_homo
    projected = proj_matrix @ viewed
    projected /= projected[3]
    to_pixel_matrix = np.array([
        [width/2, 0, 0, width/2],
        [0, -height/2, 0, height/2],
    ])
    in_pixels = to_pixel_matrix @ projected
    return in_pixels


def world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height):
    point_homo = np.array([pos[0], pos[1], pos[2], 1])
    return homo_world_coords_to_pixel(point_homo, view_matrix, proj_matrix, width, height)


def model_coords_to_pixel(model_pos, model_rot, pos, view_matrix, proj_matrix, width, height):
    point_homo = np.array([pos[0], pos[1], pos[2], 1])
    model_matrix = construct_model_matrix(model_pos, model_rot)
    # print('model_matrix\n', model_matrix)
    world_point_homo = model_matrix @ point_homo
    return homo_world_coords_to_pixel(world_point_homo, view_matrix, proj_matrix, width, height)


def create_model_rot_matrix(euler):
    x = np.radians(euler[0])
    y = np.radians(euler[1])
    z = np.radians(euler[2])

    Rx = np.array([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)]
    ], dtype=np.float)
    Ry = np.array([
        [cos(y), 0, sin(y)],
        [0, 1, 0],
        [-sin(y), 0, cos(y)]
    ], dtype=np.float)
    Rz = np.array([
        [cos(z), -sin(z), 0],
        [sin(z), cos(z), 0],
        [0, 0, 1]
    ], dtype=np.float)
    result = Rx @ Ry @ Rz
    return result


def construct_model_matrix(position, rotation):
    view_matrix = np.zeros((4, 4))
    # view_matrix[0:3, 3] = camera_pos
    view_matrix[0:3, 0:3] = create_model_rot_matrix(rotation)
    view_matrix[0:3, 3] = position
    view_matrix[3, 3] = 1

    return view_matrix
