import numpy as np
from math import tan, atan, radians, degrees, cos, sin
import time
from sympy import Line, Point


# threshold for persisting images,
THRESHOLD = 1000
MAXIMUM = np.iinfo(np.uint16).max
# depth will be assinged threshold value for values behind it, effectively projecting it nearer
PROJECTING = False


def pixel_to_ndc(pixel, size):
    p_y, p_x = pixel
    s_y, s_x = size
    s_y -= 1    # so 1 is being mapped into (n-1)th pixel
    s_x -= 1    # so 1 is being mapped into (n-1)th pixel
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
    s_y -= 1    # so 1 is being mapped into (n-1)th pixel
    s_x -= 1    # so 1 is being mapped into (n-1)th pixel
    pixels[0, :] = (-2 / s_y) * p_y + 1
    pixels[1, :] = (2 / s_x) * p_x - 1
    return pixels


def ndc_to_pixel(ndc, size):
    ndc_y, ndc_x = ndc
    s_y, s_x = size
    s_y -= 1    # so 1 is being mapped into (n-1)th pixel
    s_x -= 1    # so 1 is being mapped into (n-1)th pixel
    return (-(s_y / 2) * ndc_y + (s_y / 2), (s_x / 2) * ndc_x + (s_x / 2))


def ndcs_to_pixels(ndcs, size):
    # vectorized version, of above function
    pixels = np.copy(ndcs).astype(np.float32)
    if ndcs.shape[1] == 2 and ndcs.shape[0] != 2:
        ndcs = ndcs.T
    # pixels are in shape <pixels, 2>
    ndc_x = ndcs[0, :]
    ndc_y = ndcs[1, :]
    s_y, s_x = size
    s_y -= 1    # so 1 is being mapped into (n-1)th pixel
    s_x -= 1    # so 1 is being mapped into (n-1)th pixel
    pixels[0, :] = (-s_y / 2) * ndc_y + (s_y / 2)
    pixels[1, :] = (s_x / 2) * ndc_x + (s_x / 2)
    return pixels.astype(np.int32)


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

    if PROJECTING:
        # print('projecting')
        depth[depth < threshold] = threshold    # since 0 is far clip, depth below threshold is behind threshold, and this projects it
    # print('threshold', threshold)
    # vecs = np.zeros((4, points.shape[0]))
    valid_points = np.where(depth >= threshold)
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
    vecs_p = inv_rigid(view_matrix) @ vecs_p
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


def create_rot_matrix(rot):
    x = np.radians(rot[0])
    y = np.radians(rot[1])
    z = np.radians(rot[2])

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
    # following row works for partially visible cars, not for cars completely outside of the frustum
    projected[0:3, projected[3] < 0] *= -1  # this row is very important. It fixed invalid projection of points outside the camera view frustum
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


def model_coords_to_pixel(model_pos, model_rot, positions, view_matrix, proj_matrix, width, height):
    point_homo = np.array([positions[:, 0], positions[:, 1], positions[:, 2], np.ones_like(positions[:, 0])])
    model_matrix = construct_model_matrix(model_pos, model_rot)
    world_point_homo = model_matrix @ point_homo
    #print('world_point_homo.shape', world_point_homo.shape)
    return homo_world_coords_to_pixel(world_point_homo, view_matrix, proj_matrix, width, height)


def create_model_rot_matrix(rot):
    x = np.radians(rot[0])
    y = np.radians(rot[1])
    z = np.radians(rot[2])

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


def get_model_3dbbox(model_sizes):
    x_min, x_max, y_min, y_max, z_min, z_max = model_sizes
    # preparing points of cuboid
    points_3dbbox = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
    ])
    return points_3dbbox


def model_coords_to_world(model_pos, model_rot, positions, view_matrix, proj_matrix, width, height):
    point_homo = np.array([positions[:, 0], positions[:, 1], positions[:, 2], np.ones_like(positions[:, 0])])
    model_matrix = construct_model_matrix(model_pos, model_rot)
    world_point_homo = model_matrix @ point_homo

    world_point_homo /= world_point_homo[3, :]
    return world_point_homo.T[:, 0:3]


def model_coords_to_ndc(model_pos, model_rot, positions, view_matrix, proj_matrix, width, height):
    point_homo = np.array([positions[:, 0], positions[:, 1], positions[:, 2], np.ones_like(positions[:, 0])])
    model_matrix = construct_model_matrix(model_pos, model_rot)
    point_homo = model_matrix @ point_homo
    viewed = view_matrix @ point_homo
    projected = proj_matrix @ viewed
    projected /= projected[3, :]
    return projected.T[:, 0:3]


def is_entity_in_image(depth, stencil, row, view_matrix, proj_matrix, width, height):
    pos = np.array(row['pos'])
    rot = np.array(row['rot'])
    # todo: add entity visibility checking based on stencil data
    model_sizes = np.array(row['model_sizes'])
    points_3dbbox = get_model_3dbbox(model_sizes)

    # if row['bbox'][0] != [np.inf, np.inf]:
    #     return True

    # calculating model_coords_to_ndc, so we have both anc and viewed points
    point_homo = np.array([points_3dbbox[:, 0], points_3dbbox[:, 1], points_3dbbox[:, 2], np.ones_like(points_3dbbox[:, 0])])
    model_matrix = construct_model_matrix(pos, rot)
    point_homo = model_matrix @ point_homo
    viewed = view_matrix @ point_homo
    projected = proj_matrix @ viewed
    projected /= projected[3, :]
    bbox_3d = projected.T[:, 0:3]

    bbox_2d = bbox_3d[:, 0:2]
    # test if points are inside NDC cubloid by X and Y
    in_ndc = ((bbox_2d[:, 0] > -1) & (bbox_2d[:, 0] < 1) & (bbox_2d[:, 1] > -1) & (bbox_2d[:, 1] < 1)).any()
    # test if points are behind near clip (if they are, they should be in image)
    behind_near_clip = (viewed[2] < 0).any()    # assuming near clip is negative here, which means classical near clip

    # the new strategy is only to check cars with bbox infinities.
    # this car is either not seen or it is partly in the image.
    # Being partly in the image can be checked by some point being behind ans some poins
    # in front of near cam plane
    # return (viewed[2] < 0).any() and (viewed[2] > 0).any()

    # do this test only for entities which can not be excluded otherwise
    if not in_ndc or not behind_near_clip:
        return False

    # test of obstacles, if 3d coord of point where middle of entity should be, is in correct depth
    pixel_pos = world_coords_to_pixel(pos, view_matrix, proj_matrix, width, height)
    pix_x, pix_y = pixel_pos.astype(int)
    ndc_y, ndc_x = pixel_to_ndc((pix_y, pix_x), (height, width))
    if ndc_x < -1 or ndc_x > 1 or ndc_y < -1 or ndc_y > 1:
        # position is not in the image (e.g. partially visible objects)
        # so I can not evaluate it and thus I say it is ok, since this test is only for exclding some cars
        return True

    ndc_homo = np.array([ndc_x, ndc_y, depth[pix_y, pix_x], 1])[:, np.newaxis]
    view_homo = ndc_to_view(ndc_homo, proj_matrix)
    world_homo = view_to_world(view_homo, view_matrix)
    world_homo /= world_homo[3]
    world_pos = world_homo[0:3].T
    # now I have original entity position, and world position of pixel corresponding to its location on image.
    # Now, if they are distant more than diameter of model size, it is not in the image
    eps = np.linalg.norm(model_sizes.reshape(3, 2))
    dist = np.linalg.norm(pos - world_pos)
    return dist <= eps*2


# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
# def perp( a ) :
#     b = np.empty_like(a)
#     b[0] = -a[1]
#     b[1] = a[0]
#     return b
# def seg_intersect(p1, p2) :
#     a1, a2 = np.array(p1)
#     b1, b2 = np.array(p2)
#     da = a2-a1
#     db = b2-b1
#     dp = a1-b1
#     dap = perp(da)
#     denom = np.dot( dap, db)
#     num = np.dot( dap, dp )
#     return (num / denom.astype(float))*db + b1

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
# Return true if line segments AB and CD intersect
def are_intersecting(l1, l2):
    A, B = l1
    C, D = l2
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def calculate_2d_bbox(row, view_matrix, proj_matrix, width, height):
    pos = np.array(row['pos'])
    rot = np.array(row['rot'])
    model_sizes = np.array(row['model_sizes'])
    points_3dbbox = get_model_3dbbox(model_sizes)
    # calculating model_coords_to_ndc, so we have both anc and viewed points
    point_homo = np.array([points_3dbbox[:, 0], points_3dbbox[:, 1], points_3dbbox[:, 2], np.ones_like(points_3dbbox[:, 0])])
    model_matrix = construct_model_matrix(pos, rot)
    point_homo = model_matrix @ point_homo
    viewed = view_matrix @ point_homo
    projected = proj_matrix @ viewed
    projected /= projected[3, :]
    bbox_3d = projected.T[:, 0:3]

    bbox_2d_points = bbox_3d[:, 0:2]
    is_3d_bbox_partially_outside = (bbox_2d_points < -1).any() or (bbox_2d_points > 1).any()
    bbox_2d_points = bbox_2d_points[((bbox_2d_points <= 1) & (bbox_2d_points >= -1)).all(axis=1)]
    if is_3d_bbox_partially_outside:
        bbox_2d = model_coords_to_pixel(pos, rot, points_3dbbox, view_matrix, proj_matrix, width, height).T
        # now we need to compute intersections between end of image and points
        # now we build 12 lines for 3d bounding box
        lines = list()
        lines.append([bbox_2d[0, :], bbox_2d[1, :]])
        lines.append([bbox_2d[1, :], bbox_2d[3, :]])
        lines.append([bbox_2d[3, :], bbox_2d[2, :]])
        lines.append([bbox_2d[2, :], bbox_2d[0, :]])

        lines.append([bbox_2d[4, :], bbox_2d[5, :]])
        lines.append([bbox_2d[5, :], bbox_2d[7, :]])
        lines.append([bbox_2d[7, :], bbox_2d[6, :]])
        lines.append([bbox_2d[6, :], bbox_2d[4, :]])

        lines.append([bbox_2d[4, :], bbox_2d[0, :]])
        lines.append([bbox_2d[5, :], bbox_2d[1, :]])
        lines.append([bbox_2d[6, :], bbox_2d[2, :]])
        lines.append([bbox_2d[7, :], bbox_2d[3, :]])

        borders = list()
        borders.append([[0, 0], [0, height]])
        borders.append([[0, height], [width, height]])
        borders.append([[0, 0], [width, 0]])
        borders.append([[width, 0], [width, height]])

        for line in lines:
            for border in borders:
                if are_intersecting(line, border):
                    l1 = Line(Point(line[0][0], line[0][1]), Point(line[1][0], line[1][1]))
                    l2 = Line(Point(border[0][0], border[0][1]), Point(border[1][0], border[1][1]))
                    x, y = np.array(next(iter(l1.intersect(l2))), dtype=np.float32)
                    ndc_y, ndc_x = pixel_to_ndc((y, x), (height, width))
                    bbox_2d_points = np.vstack((bbox_2d_points, [ndc_x, ndc_y]))

    bbox_2d = np.array([
        [bbox_2d_points[:, 0].max(), -bbox_2d_points[:, 1].min()],
        [bbox_2d_points[:, 0].min(), -bbox_2d_points[:, 1].max()],
    ])
    # rescale from [-1, 1] to [0, 1]
    bbox_2d = (bbox_2d / 2) + 0.5
    return bbox_2d.tolist()


def get_depth_lut_for_linear_view(proj_matrix, z_meters_min, z_meters_max, z_range):
    X_view, Y_view, Z_view, W_view = np.meshgrid(np.linspace(1, 2, 1), np.linspace(1, 2, 1),
                                                 np.linspace(-z_meters_max, -z_meters_min, z_range),
                                                 np.linspace(1, 2, 1))
    view_positions = np.vstack([X_view.ravel(), Y_view.ravel(), Z_view.ravel(), W_view.ravel()])
    ndc_positions = proj_matrix @ view_positions
    ndc_positions /= ndc_positions[3, :]
    # ndc_z = np.flip(ndc_positions[2, :], axis=0)
    ndc_z = ndc_positions[2, :]

    return ndc_z


def grid_to_ndc_pcl_linear_view(bool_grid, proj_matrix, z_meters_min, z_meters_max):
    x_range, y_range, z_range = bool_grid.shape
    ndc_z = get_depth_lut_for_linear_view(proj_matrix, z_meters_min, z_meters_max, z_range)
    return grid_to_ndc_pcl_with_lut(bool_grid, ndc_z)


def grid_to_ndc_pcl_with_lut(bool_grid, ndc_z):
    # here bin centers are in lut
    x_range, y_range, z_range = bool_grid.shape

    points = np.argwhere(bool_grid == True)
    points = points.astype(dtype=np.float32)
    # x mapping to ndc
    transformed = pixels_to_ndcs(np.vstack((points[:, 1], points[:, 0])), (y_range, x_range))
    points[:, 0] = transformed[1, :]
    points[:, 1] = transformed[0, :]
    # we can map x and y linearly, but we need to z map hyperbolically, so we use LUT
    points[:, 2] = ndc_z[points[:, 2].astype(np.int32)]
    return points
