import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image

import visualization
from gta_math import calculate_2d_bbox, construct_model_matrix, get_model_3dbbox, is_entity_in_image, \
    model_coords_to_pixel, construct_view_matrix, create_rot_matrix, rot_matrix_to_euler_angles, \
    create_model_rot_matrix, model_rot_matrix_to_euler_angles, relative_and_absolute_camera_to_car_rotation_angles, \
    relative_and_absolute_camera_to_car_position, model_coords_to_world, calculate_2d_bbox_pixels, rectangles_overlap, \
    get_rectangles_overlap, get_rectangle_volume, ndc_to_real, points_to_homo, ndc_to_view


def vehicle_type_gta_to_toyota(gta_type):
    """
    gta types:
        Compacts
        Sedans
        SUVs
        Coupes
        Muscle
        SportsClassics
        Sports
        Super
        Motorcycles
        OffRoad
        Industrial
        Utility
        Vans
        Cycles
        Boats
        Helicopters
        Planes
        Service
        Emergency
        Military
        Commercial
        Trains
    toyota types:
        VEHICLE_PASSENGER_CAR = 0,   // cca 4.5m length
        VEHICLE_CITY_CAR = 1,                // cca 3.5m length
        VEHICLE_SUV = 2,
        VEHICLE_PICKUP = 3,
        VEHICLE_VAN = 4,
        VEHICLE_TRUCK = 5,
        VEHICLE_BUS = 6,
        VEHICLE_TRACTOR = 7,
        VEHICLE_TRAILER = 8,
        //VEHICLE_TRACTOR_WITH_TRAILER = 9,
        VEHICLE_MILITARY = 10,
        VEHICLE_AGRICULTURE = 11,
        VEHICLE_OTHER = 12
    """
    # todo: výhledově přidat sběr modelů
    mapping = {
        'Compacts': 'VEHICLE_CITY_CAR',
        'Sedans': 'VEHICLE_PASSENGER_CAR',
        'SUVs': 'SUV',
        'Coupes': 'VEHICLE_PASSENGER_CAR',
        'Muscle': 'VEHICLE_PASSENGER_CAR',
        'SportsClassics': 'VEHICLE_PASSENGER_CAR',
        'Sports': 'VEHICLE_PASSENGER_CAR',
        'Super': 'VEHICLE_PASSENGER_CAR',
        'Motorcycles': 'OTHER',
        'OffRoad': 'SUV',
        'Industrial': 'VEHICLE_TRUCK',
        'Utility': 'OTHER',
        'Vans': 'VAN',
        'Cycles': 'OTHER',
        'Boats': 'OTHER',
        'Helicopters': 'OTHER',
        'Planes': 'OTHER',
        'Service': 'OTHER',
        'Emergency': 'OTHER',
        'Military': 'MILITARY',
        'Commercial': 'OTHER',
        'Trains': 'OTHER',
        'Unknown': 'OTHER',
    }

    name_to_number = {
        'PASSENGER_CAR': 0,  # cca 4.5m length
        'CITY_CAR': 1,  # cca 3.5m length
        'SUV': 2,
        'PICKUP': 3,
        'VAN': 4,
        'TRUCK': 5,
        'BUS': 6,
        'TRACTOR': 7,
        'TRAILER': 8,
        'MILITARY': 10,
        'AGRICULTURE': 11,
        'OTHER': 12,
    }
    return name_to_number[mapping[gta_type]]


def get_3d_bbox_projected_to_pixels(entity, view_matrix, proj_matrix, width, height):
    model_sizes = np.array(entity['model_sizes'])
    points_3dbbox = get_model_3dbbox(model_sizes)
    point_homo = np.array(
        [points_3dbbox[:, 0], points_3dbbox[:, 1], points_3dbbox[:, 2], np.ones_like(points_3dbbox[:, 0])])
    bbox_3d = model_coords_to_pixel(entity['pos'], entity['rot'], point_homo.T, view_matrix, proj_matrix, width, height)
    return bbox_3d


def get_3d_bbox_projected_to_world(entity):
    model_sizes = np.array(entity['model_sizes'])
    points_3dbbox = get_model_3dbbox(model_sizes)
    point_homo = np.array(
        [points_3dbbox[:, 0], points_3dbbox[:, 1], points_3dbbox[:, 2], np.ones_like(points_3dbbox[:, 0])])
    bbox_3d = model_coords_to_world(entity['pos'], entity['rot'], point_homo.T)
    return bbox_3d


def is_entity_closer(closer, more_distant, my_car_position, my_car_rotation):
    # tuple because of lru_cache hashability
    closer_location = location_to_toyota_ego(closer['model_sizes'], closer['pos'], closer['rot'], my_car_position,
                                             my_car_rotation)
    closer_distance = np.linalg.norm(closer_location)
    more_distant_location = location_to_toyota_ego(more_distant['model_sizes'], more_distant['pos'],
                                                   more_distant['rot'], my_car_position, my_car_rotation)
    more_distant_distance = np.linalg.norm(more_distant_location)
    return closer_distance < more_distant_distance


def occlusion_2d_bbox_ratio(entity, entities, view_matrix, proj_matrix, width, height, my_car_position,
                            my_car_rotation):
    bbox_2d = calculate_2d_bbox_pixels(entity, view_matrix, proj_matrix, width, height)

    overlapping_bboxes = []
    for other in entities:
        if other['handle'] == entity['handle']:
            continue
        other_bbox_2d = calculate_2d_bbox_pixels(other, view_matrix, proj_matrix, width, height)
        if is_entity_closer(other, entity, my_car_position, my_car_rotation) and rectangles_overlap(bbox_2d,
                                                                                                    other_bbox_2d):
            overlapping_bboxes.append(other_bbox_2d)

    overlap_volume = 0
    for other_bbox in overlapping_bboxes:
        overlap = get_rectangles_overlap(bbox_2d, other_bbox)
        overlap_volume += get_rectangle_volume(overlap)

    # to correctly calculate area overlapped by any other bbox, I need to calculate overlapping between all pairs of bboxes
    # when multiple bboxes overlap original one, I need to add volume subtracted multiple times to calculate the volume correctly
    # so firstly, I subtract every intersection of bbox with others, and then I add intersection of every pair of bboxes and orig. bbox
    for i, bbox_1 in enumerate(overlapping_bboxes[:-1]):
        for bbox_2 in overlapping_bboxes[i + 1:]:
            overlap = get_rectangles_overlap(bbox_1, bbox_2)
            overlap = get_rectangles_overlap(overlap, bbox_2d)
            overlap_volume -= get_rectangle_volume(overlap)

    bbox_volume = get_rectangle_volume(bbox_2d)
    return overlap_volume / bbox_volume


def out_of_image_2dbbox_ratio(entity, view_matrix, proj_matrix, width, height):
    """
    calculates out of image ratio, 0 for vehicles fully in image, 1 for vehicles fully out of image
    id calculated by simply cropping out of image part of 2dbbox nad diving volumes,
    does not use intersections of 3d bbox with image to ceate tighter 2d bbox
    """
    model_sizes = np.array(entity['model_sizes'])
    points_3dbbox = get_model_3dbbox(model_sizes)
    point_homo = np.array(
        [points_3dbbox[:, 0], points_3dbbox[:, 1], points_3dbbox[:, 2], np.ones_like(points_3dbbox[:, 0])])
    bbox_3d = model_coords_to_pixel(entity['pos'], entity['rot'], point_homo.T, view_matrix, proj_matrix, width, height)
    bbox_2d = np.array([
        [bbox_3d[:, 0].max(), -bbox_3d[:, 1].min()],
        [bbox_3d[:, 0].min(), -bbox_3d[:, 1].max()],
    ])
    image = np.array([
        [width, height],
        [0, 0],
    ])

    if not rectangles_overlap(bbox_2d, image):
        return 0

    in_image = get_rectangles_overlap(bbox_2d, image)
    in_image_volume = get_rectangle_volume(in_image)
    whole_volume = get_rectangle_volume(bbox_2d)
    return in_image_volume / whole_volume


def get_my_car_position_and_rotation(cam_pos, cam_rot, cam_rel_pos, cam_rel_rot):
    car_pos = relative_and_absolute_camera_to_car_position(cam_pos, cam_rot, cam_rel_pos, cam_rel_rot)
    car_rot = relative_and_absolute_camera_to_car_rotation_angles(cam_rot, cam_rel_rot)
    return car_pos, car_rot


def get_my_car_model_sizes():
    model_sizes = [-0.915331721305, 0.915331602096, -2.308772802352, 2.149841547012, -0.573544025421, 0.989072620868]
    return model_sizes


def get_gta_center_to_toyota_center(model_sizes):
    x_min, x_max, y_min, y_max, z_min, z_max = model_sizes
    gta_center_to_corner = np.array([x_min, y_min, z_min])
    corner_to_toyota_center = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2, 0])
    gta_center_to_toyota_center = gta_center_to_corner + corner_to_toyota_center
    return gta_center_to_toyota_center


def entity_gta_location_to_toyota_location_world(model_sizes, pos, rot):
    """
    Gta center is somewhere in car, toyota center is in middle of car on ground. This transfer world gta car position
    into the world toyota car position
    """
    gta_center_to_toyota_center = get_gta_center_to_toyota_center(model_sizes)
    rot = create_model_rot_matrix(rot)
    world_location = pos + rot @ gta_center_to_toyota_center
    return world_location


def location_to_toyota_ego(model_sizes, pos, rot, my_car_position, my_car_rotation):
    # everything needs to be tuple for hashability, this casts to tuple
    return location_to_toyota_ego_cached(tuple(model_sizes), tuple(pos), tuple(rot), tuple(my_car_position),
                                         tuple(my_car_rotation))


@lru_cache(maxsize=32)
def location_to_toyota_ego_cached(model_sizes, pos, rot, my_car_position, my_car_rotation):
    """
    Returns location of vehicle relatively to the ego car center.
    Car center is calculated from camera position and camera relative position and rotation, and thus is same
    for all cameras, rotation is taken as ego vehicle rotation, which is calculated from camera rotation and camera relative rotation

    is used for actual location calculation and also for overlapping calculation. 32 cache size should be enough for all visible cars in one image

    because of the lru_cache, I don't pass whole entity dictionary, but only its individual parameters
    """
    model_sizes = list(model_sizes)
    pos = list(pos)
    rot = list(rot)
    my_car_position = np.array(my_car_position)
    my_car_rotation = np.array(my_car_rotation)
    world_location = entity_gta_location_to_toyota_location_world(model_sizes, pos, rot)

    world_to_car_m = construct_view_matrix(my_car_position, my_car_rotation)
    world_location = np.concatenate((world_location, [1]))
    view_location = (world_to_car_m @ world_location)
    view_location /= view_location[3]
    return view_location[0:3]


def vehicle_rotation_relative_to_my_car(entity_rotation, my_car_rotation):
    entity_m = create_model_rot_matrix(entity_rotation)
    my_car_m = create_model_rot_matrix(my_car_rotation)
    relative_m = entity_m @ my_car_m.T
    relative_rot = model_rot_matrix_to_euler_angles(relative_m)
    return relative_rot


def vehicle_rotation_relative_to_camera(entity_rotation, camera_rotation):
    entity_m = create_model_rot_matrix(entity_rotation)
    camera_m = create_rot_matrix(camera_rotation)
    relative_m = entity_m @ camera_m
    relative_rot = rot_matrix_to_euler_angles(relative_m)
    return relative_rot


def bbox_3d_side_to_normal_vector(points):
    """
    I have 4 points, to define the plane, I use just 3, A, B, C.
    Vectors AB and AC lie on the plane, so their cross product is normal to the plane
    :param points:
    :return:
    """
    a = points[0, :]
    b = points[1, :]
    c = points[2, :]
    ab = b - a
    ac = c - a
    return np.cross(ac, ab)


def is_side_visible(points, camera_position):
    """
    Because camera rotation is not part of the calculation, I assume here the camera which sees in 360°,
    but that is handled in entities filtering, where only visible entities are passed into this function.
    Decides whether the 3d bounding box side is visible or not.
    Based on dot product of camera (camera direction vector) and side plane normal.
    vzít paprsek z kamery do středu čtverce a dot product s normálou plochy toho čtverce
    to určí, zda by to bylo vidět na kameře co má záběr vše, pak musím otestovatasi přes NDC, zda je ve viditelnén jehlanu kamery

    :param points: points in world coordinates, 3D
    :param camera_position: in world coordinates, 3D
    :param camera_rotation: in world coordinates, 3D
    :return:
    """
    side_middle = np.mean(points,
                          axis=0)  # need to calculate it by mean, because the square is projected, it is not square anymore,
    # in the 2D, it is generally convex quadrilateral
    camera_to_side_middle = side_middle - camera_position
    side_normal = bbox_3d_side_to_normal_vector(points)
    product = np.dot(camera_to_side_middle, side_normal)
    if product < 0:  # negative dot product means the side is visible
        return 1
    return 0


def json_to_toyota_format(data, depth, stencil):
    """"
        instructions for toyota format are following, only entities are dumped, each line is for one entity
        The vehicle annotations again contain one line per vehicle, and the columns are:

        Vehicle ID
        Annotation status (should be 5, meaning confirmed by human annotator)
        Occlusion level (by other annotated vehicles, zero to one, zero for fully visible vehicle)
        Out-of-image level (zero to one, zero for vehicles fully in the image)
        Vehicle category (enumeration, see below)
        Vehicle length (in mm)
        Vehicle width (in mm)
        Vehicle height (in mm)
        Location X, Y, Z (in mm, the coordinate system is X to the right, Y forward and Z up)
        Heading (in degrees, respective to egovehicle, 0 is parallel with egovehicle)
        Distance to the camera (in mm)
        Orientation w.r.t camera (in degrees, 0 if facing rear side, 180 if front)
        2D bounding box, Left, Top, Right, Bottom (in pixels)
        Vehicle side visibility, Rear, Front, Left, Right (boolean)
        2D projections of 3D bounding box corners, Front-left-ground, Front-right-ground, Rear-left-ground, Rear-right-ground, Front-left-top, Front-right-top, Rear-left-top, Rear-right-top (each X and Y, in pixels)

        The vehicle categories that we are working with are the following:
        VEHICLE_PASSENGER_CAR = 0,   // cca 4.5m length
        VEHICLE_CITY_CAR = 1,                // cca 3.5m length
        VEHICLE_SUV = 2,
        VEHICLE_PICKUP = 3,
        VEHICLE_VAN = 4,
        VEHICLE_TRUCK = 5,
        VEHICLE_BUS = 6,
        VEHICLE_TRACTOR = 7,
        VEHICLE_TRAILER = 8,
        //VEHICLE_TRACTOR_WITH_TRAILER = 9,
        VEHICLE_MILITARY = 10,
        VEHICLE_AGRICULTURE = 11,
        VEHICLE_OTHER = 12

        example:
        0 5 0.3 0    0 4800 2000 1650    -1977.85 46243.1 -96.4865    0     43740.9    2.59195    623.995 216.535 643.952 231.946    1 0 0 1    626.854 231.342  643.886 231.467  623.911 231.975  642.912 232.115  626.94 217.294  643.99 217.393  624.003 216.302  643.026 216.406
        1 5 0 0    0 4800 2000 1800    3030.15 15151.7 -216.567    7.60935     12982.4    -5.91317    713.066 191.88 790.341 253.72    1 0 1 0    712.899 246.571  763.834 247.054  716.722 255.763  790.197 256.175  713.465 200.34  764.547 201.343  717.564 188.488  791.33 190.471
        46 5 0 0    0 4600 1800 1550    -3108.6 6248.44 -49.552    7.00659     4868.65    47.0695    168.146 126.213 524.949 294.797    1 0 0 1    436.697 259.641  527.039 264.411  170.04 283.114  225.506 313.484  436.017 174.593  526.912 166.347  166.825 150.469  220.872 102.884
        62 5 0.233848 0.204334    5 8850 2500 4150    -11129.6 4460.82 8.26664    -0.0767804     11305.8    80.1913    -66.2909 103.905 257.425 255.671    1 0 0 1    220.564 244.871  258.869 246.984  -44.0267 253.452  -63.0191 258.145  218.026 129.744  256.144 111.575  -47.8095 127.142  -67.7169 103.399
        943 5 0 0    8 13600 2500 3600    1307.36 34695.9 -71.2121    9.00267     32176.8    6.67362    637.821 184.629 688.925 234.98    1 0 1 0    663.667 232.386  688.794 232.668  636.993 235.527  675.197 235.983  663.985 196.208  689.179 196.268  637.372 180.515  675.736 180.179
    """
    lines = []

    view_matrix = np.array(data['view_matrix'])
    proj_matrix = np.array(data['proj_matrix'])
    width = data['width']
    height = data['height']

    cam_pos = np.array(data['camera_pos'])
    cam_rot = np.array(data['camera_rot'])
    cam_rel_pos = np.array(data['camera_relative_position'])
    cam_rel_rot = np.array(data['camera_relative_rotation'])

    visible_vehicles = [e for e in data['entities'] if
                        e['type'] == 'car' and
                        e['class'] != 'Trains' and
                        is_entity_in_image(depth, stencil, e, view_matrix, proj_matrix, width, height,
                                           vehicle_stencil_ratio=0.1)]

    my_car_position, my_car_rotation = get_my_car_position_and_rotation(cam_pos, cam_rot, cam_rel_pos, cam_rel_rot)
    my_car_position += get_gta_center_to_toyota_center(get_my_car_model_sizes())

    for entity in visible_vehicles:
        vehicle_id = entity['handle']
        annotation_status = 5  # je to fuk, nechat 5
        occlusion_level = occlusion_2d_bbox_ratio(entity, visible_vehicles, view_matrix, proj_matrix, width, height,
                                                  my_car_position, my_car_rotation)
        out_of_image_level = out_of_image_2dbbox_ratio(entity, view_matrix, proj_matrix, width,
                                                       height)  # z 2d bounding boxů
        vehicle_category = vehicle_type_gta_to_toyota(entity['class'])

        x_min, x_max, y_min, y_max, z_min, z_max = entity['model_sizes']

        vehicle_length = (y_max - y_min) * 1000  # in mm
        vehicle_width = (x_max - x_min) * 1000
        vehicle_height = (z_max - z_min) * 1000
        # location je bod na zemi pod středem auta, takže je třeba to napočítat posunem v rámci 3d bounding boxu
        # location is relative to the car center
        # tuple because of lru_cache hashability
        location = location_to_toyota_ego(tuple(entity['model_sizes']), entity['pos'], entity['rot'], my_car_position,
                                          my_car_rotation)
        location *= 1000
        # gta camera view: the coordinate system is X to the right, Y up and Z backward)
        # toyota: the coordinate system is X to the right, Y forward and Z up)
        location_x = location[0]
        location_y = -location[2]
        location_z = location[1]

        vehicle_rotation = vehicle_rotation_relative_to_my_car(entity['rot'], my_car_rotation)
        heading = 360 - vehicle_rotation[2]  # heading je vůči autě nezávisle na kameře
        distance = np.linalg.norm(location)  # bráno na střed auta v location

        vehicle_rotation_cam = vehicle_rotation_relative_to_camera(entity['rot'], data['camera_rot'])
        orientation = 360 - vehicle_rotation_cam[
            2]  # 0 je v mém směru, úhly po směru hodinových ručiček (90 když vidím zprava, 0 zezadu, 180 zepředu), vůči kameře
        bbox_2d = calculate_2d_bbox_pixels(entity, view_matrix, proj_matrix, width, height)

        bbox_3d_pixel = get_3d_bbox_projected_to_pixels(entity, view_matrix, proj_matrix, width, height)
        bbox_3d_world = get_3d_bbox_projected_to_world(entity)

        bbox_2d_left = bbox_2d[1, 0]
        bbox_2d_top = bbox_2d[1, 1]
        bbox_2d_right = bbox_2d[0, 0]
        bbox_2d_bottom = bbox_2d[0, 1]

        # same (clockwise from outside view) orientation of points sequence is crucial to determine side normal and thus visibility
        side_visibility_rear = is_side_visible(bbox_3d_world[(0, 1, 5, 4), :], cam_pos)
        side_visibility_front = is_side_visible(bbox_3d_world[(2, 6, 7, 3), :], cam_pos)
        side_visibility_left = is_side_visible(bbox_3d_world[(0, 2, 3, 1), :], cam_pos)
        side_visibility_right = is_side_visible(bbox_3d_world[(4, 5, 7, 6), :], cam_pos)

        # todo: to pořadí je po zrotování, ale nebo původního modelu před rotací?
        # front left ground
        bbox_3d_flg_x = bbox_3d_pixel[0, 2]
        bbox_3d_flg_y = bbox_3d_pixel[1, 2]
        bbox_3d_frg_x = bbox_3d_pixel[0, 6]
        bbox_3d_frg_y = bbox_3d_pixel[1, 6]
        bbox_3d_rlg_x = bbox_3d_pixel[0, 0]
        bbox_3d_rlg_y = bbox_3d_pixel[1, 0]
        bbox_3d_rrg_x = bbox_3d_pixel[0, 4]
        bbox_3d_rrg_y = bbox_3d_pixel[1, 4]
        bbox_3d_flt_x = bbox_3d_pixel[0, 3]
        bbox_3d_flt_y = bbox_3d_pixel[1, 3]
        bbox_3d_frt_x = bbox_3d_pixel[0, 7]
        bbox_3d_frt_y = bbox_3d_pixel[1, 7]
        bbox_3d_rlt_x = bbox_3d_pixel[0, 1]
        bbox_3d_rlt_y = bbox_3d_pixel[1, 1]
        bbox_3d_rrt_x = bbox_3d_pixel[0, 5]
        bbox_3d_rrt_y = bbox_3d_pixel[1, 5]
        # rear right top

        line_data = [vehicle_id, annotation_status, occlusion_level, out_of_image_level, vehicle_category,
                     vehicle_length, vehicle_width, vehicle_height,
                     location_x, location_y, location_z, heading, distance, orientation,
                     bbox_2d_left, bbox_2d_top, bbox_2d_right, bbox_2d_bottom,
                     side_visibility_rear, side_visibility_front, side_visibility_left, side_visibility_right,
                     bbox_3d_flg_x, bbox_3d_flg_y, bbox_3d_frg_x, bbox_3d_frg_y, bbox_3d_rlg_x, bbox_3d_rlg_y,
                     bbox_3d_rrg_x, bbox_3d_rrg_y, bbox_3d_flt_x, bbox_3d_flt_y, bbox_3d_frt_x, bbox_3d_frt_y,
                     bbox_3d_rlt_x, bbox_3d_rlt_y, bbox_3d_rrt_x, bbox_3d_rrt_y]
        line = ' '.join([str(i) if type(i) is int else "{0:.2f}".format(i) for i in line_data])
        lines.append(line)

    return '\n'.join(lines)


def construct_toyota_proj_matrix(data):
    m = np.zeros((3, 3))
    m[0, 2] = (data['width'] / 2) - 0.5
    m[1, 2] = (data['height'] / 2) - 0.5
    m[2, 2] = 1
    m[0, 0] = 1127.17986933
    m[1, 1] = 1127.17986933
    return m


def json_to_toyota_calibration(data):
    """
    Kalibrace kamery:
    matice vnitřní kalibrace - na diagonále je ohnisková vzdálenost v pixelech (tj, ohnisková vzdálenost v mm dělená fyzickou velikostí pixelu), v pravém sloupci střed obrázku (v pixelech)
    tři koeficienty radiálního zkreslení, u GTA nuly
    rotační matice kamery vůči autu
    translace kamery (umístění) vůči autu
    rozlišení obrázku
    """

    def matrix_to_string(m):
        return '\n'.join([' '.join([str(i) for i in row]) for row in m])

    def array_to_string(a):
        return ' '.join([str(i) for i in a])

    part_1 = construct_toyota_proj_matrix(data)
    part_2 = [0, 0, 0]
    part_3 = create_model_rot_matrix(data['camera_relative_rotation'])

    # hardcoded for ASEA car used for onroad:
    model_sizes = get_my_car_model_sizes()
    part_4 = data['camera_relative_position'] - get_gta_center_to_toyota_center(model_sizes)
    part_5 = [data['width'], data['height']]
    parts = [
        matrix_to_string(part_1),
        array_to_string(part_2),
        matrix_to_string(part_3),
        array_to_string(part_4),
        array_to_string(part_5),
    ]
    return '\n\n'.join(parts)


def try_json_to_toyota():
    directory = r'D:\output-datasets\onroad-3'
    # base_name = '2018-07-31--18-03-24--143'
    # base_name = '2018-07-31--17-37-21--852'
    # base_name = '2018-07-31--18-34-15--501'
    base_name = '2018-07-31--17-45-30--020'
    rgb_file = os.path.join(directory, '{}.jpg'.format(base_name))
    depth_file = os.path.join(directory, '{}-depth.png'.format(base_name))
    stencil_file = os.path.join(directory, '{}-stencil.png'.format(base_name))
    json_file = os.path.join(directory, '{}.json'.format(base_name))
    rgb = Image.open(rgb_file)
    depth = np.array(Image.open(depth_file))
    depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC
    stencil = np.array(Image.open(stencil_file))
    with open(json_file) as f:
        data = json.load(f)

    txt_data = json_to_toyota_format(data, depth, stencil)
    cam_data = json_to_toyota_calibration(data)

    with open(os.path.join('toyota-format', base_name + '.txt'), mode='w+') as f:
        f.writelines(txt_data)
    with open(os.path.join('toyota-format', base_name + '.cam'), mode='w+') as f:
        f.writelines(cam_data)
    rgb.save(os.path.join('toyota-format', base_name + '.jpg'))


def try_cameras_to_car():
    conn = visualization.get_connection()
    cur = conn.cursor()
    directory = r'D:\output-datasets\onroad-3'
    # base_name = '2018-07-31--18-03-24--143'
    base_name = '2018-07-31--17-37-21--852'

    cur.execute("""SELECT imagepath, player_pos FROM snapshots_view
    WHERE scene_id = (SELECT scene_id
                FROM snapshots_view
                WHERE imagepath = %(imagepath)s)
                """, {'imagepath': base_name})

    results = []
    for row in cur:
        results.append(dict(row))

    for res in results:
        json_file = os.path.join(directory, '{}.json'.format(res['imagepath']))
        with open(json_file) as f:
            data = json.load(f)

        cam_pos = np.array(data['camera_pos'])
        cam_rot = np.array(data['camera_rot'])
        cam_rel_pos = np.array(data['camera_relative_position'])
        cam_rel_rot = np.array(data['camera_relative_rotation'])
        pos, rot = get_my_car_position_and_rotation(cam_pos, cam_rot, cam_rel_pos, cam_rel_rot)
        print('cam pos', cam_pos)
        print('calc car pos', pos)
        print('car pos', res['player_pos'])
        print('cam rot', cam_rot)
        print('cam rel rot', cam_rel_rot)
        print('calc car rot', rot)


def obtain_toyota_projection_matrix_from_image():
    directory = r'D:\output-datasets\onroad-3'
    base_name = '2018-07-31--17-45-30--020'
    rgb_file = os.path.join(directory, '{}.jpg'.format(base_name))
    depth_file = os.path.join(directory, '{}-depth.png'.format(base_name))
    json_file = os.path.join(directory, '{}.json'.format(base_name))
    depth = np.array(Image.open(depth_file))
    depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC
    with open(json_file, mode='r') as f:
        data = json.load(f)

    proj_matrix = np.array(data['proj_matrix'])
    data['cam_far_clip'] = 1000
    points_ndc, pixels = points_to_homo(data, depth, tresholding=True)
    points_meters = ndc_to_view(points_ndc, proj_matrix)
    points_meters *= 1000  # distances are calculated in mm
    points_meters[2, :] *= -1

    points_pixels = np.zeros((3, points_ndc.shape[1]))
    points_pixels[0] = pixels[1] * points_meters[2, :]
    points_pixels[1] = pixels[0] * points_meters[2, :]
    points_pixels[2] = points_meters[2, :]
    # points_pixels must contain point values after the normalization (dividing by depth),
    # so now they must be multiplied by depth

    points_meters = points_meters[0:3, :]

    toyota_proj, residuals, rank, s = np.linalg.lstsq(points_meters.T, points_pixels.T)
    toyota_proj = toyota_proj.T  # need to transpose it because of matrix multiplication from the other side

    # centering to image check
    assert np.isclose((data['width'] / 2) - 0.5, toyota_proj[0, 2])
    assert np.isclose((data['height'] / 2) - 0.5, toyota_proj[1, 2])

    # zeros where they should be checking
    assert np.isclose(0, toyota_proj[0, 1], atol=2e-6)
    assert np.isclose(0, toyota_proj[1, 0], atol=2e-6)
    assert np.isclose(0, toyota_proj[2, 0], atol=2e-6)
    assert np.isclose(0, toyota_proj[2, 1], atol=2e-6)

    assert np.isclose(1, toyota_proj[2, 2])

    # fov should be same for x and y
    # fov in pixel size checking
    assert np.isclose(toyota_proj[0, 0], - toyota_proj[1, 1], atol=5e-1)

    print(toyota_proj.T, ' is the projection matrix')
    print((toyota_proj[0, 0] - toyota_proj[1, 1]) / 2, 'is the fov you are looking for')


if __name__ == '__main__':
    try_json_to_toyota()
    # try_cameras_to_car()
    # obtain_toyota_projection_matrix_from_image()
