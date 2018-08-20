import json
import os

import numpy as np
from PIL import Image

from gta_math import calculate_2d_bbox, construct_model_matrix, get_model_3dbbox, is_entity_in_image, \
    model_coords_to_pixel, construct_view_matrix, create_rot_matrix


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
    # todo: dodělat, kouknout na auta
    mapping = {
        'Compacts': 'OTHER',  # todo: dodělat, kouknout na auta
        'Sedans': 'OTHER',  # todo: dodělat, kouknout na auta
        'SUVs': 'SUV',
        'Coupes': 'OTHER',  # todo: dodělat, kouknout na auta
        'Muscle': 'OTHER',  # todo: dodělat, kouknout na auta
        'SportsClassics': 'OTHER',  # todo: dodělat, kouknout na auta
        'Sports': 'OTHER',  # todo: dodělat, kouknout na auta
        'Super': 'OTHER',  # todo: dodělat, kouknout na auta
        'Motorcycles': 'OTHER',  # todo: dodělat, kouknout na auta
        'OffRoad': 'OTHER',  # todo: dodělat, kouknout na auta
        'Industrial': 'OTHER',  # todo: dodělat, kouknout na auta
        'Utility': 'OTHER',  # todo: dodělat, kouknout na auta
        'Vans': 'VAN',
        'Cycles': 'OTHER',  # todo: dodělat, kouknout na auta
        'Boats': 'OTHER',
        'Helicopters': 'OTHER',
        'Planes': 'OTHER',
        'Service': 'OTHER',  # todo: dodělat, kouknout na auta
        'Emergency': 'OTHER',  # todo: dodělat, kouknout na auta
        'Military': 'MILITARY',
        'Commercial': 'OTHER',  # todo: dodělat, kouknout na auta
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
    return str(name_to_number[mapping[gta_type]])


def get_3d_bbox_projected_to_2d(entity, view_matrix, proj_matrix, width, height):
    model_sizes = np.array(entity['model_sizes'])
    points_3dbbox = get_model_3dbbox(model_sizes)
    point_homo = np.array(
        [points_3dbbox[:, 0], points_3dbbox[:, 1], points_3dbbox[:, 2], np.ones_like(points_3dbbox[:, 0])])
    bbox_3d = model_coords_to_pixel(entity['pos'], entity['rot'], point_homo.T, view_matrix, proj_matrix, width, height)
    return bbox_3d


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
    in_image = np.copy(bbox_2d)
    in_image[0, :] = np.minimum(bbox_2d[0, :], image[0, :])
    in_image[1, :] = np.maximum(bbox_2d[1, :], image[1, :])
    in_image_volume = (in_image[0, 0] - in_image[1, 0]) * (in_image[0, 1] - in_image[1, 1])
    whole_volume = (bbox_2d[0, 0] - bbox_2d[1, 0]) * (bbox_2d[0, 1] - bbox_2d[1, 1])
    return in_image_volume / whole_volume


def get_my_car_position_and_rotation(cam_pos, cam_rot, cam_rel_pos, cam_rel_rot):
    cam_pos = np.concatenate((cam_pos, [1]))
    # cam_to_car_m = construct_view_matrix(cam_rel_pos, -cam_rel_rot)

    view_matrix = np.zeros((4, 4))
    # view_matrix[0:3, 3] = camera_pos
    view_matrix[0:3, 0:3] = create_rot_matrix(cam_rot) @ create_rot_matrix(cam_rel_rot)
    view_matrix[3, 3] = 1

    trans_matrix = np.eye(4)
    trans_matrix[0:3, 3] = -cam_rel_pos

    # return view_matrix
    cam_to_car_m = trans_matrix @ view_matrix
    # todo: dodělat získávání rotace a vůbec věci
    car_pos = cam_to_car_m @ cam_pos
    car_pos /= car_pos[3]
    return car_pos[0:3]


def location_to_toyota(entity, my_car_position, my_car_rotation):
    """
    Returns location of vehicle relatively to the ego car center.
    Car center is calculated from camera position and camera relative position and rotation, and thus is same
    for all cameras, rotation is taken as ego vehicle rotation, which is calculated from camera rotation and camera relative rotation
    """
    x_min, x_max, y_min, y_max, z_min, z_max = entity['model_sizes']
    gta_center_to_corner = np.array([x_min, y_min, z_min])
    corner_to_toyota_center = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2, 0])
    gta_center_to_toyota_center = gta_center_to_corner + corner_to_toyota_center
    # location is relative to the car center
    world_location = entity['pos'] + gta_center_to_toyota_center
    world_to_car_m = construct_view_matrix(my_car_position, my_car_rotation)
    view_location = (world_to_car_m @ world_location)
    view_location /= view_location[3]
    return view_location[0:3]


def json_to_toyota_format(data, depth, stencil):
    """"
        instructions for toyota format are following, only entities are dumped, each line is for one entity
        The vehicle annotations again contain one line per vehicle, and the columns are:

        Vehicle ID
        Annotation status (should be 5, meaning confirmed by human annotator)
        Occlusion level (by other annotated vehicles, zero to one)
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
                        e['type'] == 'car' and e['class'] != 'Trains' and is_entity_in_image(depth, stencil, e,
                                                                                             view_matrix,
                                                                                             proj_matrix, width,
                                                                                             height)]

    for entity in visible_vehicles:
        vehicle_id = entity['handle']
        annotation_status = 5  # je to fuk, nechat 5
        oclussion_level = 0  # z 2d bounding boxů
        out_of_image_level = out_of_image_2dbbox_ratio(entity, view_matrix, proj_matrix, width, height)  # z 2d bounding boxů
        vehicle_category = vehicle_type_gta_to_toyota(entity['class'])

        x_min, x_max, y_min, y_max, z_min, z_max = entity['model_sizes']

        vehicle_length = (y_max - y_min) * 1000  # in mm
        vehicle_width = (x_max - x_min) * 1000
        vehicle_height = (z_max - z_min) * 1000
        # location je bod na zemi pod středem auta, takže je třeba to napočítat posunem v rámci 3d bounding boxu
        # location is relative to the car center
        my_car_position, my_car_rotation = get_my_car_position_and_rotation(cam_pos, cam_rot, cam_rel_pos, cam_rel_rot)
        location_to_toyota(entity, my_car_position, my_car_rotation)
        location_x = 0
        location_y = 0
        location_z = 0
        heading = 0  # heading je vůči autě nezávisle na kameře
        distance = 0  # bráno na střed auta v location
        orientation = 0  # todo: ujistit se: po směru hodinových ručiček? (jet vpravo znamená 90° a vlevo 270°?). Vzít rotaci auta, rotaci kamery transponovanou, znásobit matice a vymlátit z toho úhly
        bbox_2d = np.array(calculate_2d_bbox(entity, view_matrix, proj_matrix, width, height))
        bbox_2d[:, 0] *= width
        bbox_2d[:, 1] *= height

        bbox_3d = get_3d_bbox_projected_to_2d(entity, view_matrix, proj_matrix, width, height)

        bbox_2d_left = bbox_2d[1, 0]
        bbox_2d_top = bbox_2d[1, 1]
        bbox_2d_right = bbox_2d[0, 0]
        bbox_2d_bottom = bbox_2d[0, 1]
        side_visibility_rear = 0
        side_visibility_front = 0
        side_visibility_left = 0
        side_visibility_right = 0
        # todo: to pořadí je po zrotování, ale nebo původního modelu před rotací?
        # front left ground
        bbox_3d_flg_x = bbox_3d[0, 0]
        bbox_3d_flg_y = bbox_3d[0, 1]
        bbox_3d_frg_x = bbox_3d[1, 0]
        bbox_3d_frg_y = bbox_3d[1, 1]
        bbox_3d_rlg_x = bbox_3d[2, 0]
        bbox_3d_rlg_y = bbox_3d[2, 1]
        bbox_3d_rrg_x = bbox_3d[3, 0]
        bbox_3d_rrg_y = bbox_3d[3, 1]
        bbox_3d_flt_x = bbox_3d[4, 0]
        bbox_3d_flt_y = bbox_3d[4, 1]
        bbox_3d_frt_x = bbox_3d[5, 0]
        bbox_3d_frt_y = bbox_3d[5, 1]
        bbox_3d_rlt_x = bbox_3d[6, 0]
        bbox_3d_rlt_y = bbox_3d[6, 1]
        bbox_3d_rrt_x = bbox_3d[7, 0]
        bbox_3d_rrt_y = bbox_3d[7, 1]
        # rear right top

        line_data = [vehicle_id, annotation_status, oclussion_level, out_of_image_level, vehicle_category,
                     vehicle_length, vehicle_width, vehicle_height,
                     location_x, location_y, location_z, heading, distance, orientation,
                     bbox_2d_left, bbox_2d_top, bbox_2d_right, bbox_2d_bottom,
                     side_visibility_rear, side_visibility_front, side_visibility_left, side_visibility_right,
                     bbox_3d_flg_x, bbox_3d_flg_y, bbox_3d_frg_x, bbox_3d_frg_y, bbox_3d_rlg_x, bbox_3d_rlg_y,
                     bbox_3d_rrg_x, bbox_3d_rrg_y, bbox_3d_flt_x, bbox_3d_flt_y, bbox_3d_frt_x, bbox_3d_frt_y,
                     bbox_3d_rlt_x, bbox_3d_rlt_y, bbox_3d_rrt_x, bbox_3d_rrt_y]
        line = ' '.join(line_data)
        lines.append(line)

    return '\r\n'.join(lines)


if __name__ == '__main__':
    directory = r'D:\output-datasets\onroad-3'
    # base_name = '2018-07-31--18-03-24--143'
    base_name = '2018-07-31--17-37-21--852'
    depth_file = os.path.join(directory, '{}-depth.png'.format(base_name))
    stencil_file = os.path.join(directory, '{}-stencil.png'.format(base_name))
    json_file = os.path.join(directory, '{}.json'.format(base_name))
    depth = np.array(Image.open(depth_file))
    depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC
    stencil = np.array(Image.open(stencil_file))
    with open(json_file) as f:
        data = json.load(f)
    json_to_toyota_format(data, depth, stencil)
