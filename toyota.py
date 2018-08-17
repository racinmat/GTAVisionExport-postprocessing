import json
import numpy as np
from gta_math import calculate_2d_bbox, construct_model_matrix, get_model_3dbbox


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
        'Compacts': 'OTHER',# todo: dodělat, kouknout na auta
        'Sedans': 'OTHER',# todo: dodělat, kouknout na auta
        'SUVs': 'SUV',
        'Coupes': 'OTHER',# todo: dodělat, kouknout na auta
        'Muscle': 'OTHER',# todo: dodělat, kouknout na auta
        'SportsClassics': 'OTHER',# todo: dodělat, kouknout na auta
        'Sports': 'OTHER',# todo: dodělat, kouknout na auta
        'Super': 'OTHER',# todo: dodělat, kouknout na auta
        'Motorcycles': 'OTHER',# todo: dodělat, kouknout na auta
        'OffRoad': 'OTHER',# todo: dodělat, kouknout na auta
        'Industrial': 'OTHER',# todo: dodělat, kouknout na auta
        'Utility': 'OTHER',# todo: dodělat, kouknout na auta
        'Vans': 'VAN',
        'Cycles': 'OTHER',# todo: dodělat, kouknout na auta
        'Boats': 'OTHER',
        'Helicopters': 'OTHER',
        'Planes': 'OTHER',
        'Service': 'OTHER',# todo: dodělat, kouknout na auta
        'Emergency': 'OTHER',# todo: dodělat, kouknout na auta
        'Military': 'MILITARY',
        'Commercial': 'OTHER', # todo: dodělat, kouknout na auta
        'Trains': 'OTHER',
        'Unknown': 'OTHER',
    }

    name_to_number = {
        'PASSENGER_CAR': 0,   # cca 4.5m length
        'CITY_CAR': 1,        # cca 3.5m length
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
    model_matrix = construct_model_matrix(entity['pos'], entity['rot'])
    point_homo = model_matrix @ point_homo
    viewed = view_matrix @ point_homo
    projected = proj_matrix @ viewed
    projected /= projected[3, :]
    bbox_3d = projected.T[:, 0:3]
    return bbox_3d


def json_to_toyota_format(data):
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

    for entity in data['entities']:
        # todo: chodce asi přeskakovat?
        if entity['type'] != 'car':
            continue
        vehicle_id = entity['handle']
        annotation_status = 5   # todo: ujistit se, že číslo 5, když je to automatické
        oclussion_level = 0     # todo: z čeho se to má počítat, 3d nebo 2d bounding boxy?
        out_of_image_level = 0     # todo: z čeho se to má počítat, 3d nebo 2d bounding boxy?
        vehicle_category = vehicle_type_gta_to_toyota(entity['class'])

        x_min, x_max, y_min, y_max, z_min, z_max = entity['model_sizes']

        length = (x_max - x_min) * 1000    # in mm
        width = (y_max - y_min) * 1000
        height = (z_max - z_min) * 1000  # todo: zkontrolovat, že to jsou správné souřadnice
        location_x = 0  # todo: lokace čeho to je, jakého rohu?
        location_y = 0
        location_z = 0
        heading = 0     # todo: popis tohohle zní jako by to bylo jen pro přední kameru, ale co pro ostatní?
        distance = 0    # todo: distance jakého bodu auta to má být od kamery?
        orientation = 0     # todo: ujistit se: po směru hodinových ručiček? (jet vpravo znamená 90° a vlevo 270°?). Vzít rotaci auta, rotaci kamery transponovanou, znásobit matice a vymlátit z toho úhly
        bbox_2d = np.array(calculate_2d_bbox(entity, view_matrix, proj_matrix, width, height))
        bbox_3d = get_3d_bbox_projected_to_2d(entity, view_matrix, proj_matrix, width, height)

        bbox_2d_left = 0  # todo: dodělat
        bbox_2d_top = 0
        bbox_2d_right = 0
        bbox_2d_bottom = 0
        side_visibility_rear = 0
        side_visibility_front = 0
        side_visibility_left = 0
        side_visibility_right = 0
        bbox_3d_flg_x = 0  # front left ground
        bbox_3d_flg_y = 0
        bbox_3d_frg_x = 0
        bbox_3d_frg_y = 0
        bbox_3d_rlg_x = 0
        bbox_3d_rlg_y = 0
        bbox_3d_rrg_x = 0
        bbox_3d_rrg_y = 0
        bbox_3d_flt_x = 0
        bbox_3d_flt_y = 0
        bbox_3d_frt_x = 0
        bbox_3d_frt_y = 0
        bbox_3d_rlt_x = 0
        bbox_3d_rlt_y = 0
        bbox_3d_rrt_x = 0
        bbox_3d_rrt_y = 0  # rear right top

        line_data = [vehicle_id, annotation_status, oclussion_level, out_of_image_level, vehicle_category, length, width, height,
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
    with open(r'D:\output-datasets\onroad-3\2018-07-31--18-03-24--143.json') as f:
        data = json.load(f)
    json_to_toyota_format(data)
