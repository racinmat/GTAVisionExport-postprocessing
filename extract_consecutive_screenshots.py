from itertools import groupby
import os
import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    havedisplay = (exitval == 0)

if not havedisplay:
    matplotlib.use('Agg')

import numpy as np
from progressbar import ProgressBar, Percentage, Bar, Counter
import sys
from GTAVisionExport_postprocessing.visualization import *
import datetime
from math import inf
import pickle


def has_different_positions(obj):
    # set merges same positions together, leacing only different positions
    return len(set([i['position'] for i in obj['snapshots']])) > 1


def get_pickle_name(run_id):
    return os.path.join('runs', 'run_{}_pickle.rick'.format(run_id))


def save_objects(run_id, objects):
    data_file = get_pickle_name(run_id)
    with open(data_file, 'wb+') as file:
        pickle.dump(objects, file)


def load_objects(run_id):
    data_file = get_pickle_name(run_id)
    with open(data_file, 'rb') as file:
        return pickle.load(file)


def load_snapshot_data(snapshot_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""SELECT snapshot_id, proj_matrix, view_matrix, run_id, imagepath, timestamp, timeofday, camera_pos, 
                    camera_direction
                  FROM snapshots
                    WHERE
                    snapshots.snapshot_id = {0} 
                    """.format(snapshot_id))
    row = cur.fetchone()

    snapshot = {
        'snapshot_id': row['snapshot_id'],
        'proj_matrix': np.array(row['proj_matrix']),
        'view_matrix': np.array(row['view_matrix']),
        'run_id': row['run_id'],
        'image': row['imagepath'],
        'timestamp': row['timestamp'],
        'timeofday': row['timeofday'],
        'camera_pos': row['camera_pos'],
        'camera_direction': row['camera_direction'],
    }
    return snapshot


# def load_detection_data(detection_id):
#     conn = get_connection()
#     cur = conn.cursor()
#
#     cur.execute("""SELECT snapshot_id, detection_id, handle, pos, bbox, bbox3d
#                   FROM detections
#                     WHERE
#                     snapshots.snapshot_id = {0}
#                     """.format(detection_id))
#     row = cur.fetchone()
#
#     return row


def analyze_run(run_id):
    data_file = get_pickle_name(run_id)

    # if os.path.exists(data_file):
    #     return

    conn = get_connection()
    cur = conn.cursor()
    # gets all cars appearing at least twice
    print("going to get detections from database for run {}".format(run_id))

    cur.execute("""SELECT detection_id, type, class, bbox, imagepath, snapshots.snapshot_id, handle, 
                    ARRAY[st_x(pos), st_y(pos), st_z(pos)] as pos, created
                  FROM detections
                    JOIN snapshots ON detections.snapshot_id = snapshots.snapshot_id
                    WHERE
                    snapshots.run_id = {0} 
                    AND NOT bbox @> POINT '(Infinity, Infinity)'
                    AND handle IN (SELECT handle
                      FROM detections GROUP BY handle HAVING count(*) > 1)
                    ORDER BY handle, detection_id
                    """.format(run_id))
    # rows = cur.fetchall()
    # handle is like object ID, yay
    objects = {}

    print("going to process db rows, total: {}".format(cur.rowcount))
    if cur.rowcount == 0:
        return

    # pbar = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', Counter()], maxval=cur.rowcount, fd=os.fdopen(sys.stdout.fileno(), 'w', 1))
    # pbar.start()
    for i, row in enumerate(cur):
        print(i)
        # pbar.update(i + 1)
        # # sys.stdout.write('\n')
        # pbar.fd.write('')
        # pbar.fd.flush()
        # print('')
        detection_id = row['detection_id']
        type = row['type']
        type_class = row['class']
        bbox = bbox_from_string(row['bbox'])
        image = row['imagepath']
        snapshot_id = row['snapshot_id']
        handle = row['handle']
        position = row['pos']
        if handle not in objects:
            props = {
                'type': type,
                'type_class': type_class,
                'handle': handle,
                'snapshots': [],
                'max_snapshot_id': - inf,
                'min_snapshot_id': inf,
            }
            objects[handle] = props

        snapshot = {
            'detection_id': detection_id,
            'bbox': bbox,
            'image': image,
            'snapshot_id': snapshot_id,
            'position': tuple(position),
        }
        objects[handle]['snapshots'].append(snapshot)
        objects[handle]['max_snapshot_id'] = max(objects[handle]['max_snapshot_id'], snapshot_id)
        objects[handle]['min_snapshot_id'] = min(objects[handle]['min_snapshot_id'], snapshot_id)
    # pbar.finish()

    # done building objects, pickling them
    with open(data_file, 'wb+') as file:
        pickle.dump(objects, file)

    # done pickling them, analyzing and plotting them

    # for every handle, it tells True or False, whether this has nonstrop consecutive snapshots
    consecutives = {i: len(o['snapshots']) == (o['max_snapshot_id'] - o['min_snapshot_id']) + 1 for i, o in objects.items()}
    moving_objects = {i: obj for i, obj in objects.items() if has_different_positions(obj) and consecutives[i]}
    print("{} objects, {} of them moving, {} of them staying".format(len(objects), len(moving_objects),
                                                                     len(objects) - len(moving_objects)))
    consecutive_frames = [len(obj['snapshots']) for i, obj in moving_objects.items()]
    values_range = range(min(consecutive_frames), max(consecutive_frames) + 1)
    plt.figure(figsize=(30, 30))
    plt.xticks(values_range)
    plt.hist(consecutive_frames, bins=values_range, edgecolor='black')
    plt.draw()
    plt.savefig('run_{}_consecutive_frames_hist_{}.png'.format(run_id, datetime.datetime.now().timestamp()))
    plt.close()


def get_runs():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""SELECT DISTINCT run_id
                  FROM runs
                    """)
    return [x[0] for x in cur.fetchall()]


def main():
    run_ids = get_runs()
    print(run_ids)
    for run_id in run_ids:
        analyze_run(run_id)

if __name__ == '__main__':
    main()
