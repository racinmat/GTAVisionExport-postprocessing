from itertools import groupby
import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import numpy as np
from progressbar import ProgressBar, Percentage, Bar, Counter
import sys
from GTAVisionExport_postprocessing.visualization import *
import datetime
from math import inf


def has_different_positions(obj):
    # set merges same positions together, leacing only different positions
    return len(set([i['position'] for i in obj['snapshots']])) > 1


def analyze_run(run_id):
    conn = get_connection()
    cur = conn.cursor()
    # gets all cars appearing at least twice
    print("going to get detections from database for run {}".format(run_id))

    cur.execute("""SELECT detection_id, type, class, bbox, imagepath, snapshots.snapshot_id, handle, 
                    ARRAY[st_x(pos), st_y(pos), st_z(pos)], created
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
        detection_id = row[0]
        type = row[1]
        type_class = row[2]
        bbox = bbox_from_string(row[3])
        image = row[4]
        snapshot_id = row[5]
        handle = row[6]
        position = row[7]
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

    moving_objects = {i: obj for i, obj in objects.items() if has_different_positions(obj)}
    print("{} objects, {} of them moving, {} of them staying".format(len(objects), len(moving_objects),
                                                                     len(objects) - len(moving_objects)))
    consecutive_frames = [len(obj['snapshots']) for i, obj in moving_objects.items()]
    values_range = range(min(consecutive_frames), max(consecutive_frames) + 1)
    plt.figure()
    plt.xticks(values_range)
    plt.hist(consecutive_frames, bins=values_range, edgecolor='black')
    plt.draw()
    plt.savefig('run_{}_consecutive_frames_hist_{}.png'.format(run_id, datetime.datetime.now().timestamp()))


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
