from itertools import groupby

import numpy as np
from progressbar import ProgressBar, Percentage, Bar, Counter

from GTAVisionExport_postprocessing.visualization import *


def has_different_positions(obj):
    # set merges same positions together, leacing only different positions
    return len(set([i['position'] for i in obj['snapshots']])) > 1


def main():
    conn = get_connection()
    cur = conn.cursor()
    # gets all cars appearing at least twice
    cur.execute("""SELECT detection_id, type, class, bbox, imagepath, snapshots.snapshot_id, handle, ARRAY[st_x(pos), st_y(pos), st_z(pos)], created
                  FROM detections
                    JOIN snapshots ON detections.snapshot_id = snapshots.snapshot_id
                    WHERE NOT bbox @> POINT '(Infinity, Infinity)'
                    AND handle IN (SELECT handle
                      FROM detections GROUP BY handle HAVING count(*) > 1)
                    ORDER BY handle, detection_id""")
    # rows = cur.fetchall()
    # handle is like object ID, yay
    objects = {}

    print("going to process db rows")
    pbar = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', Counter()], maxval=cur.rowcount)
    pbar.start()
    for i, row in enumerate(cur):
        pbar.update(i + 1)
        print('')
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
                'snapshots': []
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
    pbar.finish()

    moving_objects = {i: obj for i, obj in objects.items() if has_different_positions(obj)}
    print("{} objects, {} of them moving, {} of them staying".format(len(objects), len(moving_objects),
                                                                     len(objects) - len(moving_objects)))
    consecutive_frames = [len(obj['snapshots']) for i, obj in moving_objects.items()]
    values_range = range(min(consecutive_frames), max(consecutive_frames) + 1)
    plt.figure()
    plt.xticks(values_range)
    plt.hist(consecutive_frames, bins=values_range, edgecolor='black')
    plt.draw()
    plt.savefig('consecutive_frames_hist.png')


if __name__ == '__main__':
    main()
