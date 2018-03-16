import numpy as np
import matplotlib.pyplot as plt
import visualization
from math import tan, atan, radians, degrees

def construct_proj_matrix():
    fov = 50.0
    # for z coord
    f = 1.5  # the near clip, but f in the book
    n = 10003  # the far clip, somewhere in the ned of the world
    # x coord
    H = 1080
    W = 1914
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


if __name__ == '__main__':
    ini_file = "gta-postprocessing.ini"
    visualization.multi_page = False
    visualization.ini_file = ini_file

    conn = visualization.get_connection()
    cur = conn.cursor()
    cur.execute("""SELECT snapshot_id, imagepath, proj_matrix \
      FROM snapshots \
      WHERE run_id = 6
      ORDER BY snapshot_id DESC \
      """)

    results = []
    for row in cur:
        res = dict(row)
        res['proj_matrix'] = np.array(res['proj_matrix'])
        results.append(res)

    print('There are {} records'.format(len(results)))
    projs = [i['proj_matrix'] for i in results]

    # todo: estimate far plane by least squares. The rest is known. Before that, get rid of all outliers
    # x00 = proj[0, 0]
    # x11 = proj[1, 1]
    x22s = np.array([proj[2, 2] for proj in projs])
    x23s = np.array([proj[2, 3] for proj in projs])

    ns = -x23s / x22s
    fs = -x23s / (x22s - 1)

    # filtering values
    print('weird x22 value in proj: ')
    print([proj for proj in projs if abs(proj[2, 2] - (-1)) < 1e-4])

    # showing plots with values
    fig, axes = plt.subplots(1, 3)
    # axes[0].plot(x22s, np.zeros_like(x22s), 'x')
    axes[0].set_yscale('log', nonposy='clip')
    axes[0].set_title('$X_{22}$')
    axes[0].hist(x22s, bins=1000)

    axes[1].set_yscale('log', nonposy='clip')
    axes[1].set_title('$X_{23}$')
    axes[1].hist(x23s, bins=1000)

    axes[2].set_title('both')
    axes[2].set_xlabel('$X_{22}$')
    axes[2].set_ylabel('$X_{23}$')
    axes[2].loglog(x22s, x23s, 'o')
    plt.show()

    # print('n:', n)
    # print('f:', f)

    # fov = res['camera_fov']
    # print('fov: ', fov)
    # tan(radians(fov))
    # n = 1.5
    # t = n * tan(radians(fov) / 2)
    # print('t: ', t)
    # W = res['width']
    # H = res['height']
    # fov_h = fov * W / H
    # print('fov_h: ', fov_h)
    # r = n * tan(radians(fov_h) / 2)
    # print('r: ', r)
    # print('W: ', W)
    # print('H: ', H)
    # print('near clip: ', near_clip)
    # print('x00_c: ', 1 / tan(radians(fov_h) / 2))
    # print('x00_c2: ', H / (W * tan(radians(fov) / 2)))
    # print('x11_c: ', 1 / tan(radians(fov) / 2))
    # print('x00: ', x00)
    # print('x11: ', x11)
    #
    # fov_c_v = degrees(atan(1 / x11)) * 2
    # print('fov_c_v: ', fov_c_v)
    # fov_c_h = fov_c_v * W / H
    # print('fov_c_h: ', fov_c_h)
    # fov_c_h2 = degrees(atan(1 / x00)) * 2
    # print('fov_c_h2: ', fov_c_h2)
    #
    # H = res['height']
    # # H=res['ui_height']
    # print('for height {}, width is {}'.format(H, (fov_c_v * H) / fov_c_h2))
    # print('for height {}, width is {}'.format(H, (tan(radians(fov_c_h2)) * H) / tan(radians(fov_c_v))))
    # print('ui_width', res['ui_width'])
    # print('width', res['width'])
    #
    # print('aspect ration from matrix:', x11 / x00)
    # print('H/W:', H / W)
    # print('aspect ration from db:', res['width'] / res['height'])
    # print('aspect ration from ui_db:', res['ui_width'] / res['ui_height'])
    # print('width by db ratio:', (x11 / x00) * res['height'])
