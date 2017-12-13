import pickle

import time

from joblib import Parallel, delayed
from progressbar import ProgressBar

from extract_consecutive_screenshots import get_pickle_name, load_objects
from visualization import get_connection


def some_operation(i):
    pbar.update(i)
    time.sleep(0.2)
    print('{} done'.format(i))


if __name__ == '__main__':
    # conn = get_connection()
    # run_id = 11
    # cars = load_objects(run_id)
    # car1 = cars[cars.keys()[0]]
    workers = 20
    size = 40
    p_start = time.time()
    pbar = ProgressBar(maxval=size).start()
    Parallel(n_jobs=workers, backend='threading')(delayed(some_operation)(i) for i in range(size))
    pbar.finish()
    p_end = time.time()
    n_start = time.time()
    pbar = ProgressBar(maxval=size).start()
    for i in range(size):
        some_operation(i)
    pbar.finish()
    n_end = time.time()
    print('parallel: {}, normal: {}'.format(p_end - p_start, n_end - n_start))

    pass
