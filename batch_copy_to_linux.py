import glob
import os

import progressbar
from joblib import Parallel, delayed
from PIL import Image
from os.path import isfile, join
from shutil import copyfile


def copy_file(in_path, out_path):
    global counter
    counter += 1
    pbar.update(counter)
    if os.path.exists(out_path):
        return
    copyfile(in_path, out_path)


if __name__ == '__main__':
    workers = 20

    in_directory = 'D:\\GTAV_extraction_output\\rgb-jpeg'
    out_directory = 'Y:\\GTA-jpg'
    widgets = [progressbar.Percentage(), ' ', progressbar.Counter(), ' ', progressbar.Bar(), ' ',
               progressbar.FileTransferSpeed()]

    files = [f for f in os.listdir(in_directory) if isfile(join(in_directory, f))]

    pbar = progressbar.ProgressBar(widgets=widgets, max_value=len(files)).start()
    counter = 0

    Parallel(n_jobs=workers, backend='threading')(
        delayed(copy_file)(os.path.join(in_directory, name), os.path.join(out_directory, name)) for name in files)
