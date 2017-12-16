import glob
import os

import progressbar
from joblib import Parallel, delayed
from PIL import Image


def get_base_name(name):
    return os.path.basename(os.path.splitext(name)[0])


def tiff_to_jpg(in_directory, out_directory, out_name, name, frame):
    if 'pbar' in globals() and 'counter' in globals():
        global counter
        counter += 1
        pbar.update(counter)

    outfile = os.path.join(out_directory, out_name)
    if os.path.exists(outfile):
        return

    try:
        im = Image.open(os.path.join(in_directory, name))
        im.seek(frame)
        im = im.convert(mode="RGB")
        # print("Generating jpeg for {}".format(name))
        im.save(outfile)
    except OSError:
        # print("Skipping invalid file {}".format(name))
        return


if __name__ == '__main__':
    workers = 10
    in_directory = 'D:\\GTAV_extraction_output'
    # in_directory = 'D:\\projekty\\GTA-V-extractors\\output'
    out_directory = 'D:\\GTAV_extraction_output\\rgb-jpeg'
    # out_directory = 'D:\\projekty\\GTA-V-extractors\\output\\rgb-jpeg'
    pattern = 'info-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9][0-9].tiff'
    frames = [
        0
    ]

    widgets = [progressbar.Percentage(), ' ', progressbar.Counter(), ' ', progressbar.Bar(), ' ',
               progressbar.FileTransferSpeed()]

    files = glob.glob(os.path.join(in_directory, pattern))

    pbar = progressbar.ProgressBar(widgets=widgets, max_value=len(files) * len(frames)).start()
    counter = 0

    Parallel(n_jobs=workers, backend='threading')(delayed(tiff_to_jpg)
                             (in_directory, out_directory, "{}-{}.jpg".format(get_base_name(name), frame), name, frame) for frame in frames
                             for name in files)

    # for name in glob.glob(os.path.join(in_directory, pattern)):
    #     base_name = get_base_name(name)
    #     for frame in frames:
    #         tiff_to_jpg(in_directory, out_directory, "{}-{}.jpg".format(get_base_name(name), frame), name, frame)
