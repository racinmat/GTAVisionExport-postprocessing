import glob
import os
from joblib import Parallel, delayed
from PIL import Image


def get_base_name(name):
    return os.path.basename(os.path.splitext(name)[0])


def tiff_to_jpg(in_directory, out_directory, base_name, name, frame):
    outfile = os.path.join(out_directory, base_name) + "-{}.jpg".format(frame)
    if os.path.exists(outfile):
        return

    im = Image.open(os.path.join(in_directory, name))
    im.seek(frame)
    im = im.convert(mode="RGB")
    print("Generating jpeg for {}".format(name))
    im.save(outfile)


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

    Parallel(n_jobs=workers)(delayed(tiff_to_jpg)
                             (in_directory, out_directory, get_base_name(name), name, frame) for frame in frames
                             for name in glob.glob(os.path.join(in_directory, pattern)))

    # for name in glob.glob(os.path.join(in_directory, pattern)):
    #     base_name = get_base_name(name)
    #     for frame in frames:
    #         tiff_to_jpg(in_directory, out_directory, base_name, frame)
