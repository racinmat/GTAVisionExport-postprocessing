import glob
import os

from PIL import Image

if __name__ == '__main__':
    in_directory = 'D:\\GTAV_extraction_output'
    # in_directory = 'D:\\projekty\\GTA-V-extractors\\output'
    out_directory = 'D:\\GTAV_extraction_output\\rgb-jpeg'
    # out_directory = 'D:\\projekty\\GTA-V-extractors\\output\\rgb-jpeg'
    pattern = 'info-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9]-[0-9][0-9]-[0-9][0-9]--[0-9][0-9][0-9].tiff'
    frames = [
        0
    ]
    for name in glob.glob(os.path.join(in_directory, pattern)):
        base_name = os.path.basename(os.path.splitext(name)[0])
        for frame in frames:
            outfile = os.path.join(out_directory, base_name) + "-{}.jpg".format(frame)
            if os.path.exists(outfile):
                continue

            im = Image.open(os.path.join(in_directory, name))
            im.seek(frame)
            im = im.convert(mode="RGB")
            print("Generating jpeg for {}".format(name))
            im.save(outfile)
