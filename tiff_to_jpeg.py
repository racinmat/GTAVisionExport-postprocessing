import os

from PIL import Image

if __name__ == '__main__':
    in_directory = './../output'
    out_directory = './../Mask_RCNN/GTA-images'
    files = [
        'info10.tiff',
        'info11.tiff',
        'info17.tiff',
        'info18.tiff',
        'info50.tiff',
        'info51.tiff',
    ]
    for name in files:
        base_name = os.path.splitext(name)[0]
        im = Image.open(os.path.join(in_directory, name))
        im.seek(0)  # 1.frame of tiff
        im = im.convert(mode="RGB")
        print("Generating jpeg for {}".format(name))
        outfile = os.path.join(out_directory, base_name) + ".jpg"
        im.save(outfile)
