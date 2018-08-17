import numpy as np
import os
from gta_math import points_to_homo, ndc_to_view, view_to_world
from PIL import Image
import json


def save_csv(vecs_p, name):
    a = np.asarray(vecs_p[0:3, :].T)
    np.savetxt("points-{}.csv".format(name), a, delimiter=",")


directory = r'D:\output-datasets\offroad-7\0'

file_name = '2018-08-13--11-15-01--499'
rgb_file = os.path.join(directory, '{}.jpg'.format(file_name))
depth_file = os.path.join(directory, '{}-depth.png'.format(file_name))
json_file = os.path.join(directory, '{}.json'.format(file_name))

rgb = np.array(Image.open(rgb_file))
depth = np.array(Image.open(depth_file))
depth = depth / np.iinfo(np.uint16).max  # normalizing into NDC
with open(json_file, mode='r') as f:
    data = json.load(f)
data['proj_matrix'] = np.array(data['proj_matrix'])
data['view_matrix'] = np.array(data['view_matrix'])

vecs, _ = points_to_homo(data, depth, tresholding=False)
vecs_p = ndc_to_view(vecs, np.array(data['proj_matrix']))
vecs_p_world = view_to_world(vecs_p, np.array(data['view_matrix']))
save_csv(vecs_p, 'my-points-'+file_name)
