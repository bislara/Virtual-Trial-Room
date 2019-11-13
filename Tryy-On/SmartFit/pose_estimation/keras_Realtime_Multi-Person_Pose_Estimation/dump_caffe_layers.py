
from __future__ import division, print_function
import caffe
import numpy as np
import os

layers_output = 'model/caffe/layers'
caffe_model = 'model/caffe/_trained_COCO/pose_iter_440000.caffemodel'
caffe_proto = 'model/caffe/_trained_COCO/pose_deploy.prototxt'

caffe.set_mode_cpu()
net = caffe.Net(caffe_proto, caffe_model, caffe.TEST)

for layer_name, blob in net.blobs.iteritems():
    print(layer_name, blob.data.shape)

for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    np.save(os.path.join(layers_output, "W_{:s}.npy".format(k)), v[0].data)
    np.save(os.path.join(layers_output, "b_{:s}.npy".format(k)), v[1].data)

print("Done !")
