import numpy as np
import lmdb
import caffe

env = lmdb.open('mnist_test_lmdb', readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print key, value
