import h5py
import os

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


print("Before")

f = h5py.File("output/models_baseline/rnn.256.128-64.00344-acc0.9459.weights.hdf5", "r")

print([n.decode('utf8') for n in f.attrs['layer_names']])
print([k for k in f.attrs.keys()])

for name in f:
    print(name, len(f[name]))


fname_out = "output/models_baseline/rnn.256.128-64.00344-acc0.9459.weights.copy.hdf5"
if os.path.exists(fname_out):
    os.remove(fname_out)
f_out = h5py.File(fname_out, "w")

list_before_after = [("dense_1", "hidden_rnn_0"),
                                ("dense_2", "hidden_rnn_1"),
                                ("dense_3", "hidden_rnn_2"),
                                ("lstm_1", "lstm")]
for name_before, name_after in list_before_after:
    f.copy(name_before, f_out)
    f_out[name_after] = f_out[name_before]
    del f_out[name_before]

f_out.attrs["layer_names"] = [name_after.encode('utf8') for (name_before, name_after) in list_before_after]
f_out.attrs["backend"] = str(f.attrs['backend']).encode('utf8')
f_out.attrs["keras_version"] = str(f.attrs['keras_version']).encode('utf8')

f_out.close()
f.close()

print("After")

f_after = h5py.File("output/models_baseline/rnn.256.128-64.00344-acc0.9459.weights.copy.hdf5", "r")

print([n.decode('utf8') for n in f_after.attrs['layer_names']])
print([k for k in f_after.attrs.keys()])

for name in f_after:
    print(name, len(f_after[name]))

f_after.close()
