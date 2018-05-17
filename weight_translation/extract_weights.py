from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import glob
from tensorflow.python import pywrap_tensorflow

checkpoint_file = 'checkpoint/model.ckpt'

fp = "weights/"

if not os.path.exists(fp):
    os.makedirs(fp)


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
  """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
  """

  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        print("tensor_name: ", key)

        path = fp + str(key).replace("/", "_") + ".npy"
        array = reader.get_tensor(key)

        np.save(path, array)
        print("Saved weights")

    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      print(reader.get_tensor(tensor_name))
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in file_name for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))


print_tensors_in_checkpoint_file(checkpoint_file, None, True)

os.remove('weights/global_step.npy')
for fn in glob.glob('weights/*_ExponentialMovingAverage.npy'):
  os.remove(fn)

# Remove non EMA weights
# for fn in glob.glob('weights/*.npy'):
#     if 'moving' not in fn and 'Exponential' not in fn:
#         os.remove(fn)

for fn in glob.glob('weights/*_RMSProp.npy'):
  os.remove(fn)

for fn in glob.glob('weights/*_RMSProp_1.npy'):
  os.remove(fn)