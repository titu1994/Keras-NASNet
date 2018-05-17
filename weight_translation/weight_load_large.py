import numpy as np
import glob
import os

all_files = []

conv0_path = 'weights/conv0_*.npy'
conv0_fns = (glob.glob(conv0_path))
all_files.extend(conv0_fns)

stem_0_path = 'weights/cell_stem_0_*.npy'
stem_0_fns = (glob.glob(stem_0_path))
all_files.extend(stem_0_fns)

stem_1_path = 'weights/cell_stem_1_*.npy'
stem_1_fns = (glob.glob(stem_1_path))
all_files.extend(stem_1_fns)

# alias for stems
stems = [stem_0_fns, stem_1_fns]

normal_path_base = 'weights/cell_'
normal_fns = {}

NB_CELLS = 18
for i in range(NB_CELLS):
    normal_path = normal_path_base + "%d_*.npy" % i
    normal_fn = (glob.glob(normal_path))
    normal_fns[i] = normal_fn

    all_files.extend(normal_fn)

reduction_0_path = 'weights/reduction_cell_0_*.npy'
reduction_0_fns = (glob.glob(reduction_0_path))
all_files.extend(reduction_0_fns)

reduction_1_path = 'weights/reduction_cell_1_*.npy'
reduction_1_fns = (glob.glob(reduction_1_path))
all_files.extend(reduction_1_fns)

# alias for reduction cells
reduction_cells = [reduction_0_fns, reduction_1_fns]

aux_path = 'weights/aux_*.npy'
aux_fns = glob.glob(aux_path)
all_files.extend(aux_fns)

final_dense_path = 'weights/final_layer_FC_*.npy'
final_dense_fns = (glob.glob(final_dense_path))
all_files.extend(final_dense_fns)

all_files_paths = (glob.glob('weights/*.npy'))
total_file_count = len(all_files_paths)

print('Total number of weight files : ', total_file_count)
print('Properly loaded weights count : ', len(all_files))

if total_file_count != len(all_files):
    print("Printing files not loaded yet")
    print()

    for fn in all_files_paths:
        if fn not in all_files:
            print(fn)

    exit()
else:
    print("All files loaded up properly")
    print()


def correct_bn(bn_weights):
    return [bn_weights[1], bn_weights[0], bn_weights[2], bn_weights[3]]

def load_conv0():
    print("Loading conv 0 weights")
    conv0_weights = np.load(conv0_fns[-1])
    conv0_bn = [np.load(fn) for fn in conv0_fns[:4]]
    return {'w': conv0_weights, 'bn': correct_bn(conv0_bn)}

def load_stem_0():
    print("Loading stem 0 weights")

    stem_fn = stems[0]
    stem_beginning_weights = np.load(stem_fn[0])
    stem_beginning_bn = [np.load(fn) for fn in stem_fn[1:5]]

    stem0 = {'begin_W': stem_beginning_weights,
             'begin_bn': correct_bn(stem_beginning_bn)}

    for i in range(5):
        left_path = 'weights/cell_stem_%d_comb_iter_%d_left_*.npy' % (0, i)
        right_path = 'weights/cell_stem_%d_comb_iter_%d_right_*.npy' % (0, i)
        left_fns = (glob.glob(left_path))
        right_fns = (glob.glob(right_path))

        if len(left_fns) > 0:
            print("Loaded left at %d" % i)
            left_bn_1 = [np.load(fn) for fn in left_fns[:4]]
            left_bn_2 = [np.load(fn) for fn in left_fns[4:8]]
            left_d1 = np.load(left_fns[8])
            left_p1 = np.load(left_fns[9])
            left_d2 = np.load(left_fns[10])
            left_p2 = np.load(left_fns[11])

        if len(right_fns) > 0:
            print("Loaded right at %d" % i)
            right_bn_1 = [np.load(fn) for fn in right_fns[:4]]
            right_bn_2 = [np.load(fn) for fn in right_fns[4:8]]
            right_d1 = np.load(right_fns[8])
            right_p1 = np.load(right_fns[9])
            right_d2 = np.load(right_fns[10])
            right_p2 = np.load(right_fns[11])

        if len(left_fns) > 0:
            stem0['left_%d' % i] = {'d1': left_d1, 'd2': left_d2, 'p1': left_p1, 'p2': left_p2,
                                    'bn1': correct_bn(left_bn_1), 'bn2': correct_bn(left_bn_2)}

        if len(right_fns) > 0:
            stem0['right_%d' % i] = {'d1': right_d1, 'd2': right_d2, 'p1': right_p1, 'p2': right_p2,
                                     'bn1': correct_bn(right_bn_1), 'bn2': correct_bn(right_bn_2)}

    return stem0

def load_stem_1():
    print("Loading stem 1 weights")

    stem_fn = stems[1]
    stem_beginning_weights = np.load(stem_fn[0])
    stem_beginning_bn = [np.load(fn) for fn in stem_fn[1:5]]

    stem1 = {'begin_W': stem_beginning_weights,
             'begin_bn': correct_bn(stem_beginning_bn)}

    for i in range(5):
        left_path = 'weights/cell_stem_%d_comb_iter_%d_left_*.npy' % (1, i)
        right_path = 'weights/cell_stem_%d_comb_iter_%d_right_*.npy' % (1, i)
        left_fns = (glob.glob(left_path))
        right_fns = (glob.glob(right_path))

        if len(left_fns) > 0:
            print("Loaded left at %d" % i)
            left_bn_1 = [np.load(fn) for fn in left_fns[:4]]
            left_bn_2 = [np.load(fn) for fn in left_fns[4:8]]
            left_d1 = np.load(left_fns[8])
            left_p1 = np.load(left_fns[9])
            left_d2 = np.load(left_fns[10])
            left_p2 = np.load(left_fns[11])

        if len(right_fns) > 0:
            print("Loaded right at %d" % i)
            right_bn_1 = [np.load(fn) for fn in right_fns[:4]]
            right_bn_2 = [np.load(fn) for fn in right_fns[4:8]]
            right_d1 = np.load(right_fns[8])
            right_p1 = np.load(right_fns[9])
            right_d2 = np.load(right_fns[10])
            right_p2 = np.load(right_fns[11])

        if len(left_fns) > 0:
            stem1['left_%d' % i] = {'d1': left_d1, 'd2': left_d2, 'p1': left_p1, 'p2': left_p2,
                                    'bn1': correct_bn(left_bn_1), 'bn2': correct_bn(left_bn_2)}

        if len(right_fns) > 0:
            stem1['right_%d' % i] = {'d1': right_d1, 'd2': right_d2, 'p1': right_p1, 'p2': right_p2,
                                     'bn1': correct_bn(right_bn_1), 'bn2': correct_bn(right_bn_2)}

    final_bn = (glob.glob('weights/cell_stem_1_final_path_bn_*.npy'))
    final_bn = [np.load(fn) for fn in final_bn]
    path1_conv = np.load(stem_fn[-2])
    path2_conv = np.load(stem_fn[-1])

    stem1['final_bn'] = correct_bn(final_bn)
    stem1['path1_conv'] = path1_conv
    stem1['path2_conv'] = path2_conv

    return stem1

def load_normal_call(index):
    print("Loading Normal NASNet Cell #%d" % (index))
    cell_fn = normal_fns[index]
    cell_beginning_weights = np.load(cell_fn[0])
    cell_beginning_bn = [np.load(fn) for fn in cell_fn[1:5]]

    cell = {'begin_W': cell_beginning_weights,
             'begin_bn': correct_bn(cell_beginning_bn)}

    for i in range(5):
        left_path = 'weights/cell_%d_comb_iter_%d_left_*.npy' % (index, i)
        right_path = 'weights/cell_%d_comb_iter_%d_right_*.npy' % (index, i)
        left_fns = (glob.glob(left_path))
        right_fns = (glob.glob(right_path))

        if len(left_fns) > 0:
            print("Loaded left at %d" % i)
            left_bn_1 = [np.load(fn) for fn in left_fns[:4]]
            left_bn_2 = [np.load(fn) for fn in left_fns[4:8]]
            left_d1 = np.load(left_fns[8])
            left_p1 = np.load(left_fns[9])
            left_d2 = np.load(left_fns[10])
            left_p2 = np.load(left_fns[11])

        if len(right_fns) > 0:
            print("Loaded right at %d" % i)
            right_bn_1 = [np.load(fn) for fn in right_fns[:4]]
            right_bn_2 = [np.load(fn) for fn in right_fns[4:8]]
            right_d1 = np.load(right_fns[8])
            right_p1 = np.load(right_fns[9])
            right_d2 = np.load(right_fns[10])
            right_p2 = np.load(right_fns[11])

        if len(left_fns) > 0:
            cell['left_%d' % i] = {'d1': left_d1, 'd2': left_d2, 'p1': left_p1, 'p2': left_p2,
                                    'bn1': correct_bn(left_bn_1), 'bn2': correct_bn(left_bn_2)}

        if len(right_fns) > 0:
            cell['right_%d' % i] = {'d1': right_d1, 'd2': right_d2, 'p1': right_p1, 'p2': right_p2,
                                     'bn1': correct_bn(right_bn_1), 'bn2': correct_bn(right_bn_2)}

    final_bn = (glob.glob('weights/cell_%d_final_path_bn_*.npy' % (index)))
    if len(final_bn) > 0:
        print("Loaded final path")
        final_bn = [np.load(fn) for fn in final_bn]
        conv_paths = (glob.glob('weights/cell_%d_path*_conv_weights.npy' % (index)))
        path1_conv = np.load(conv_paths[0])
        path2_conv = np.load(conv_paths[1])

        cell['final_bn'] = correct_bn(final_bn)
        cell['path1_conv'] = path1_conv
        cell['path2_conv'] = path2_conv

    prev_bn = (glob.glob('weights/cell_%d_prev_bn_*.npy' % (index)))
    if len(prev_bn) > 0:
        print("Loaded previous path")
        prev_bn = [np.load(fn) for fn in prev_bn]
        prev_conv = glob.glob('weights/cell_%d_prev_1x1_*.npy' % (index))[0]

        cell['prev_bn'] = correct_bn(prev_bn)
        cell['prev_conv'] = np.load(prev_conv)

    return cell

def load_reduction_call(index):
    print("Loading Reduction NASNet Cell #%d" % (index))
    cell_fn = reduction_cells[index]
    cell_beginning_weights = np.load(cell_fn[0])
    cell_beginning_bn = [np.load(fn) for fn in cell_fn[1:5]]

    cell = {'begin_W': cell_beginning_weights,
            'begin_bn': correct_bn(cell_beginning_bn)}

    for i in range(5):
        left_path = 'weights/reduction_cell_%d_comb_iter_%d_left_*.npy' % (index, i)
        right_path = 'weights/reduction_cell_%d_comb_iter_%d_right_*.npy' % (index, i)
        left_fns = (glob.glob(left_path))
        right_fns = (glob.glob(right_path))

        if len(left_fns) > 0:
            print("Loaded left at %d" % i)
            left_bn_1 = [np.load(fn) for fn in left_fns[:4]]
            left_bn_2 = [np.load(fn) for fn in left_fns[4:8]]
            left_d1 = np.load(left_fns[8])
            left_p1 = np.load(left_fns[9])
            left_d2 = np.load(left_fns[10])
            left_p2 = np.load(left_fns[11])

        if len(right_fns) > 0:
            print("Loaded right at %d" % i)
            right_bn_1 = [np.load(fn) for fn in right_fns[:4]]
            right_bn_2 = [np.load(fn) for fn in right_fns[4:8]]
            right_d1 = np.load(right_fns[8])
            right_p1 = np.load(right_fns[9])
            right_d2 = np.load(right_fns[10])
            right_p2 = np.load(right_fns[11])

        if len(left_fns) > 0:
            cell['left_%d' % i] = {'d1': left_d1, 'd2': left_d2, 'p1': left_p1, 'p2': left_p2,
                                    'bn1': correct_bn(left_bn_1), 'bn2': correct_bn(left_bn_2)}

        if len(right_fns) > 0:
            cell['right_%d' % i] = {'d1': right_d1, 'd2': right_d2, 'p1': right_p1, 'p2': right_p2,
                                     'bn1': correct_bn(right_bn_1), 'bn2': correct_bn(right_bn_2)}

    final_bn = (glob.glob('weights/reduction_cell_%d_final_path_bn_*.npy' % (index)))
    if len(final_bn) > 0:
        print("Loaded final")
        final_bn = [np.load(fn) for fn in final_bn]
        conv_paths = glob.glob('weights/reduction_cell_%d_path*_conv_weights.npy' % (index))
        print("Conv paths : ", conv_paths)
        path1_conv = np.load(conv_paths[0])
        path2_conv = np.load(conv_paths[1])

        cell['final_bn'] = correct_bn(final_bn)
        cell['path1_conv'] = path1_conv
        cell['path2_conv'] = path2_conv

    prev_bn = (glob.glob('weights/reduction_cell_%d_prev_bn_*.npy' % (index)))
    if len(prev_bn) > 0:
        print("Loaded previous")
        prev_bn = [np.load(fn) for fn in prev_bn]
        prev_conv = (glob.glob('weights/reduction_cell_%d_prev_1x1_*.npy' % (index))[0])

        cell['prev_bn'] = correct_bn(prev_bn)
        cell['prev_conv'] = np.load(prev_conv)

    return cell

def load_auxilary_branch():
    print("Loading auxilary branch")
    bn1 = [np.load(fn) for fn in aux_fns[:4]]
    bn2 = [np.load(fn) for fn in aux_fns[4:8]]
    conv_aux_2 = np.load(aux_fns[8])
    fc_bias = np.load(aux_fns[9])
    fc_weights = np.load(aux_fns[10])
    conv_aux_1 = np.load(aux_fns[11])

    fc_weights = fc_weights[:, 1:]
    fc_bias = fc_bias[1:]

    aux = {'conv1': conv_aux_1, 'conv2': conv_aux_2, 'fc': [fc_weights, fc_bias],
           'bn1': correct_bn(bn1), 'bn2': correct_bn(bn2)}
    return aux

def load_head():
    print("Loading Head")
    weights = np.load(final_dense_fns[1])
    bias = np.load(final_dense_fns[0])

    weights = weights[:, 1:]
    bias = bias[1:]

    return [weights, bias]

if __name__ == '__main__':
    pass