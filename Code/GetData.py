import pickle
import random
import os
import numpy as np
import tensorflow as tf
from UtilsForTrainings import get_indexed_shuffled

def get_data_single(data_dir, step, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    data = open(os.path.normpath('/'.join([data_dir, 'DatasetSingleNote_NF_fixed.pickle'])), 'rb')
    Z = pickle.load(data)
    res = 240//step

    f = np.repeat(Z[0], 2*res).reshape(167, -1)[:, :250*res]
    v = np.repeat((Z[1]/127), 2*res).reshape(167, -1)[:, :250*res]

    ind = np.zeros((v.shape[0], v.shape[1]))
    for j in range(v.shape[0]):
        for i in range(v.shape[1]):
            #if v[j, i] != 0:
            ind[j, i] = i

    y = Z[2] #harm
    f0 = np.array(Z[4], dtype=np.float32)
    f0 = (np.repeat(f0, 250).reshape(167, 250))
    f0 = np.repeat(f0, res).reshape(167, -1)
    del Z

    amps = []
    for j in range(y.shape[0]):
        _amp = []
        for i in range(y.shape[1] // step):
            _amp.append(tf.sqrt(tf.reduce_mean(tf.square(tf.constant(y[j, step * i:step * (i + 1)])))))
        amps.append(_amp)
    amps = np.array(amps)[:, :240*res]


    f_t = f[99:105]
    v_t = v[99:105]
    ind_t = ind[99:105]
    y_t = y[99:105]
    f0_t = f0[99:105]
    amps_t = amps[99:105]

    f = np.concatenate((f[0:99], f[105:]), axis=0)
    v = np.concatenate((v[0:99], v[105:]), axis=0)
    ind = np.concatenate((ind[0:99], ind[105:]))
    y = np.concatenate((y[0:99], y[105:]))
    f0 = np.concatenate((f0[0:99], f0[105:]))
    amps = np.concatenate((amps[0:99], amps[105:]))

    indxs = get_indexed_shuffled(ind)
    for indx_batch in indxs:
        f = f[indx_batch]
        v = v[indx_batch]
        ind = ind[indx_batch]
        y = y[indx_batch]
        f0 = f0[indx_batch]
        amps = amps[indx_batch]

    f_v = f[-16:]
    v_v = v[-16:]
    ind_v = ind[-16:]
    y_v = y[-16:]
    f0_v = f0[-16:]
    amps_v = amps[-16:]

    f = f[:-16]
    v = v[:-16]
    ind = ind[:-16]
    y = y[:-16]
    f0 = f0[:-16]
    amps = amps[:-16]

    f = f[:, :240*res].reshape(1, 1, -1).T
    v = v[:, :240*res].reshape(1, 1, -1).T
    ind = ind[:, :240*res].reshape(1, 1, -1).T
    v = np.concatenate([v, ind/np.max(ind)], axis=2)
    f0 = f0[:, :240*res].reshape(1, 1, -1).T
    y = y[:, :step*240*res].reshape(-1, step)
    amps = amps[:, :step*240*res].reshape(-1)

    f_v = f_v[:, :240 * res].reshape(1, 1, -1).T
    v_v = v_v[:, :240 * res].reshape(1, 1, -1).T
    ind_v = ind_v[:, :240 * res].reshape(1, 1, -1).T
    v_v = np.concatenate([v_v, ind_v / np.max(ind_v)], axis=2)
    f0_v = f0_v[:, :240 * res].reshape(1, 1, -1).T
    y_v = y_v[:, :step * 240 * res].reshape(-1, step)
    amps_v = amps_v[:, :step * 240 * res].reshape(-1)

    f_t = f_t[:, :240*res].reshape(1, 1, -1).T
    v_t = v_t[:, :240*res].reshape(1, 1, -1).T
    ind_t = ind_t[:, :240*res].reshape(1, 1, -1).T
    v_t = np.concatenate([v_t, ind_t/np.max(ind_t)], axis=2)
    f0_t = f0_t[:, :240*res].reshape(1, 1, -1).T
    y_t = y_t[:, :step*240*res].reshape(-1, step)
    amps_t = amps_t[:, :step*240*res].reshape(-1)

    return f, v, y, f0, amps, ind, f_v, v_v, y_v, f0_v, amps_v, ind_v, f_t, v_t, y_t, f0_t, amps_t, ind_t




def get_data_single_2(data_dir, step, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    data = open(os.path.normpath('/'.join([data_dir, 'DatasetSingleNote_NF_fixed.pickle'])), 'rb')
    Z = pickle.load(data)
    res = 240//step

    f = np.repeat(Z[0], 2*res).reshape(167, -1)[:, :250*res]
    v = np.repeat((Z[1]/127), 2*res).reshape(167, -1)[:, :250*res]

    ind = np.zeros((v.shape[0], v.shape[1]))
    for j in range(v.shape[0]):
        for i in range(v.shape[1]):
            #if v[j, i] != 0:
            ind[j, i] = i

    y = Z[2] #harm
    f0 = np.array(Z[4], dtype=np.float32)
    f0 = (np.repeat(f0, 250).reshape(167, 250))
    f0 = np.repeat(f0, res).reshape(167, -1)
    del Z

    amps = []
    for j in range(y.shape[0]):
        _amp = []
        for i in range(y.shape[1] // step):
            _amp.append(tf.sqrt(tf.reduce_mean(tf.square(tf.constant(y[j, step * i:step * (i + 1)])))))
        amps.append(_amp)
    amps = np.array(amps)[:, :240*res]

    f_t = f[3::7]
    v_t = v[3::7]
    ind_t = ind[3::7]
    y_t = y[3::7]
    f0_t = f0[3::7]
    amps_t = amps[3::7]

    f_ = f[:3]
    v_ = v[:3]
    ind_ = ind[:3]
    y_ = y[:3]
    f0_ = f0[:3]
    amps_ = amps[:3]

    for i in range(0, 167, 7):
        f_ = np.concatenate((f_, f[i + 4:i + 10]), axis=0)
        v_ = np.concatenate((v_, v[i + 4:i + 10]), axis=0)
        ind_ = np.concatenate((ind_, ind[i + 4:i + 10]))
        y_ = np.concatenate((y_, y[i + 4:i + 10]))
        f0_ = np.concatenate((f0_, f0[i + 4:i + 10]))
        amps_ = np.concatenate((amps_, amps[i + 4:i + 10]))

    indxs = get_indexed_shuffled(ind_)
    for indx_batch in indxs:
        f = f_[indx_batch]
        v = v_[indx_batch]
        ind = ind_[indx_batch]
        y = y_[indx_batch]
        f0 = f0_[indx_batch]
        amps = amps_[indx_batch]

    f_v = f[-14:]
    v_v = v[-14:]
    ind_v = ind[-14:]
    y_v = y[-14:]
    f0_v = f0[-14:]
    amps_v = amps[-14:]

    f = f[:-14]
    v = v[:-14]
    ind = ind[:-14]
    y = y[:-14]
    f0 = f0[:-14]
    amps = amps[:-14]

    f = f[:, :240*res].reshape(1, 1, -1).T
    v = v[:, :240*res].reshape(1, 1, -1).T
    ind = ind[:, :240*res].reshape(1, 1, -1).T
    v = np.concatenate([v, ind/np.max(ind)], axis=2)
    f0 = f0[:, :240*res].reshape(1, 1, -1).T
    y = y[:, :step*240*res].reshape(-1, step)
    amps = amps[:, :step*240*res].reshape(-1)

    f_v = f_v[:, :240*res].reshape(1, 1, -1).T
    v_v = v_v[:, :240*res].reshape(1, 1, -1).T
    ind_v = ind_v[:, :240*res].reshape(1, 1, -1).T
    v_v = np.concatenate([v_v, ind_v/np.max(ind_v)], axis=2)
    f0_v = f0_v[:, :240*res].reshape(1, 1, -1).T
    y_v = y_v[:, :step*240*res].reshape(-1, step)
    amps_v = amps_v[:, :step*240*res].reshape(-1)

    f_t = f_t[:, :240*res].reshape(1, 1, -1).T
    v_t = v_t[:, :240*res].reshape(1, 1, -1).T
    ind_t = ind_t[:, :240*res].reshape(1, 1, -1).T
    v_t = np.concatenate([v_t, ind_t/np.max(ind_t)], axis=2)
    f0_t = f0_t[:, :240*res].reshape(1, 1, -1).T
    y_t = y_t[:, :step*240*res].reshape(-1, step)
    amps_t = amps_t[:, :step*240*res].reshape(-1)
    return f, v, y, f0, amps, ind, f_v, v_v, y_v, f0_v, amps_v, ind_v, f_t, v_t, y_t, f0_t, amps_t, ind_t
