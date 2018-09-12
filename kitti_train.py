'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

'''
深度学习中的几个重要超参数:
    batchsize：每批数据量的大小。DL通常用SGD的优化算法进行训练，也就是一次（1 个iteration）一起训练batchsize个样本，计算它们的平均损失函数值，来更新参数。
    iteration：1个iteration即迭代一次，也就是用batchsize个样本训练一次。
    epoch：1个epoch指用训练集中的全部样本训练一次，此时相当于batchsize 等于训练集的样本数。
'''


'''
#把模型的权重值保存起来
'''
save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

'''
Data files，数据集文件
'''
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

'''
  Training parameters，训练参数的设置：
  nb_epoch表示迭代次数；
  batch_size表示XXX大小
  samples_per_epoch表示XXX；
  N_seq_val表示验证序列的个数
'''
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation


'''
  Model parameters，模型参数：
  input_shape中shape指的是张量，即表示从最外层向量逐步到达最底层向量的降维解包过程。
  图像在程序中表示一张彩色图片一般都分为（RGB三通道，输入图像高度，输入图像宽度），但是参数顺序有格式问题；
  image_data_format表示数据格式问题，channels_first为(通道个数，输入图像攻读，输入图像宽度)，channels_last为(输入图像攻读，输入图像宽度，通道个数)
  stack_sizes = R_stack_sizes是？？？
  A_filt_sizes是卷积层的大小？？？

  layer_loss_weights代表每一层损失的权重，0层是【1，0，0，0]，其他都是【1，0.1，0.1，0.1】
  nt是训练的时候使用了前nt张图片
'''
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0

# 利用以上Model parameters初始化prednet网络
prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)  ## 定义输入张量形状（batch_size,序列长，img_row,img_col,img_channels）
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)，计算各层A与Ahat误差？运行topology中Layer.__call__
# calculate weighted error by layer一个不训练有权重的dense全连接层，实际就是给各Layer的loss加权
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)
# 对batch中每个样本，展平成一维向量
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)

# 一个全连接层，为各时刻error加权重
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time

# keras中的模型主要包括model和weight两个部分:model可以通过json文件保存，保存权重可以通过保存（系数）
model = Model(inputs=inputs, outputs=final_errors)
# 自定义损失函数，参数为损失函数名字+优化器
model.compile(loss='mean_absolute_error', optimizer='adam')

# 根据训练的数据集以及batch size，生成每次epoch迭代的训练数据
train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)


# 回调函数：学习率调度器，以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）。如果epoch < 75,学习率为0.001，否则0.0001
lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    # 使用回调函数来观察训练过程中网络内部的状态和统计信息
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

# 与fit功能类似，利用Python的生成器，逐个生成数据的batch并进行训练，速度快
history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
