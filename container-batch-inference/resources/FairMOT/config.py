from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class Config(object):
  def __init__(self, load_model, frame_rate):
    # basic experiment setting
    self.task = 'mot'
    self.dataset = 'jde'
    self.exp_id = 'default'
    self.test = True
    self.pretrained = False

    # path to pretrained model
    self.load_model = load_model
    # resume an experiment. Reloaded the selfimizer parameter and 
    # set load_model to model_last.pth in the exp dir if load_model is empty.
    self.resume = True

    # system
    # -1 for CPU, use comma for multiple gpus
    self.gpus = '0'
    # dataloader threads. 0 for single-thread.
    self.num_workes = 0
    # disable when the input size is not fixed.
    self.not_cuda_benchmark = True
    self.seed = 317 # random seed for CornerNet

    # log
    # disable progress bar and print to screen.
    self.print_iter = 0
    # not display time during training.
    self.hide_data_time = True
    # save model to disk every 5 epochs.
    self.save_all = True
    # main metric to save best model
    self.metric = 'loss'
    # visualization threshold.
    self.vis_thresh = 0.5
   
    # model
    # model architecture. Currently tested resdcn_34 | resdcn_50 | resfpndcn_34 | dla_34 | hrnet_18
    self.arch = 'dla_34'
    # conv layer channels for output head 0 for no conv layer -1 for default setting: 
    # 256 for resnets and 256 for dla.
    self.head_conv = -1
    # output stride. Currently only supports 4.
    self.down_ratio = 4

    # input
    # input height and width. -1 for default from
    # dataset. Will be overriden by input_h | input_w
    self.input_res = -1
    # input height. -1 for default from dataset.
    self.input_h = -1
    # input width. -1 for default from dataset.
    self.input_w = -1

    self.frame_w = 1920
    self.frame_h = 1080
    self.inp_w = 1088
    self.inp_h = 608
    
    self.frame_rate = frame_rate
   
    # train
    # learning rate for batch size 12.
    self.lr = 1e-4
    # drop learning rate by 10.
    self.lr_step = '20'
    # total training epochs.
    self.num_epochs = 30
    # batch size
    self.batch_size = 12
    # batch size on the master gpu.
    self.master_batch_size = -1
    # default: #samples / batch_size.
    self.num_iters = -1
    # number of epochs to run validation.
    self.val_intervals = 5
    # include validation in training and test on test set
    self.trainval = True

    # test
    # max number of output objects.
    self.K = 500
    # not use parallal data pre-processing.
    self.not_prefetch_test = True
    # fix testing resolution or keep the original resolution
    self.fix_res = True
    # keep the original resolution during validation.
    self.keep_res = True
    
    # confidence thresh for tracking
    self.conf_thres = 0.4
    # confidence thresh for detection
    self.det_thres = 0.3
    # iou thresh for nms
    self.nms_thres = 0.4
    # track_buffer
    self.track_buffer = 2

    # filter out tiny boxes
    self.min_box_area = 300
    # path to the input video
    self.input_video = '../videos/MOT16-03.mp4'
    # video or text
    self.output_format = 'video'
    # expected output root path
    self.output_root = '../demos'

    # mot
    # load data from cfg
    self.data_cfg = '../src/lib/cfg/data.json'
    # 
    self.data_dir = '/home/ubuntu/mot/FairMOT'

    # loss
    # use mse loss or focal loss to train keypoint heatmaps.
    self.mse_loss = True
    # regression loss: sl1 | l1 | l2
    self.reg_loss = 'l1'
    # loss weight for keypoint heatmaps.
    self.hm_weight = 1
    # loss weight for keypoint local offsets.
    self.off_weight = 1
    # loss weight for bounding box size.
    self.wh_weight = 0.1
    # reid loss: ce | triplet
    self.id_loss = 'ce'
    # loss weight for id
    self.id_weight = 1
    # feature dim for reid
    # self.reid_dim = 128
    self.reid_dim = 32
    # regress left, top, right, bottom of bbox
    self.ltrb = True
    # L1(\hat(y) / y, 1) or L1(\hat(y), y)
    self.norm_wh = True
    # apply weighted regression near center or just apply regression on center point.
    self.dense_wh = True
    # category specific bounding box size.
    self.cat_spec_wh = True
    # not regress local offset.
    self.not_reg_offset = False

    self.init()

  def parse(self, args=''):
    #self.gpus='0'
    self.gpus_str = self.gpus
    self.gpus = [int(gpu) for gpu in self.gpus.split(',')]
    self.gpus = [i for i in range(len(self.gpus))] if self.gpus[0] >=0 else [-1]
    self.lr_step = [int(i) for i in self.lr_step.split(',')]

    self.fix_res = not self.keep_res
    print('Fix size testing.' if self.fix_res else 'Keep resolution testing.')
    self.reg_offset = not self.not_reg_offset

    if self.head_conv == -1: # init default head_conv
      self.head_conv = 256 if 'dla' in self.arch else 256
    self.pad = 31
    self.num_stacks = 1

    if self.trainval:
      self.val_intervals = 100000000

    if self.master_batch_size == -1:
      self.master_batch_size = self.batch_size // len(self.gpus)
    rest_batch_size = (self.batch_size - self.master_batch_size)
    self.chunk_sizes = [self.master_batch_size]
    for i in range(len(self.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(self.gpus) - 1)
      if i < rest_batch_size % (len(self.gpus) - 1):
        slave_chunk_size += 1
      self.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', self.chunk_sizes)

    self.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    self.exp_dir = os.path.join(self.root_dir, 'exp', self.task)
    self.save_dir = os.path.join(self.exp_dir, self.exp_id)
    self.debug_dir = os.path.join(self.save_dir, 'debug')
    print('The output will be saved to ', self.save_dir)
    
    if self.resume and self.load_model == '':
      model_path = self.save_dir[:-4] if self.save_dir.endswith('TEST') \
                  else self.save_dir
      self.load_model = os.path.join(model_path, 'model_last.pth')

  def update_dataset_info_and_set_heads(self, dataset):
    input_h, input_w = dataset.default_resolution
    self.mean, self.std = dataset.mean, dataset.std
    self.num_classes = dataset.num_classes

    # input_h(w): self.input_h overrides self.input_res overrides dataset default
    input_h = self.input_res if self.input_res > 0 else input_h
    input_w = self.input_res if self.input_res > 0 else input_w
    self.input_h = self.input_h if self.input_h > 0 else input_h
    self.input_w = self.input_w if self.input_w > 0 else input_w
    self.output_h = self.input_h // self.down_ratio
    self.output_w = self.input_w // self.down_ratio
    self.input_res = max(self.input_h, self.input_w)
    self.output_res = max(self.output_h, self.output_w)

    if self.task == 'mot':
      self.heads = {'hm': self.num_classes,
                   'wh': 2 if not self.ltrb else 4,
                   'id': self.reid_dim}
      if self.reg_offset:
        self.heads.update({'reg': 2})
      self.nID = dataset.nID
      self.img_size = (1088, 608)
      #self.img_size = (864, 480)
      #self.img_size = (576, 320)
      #self.img_size = (576, 320)
    else:
      assert 0, 'task not defined!'
    print('heads', self.heads)

  def init(self, args=''):
    default_dataset_info = {
      'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'jde', 'nID': 14455},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    self.parse()
    dataset = Struct(default_dataset_info[self.task])
    self.dataset = dataset.dataset
    self.update_dataset_info_and_set_heads(dataset)
