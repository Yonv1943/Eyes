import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm  # progress bar
import numpy.random as rd

import torch
import torch.nn as nn

"""
Source: https://github.com/piergiaj/pytorch-i3d
Modify: GitHub Yonv1943 Zen4 Jia1Hao2
2019-12-22
"""

GPU_id = sys.argv[0][-4]
Data_dir = '/mnt/sdb1/weit/datasets/UCF_Crimes'
Mod_dir = 'VideoDetect_{}'.format(GPU_id)
Inp_dim = 1024
Mid_dim = 192
Out_dim = 1  # 14-1
Mid_layers = 3
"""I3D PyTorch"""


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


"""Main"""


class I3D:
    def __init__(self):
        mod_path, num_classes = './weights_of_I3D/rgb_imagenet.pt', 400
        # mod_path, num_classes = './models/rgb_charades.pt', 157

        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(16)
        torch.manual_seed(1943)

        '''build model'''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i3d = InceptionI3d(num_classes=400, in_channels=3, final_endpoint='Logits').to(device)
        # i3d.replace_logits(num_classes) # for the pre-training model in charades dataset (indoor video)
        i3d.load_state_dict(torch.load(mod_path, map_location=lambda storage, loc: storage))
        i3d.eval()
        self.net = i3d
        self.device = device

    def extract_features(self, imgs, min_len=33, max_len=2 ** 8):
        imgs_len = imgs.shape[0]
        clip_len = (imgs_len - min_len) // max_len + 1
        outs = list()
        for i in range(clip_len):
            j = i * max_len
            k = j + max_len + 1
            if k <= imgs_len:
                pass
            elif k - j >= min_len:
                k -= (k - 1) % (min_len - 1)
            else:  # elif k - j < min_len:
                continue

            ary = imgs[j:k]
            ary = ary[np.newaxis]
            # ary.shape == (batch, time, high, width, channel)
            ary = ary.transpose((0, 4, 1, 2, 3))
            # ary.shape == (batch, channel, time, high, width)
            inp = torch.tensor(ary, dtype=torch.float32, device=self.device)
            inp /= 128.0
            inp -= 1.0

            out = self.net.extract_features(inp)
            out = out.cpu().data.numpy()[0, :, :, 0, 0]
            out = out.transpose((1, 0))

            outs.append(out)

        # print(imgs_len, clip_len, len(outs))
        outs = np.vstack(outs)
        return outs

    def run_test(self):
        imgs = np.ones((rd.randint(1234, 2345), 224, 224, 3))
        print('Inp.shape', imgs.shape)

        ary = imgs
        time_gap = 33
        max_batch = 8

        ary_len0 = ary.shape[0]
        ary = np.reshape(ary[:ary_len0 - (ary_len0 % time_gap)],
                         (-1, time_gap, 224, 224, 3))
        # ary.shape == (batch, time, high, width, channel)

        # out_list = list()
        ary_len1 = ary.shape[0]
        outs = np.empty((ary_len1, 400, 4), dtype=np.float32)
        for i in range(0, ary_len1, max_batch):
            inp = torch.tensor(ary[i:i + max_batch], dtype=torch.float32, device=self.device)
            inp /= 128
            inp -= 1.0
            # inp.shape == (batch, time, high, width, channel)
            inp = inp.permute(0, 4, 1, 2, 3)
            # inp.shape == (batch, channel, time, high, width)

            out = self.net(inp)
            out = out.cpu().data.numpy()
            # out_list.append(out)
            print(out.shape, i)
            outs[i:i + out.shape[0]] = out

        # outs = np.concatenate(out_list, axis=0)
        print(outs.shape)
        return outs

    def run_demo(self):
        mod_path, num_classes = './models/rgb_imagenet.pt', 400
        # mod_path, num_classes = './models/rgb_charades.pt', 157

        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(16)
        torch.manual_seed(1943)

        '''build model'''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.net
        # from pytorch_i3d import InceptionI3d
        # net = InceptionI3d(num_classes=400, in_channels=3).to(device)
        # net.replace_logits(num_classes) # for the pre-training model in charades dataset (indoor video)

        '''load model'''
        net.load_state_dict(torch.load(mod_path, map_location=lambda storage, loc: storage))

        b, c, t, h, w = 2, 3, 32 + 1, 224, 224
        # batch, channel, time, high, weight
        inp = torch.randn(b, c, t, h, w, dtype=torch.float32, device=device)
        print('inp.size():', inp.size(), '==', (b, c, t, 224, 224))
        out = net(inp)
        print('out.size():', out.size(), '==', (b, num_classes, (t - 1) // 8))
        out = net.extract_features(inp)
        print('out.size():', out.size(), '==', (b, num_classes, (t - 1) // 8))


def run_extract_features_from_video_by_i3d():
    #  convert video (mp4) to I3D feature (npy)

    data = DataUCFCrimes(Data_dir)
    mp4_dir_name = data.mp4_dir_name

    i3d = I3D()

    '''loop'''
    npy_dir_name = data.npy_dir_name
    npy_replace_name = '/{}/'.format(npy_dir_name)
    npy_dir = '{}/{}'.format(Data_dir, npy_dir_name)
    os.makedirs(npy_dir, exist_ok=True)

    mp4_walks = data.get_dir_walks(mp4_dir_name)
    for mp4_dir_path, dirs, files in mp4_walks[::-1]:
        npy_dir_path = mp4_dir_path.replace('/Videos/', npy_replace_name)
        os.makedirs(npy_dir_path, exist_ok=True)

        if len(os.listdir(mp4_dir_path)) == len(os.listdir(npy_dir_path)):
            print('finish :', mp4_dir_path)
            continue
        else:
            print('convert:', mp4_dir_path)

        for file in tqdm(files[::-1]):
            mp4_path = '{}/{}'.format(mp4_dir_path, file)
            npy_path = mp4_path.replace('/Videos/', npy_replace_name).replace('.mp4', '.npy')
            if os.path.exists(npy_path):
                continue

            imgs = data.mp42npy(mp4_path)

            ary = i3d.extract_features(imgs)
            if ary is not None:
                np.save(npy_path, ary)


if __name__ == '__main__':
    run_extract_features_from_video_by_i3d()
