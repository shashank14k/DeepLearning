import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, activation_func, **kwargs):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(**kwargs)
        self.batch_norm = nn.BatchNorm2d(kwargs['out_channels'])
        self.activation = activation_func

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class yolo(nn.Module):
    def __init__(self, config, img_channels, activation_func, window_size,
                 n_boxes, n_classes, fcn_units):
        super(yolo, self).__init__()
        # self.config = config
        self.img_channels = img_channels
        self.activation_func = activation_func
        self.fcn_inp_channels = None
        self.conv_layers = self.init_conv_layers(config)
        self.fcn_layers = self.init_fcn_layers(window_size, n_boxes, n_classes,
                                               fcn_units)

    def forward(self, x):
        return self.fcn_layers(torch.flatten(self.conv_layers(x),
                                             start_dim=1))  # start_dim =1 as dim=0 is n_batches

    def init_conv_layers(self, config):
        layers = []
        inp_channels = self.img_channels  # Initializing input channels
        for x in config:
            if x == 'MaxPool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(x) == tuple:
                params = {'in_channels': inp_channels, 'out_channels': x[0],
                          'kernel_size': x[1], 'stride': x[2], 'padding': x[3]}
                layers.append(conv_block(self.activation_func, **params))
                inp_channels = x[0]
            elif type(x) == list:
                num_reps = x[-1]
                for i in range(num_reps):
                    for layer in x[:-1]:
                        print(layer)
                        params = {'in_channels': inp_channels,
                                  'out_channels': layer[0],
                                  'kernel_size': layer[1], 'stride': layer[2],
                                  'padding': layer[3]}
                        layers.append(
                            conv_block(self.activation_func, **params))
                        inp_channels = layer[0]
            self.fcn_inp_channels = inp_channels
        return nn.Sequential(*layers)

        def init_fcn_layers(self, window_size, n_boxes, n_classes, fcn_units):
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.fcn_inp_channels * window_size * window_size,
                          fcn_units),
                self.activation_func,
                nn.Linear(fcn_units,
                          window_size * window_size * (n_classes + n_boxes * 5))
            )
