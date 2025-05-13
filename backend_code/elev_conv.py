import torch


class ElevationConv(torch.nn.Module):
    
    def __init__(self,
                 img_in_ch,
                 elev_in_ch,
                 out_channels, 
                 kernel_size = 3,
                 padding = 1,
                 bias = False,
                 padding_mode = 'replicate'):
        
        
        super(ElevationConv, self).__init__()
        
        self.img_in_ch = img_in_ch
        self.elev_in_ch = elev_in_ch
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        self.conv_layer_img = torch.nn.Conv2d(self.img_in_ch,
                                              self.out_channels, 
                                              kernel_size = self.kernel_size, 
                                              stride=1, 
                                              padding=self.padding, 
                                              padding_mode=self.padding_mode, 
                                              device = self.device)
        
        
        self.conv_layer_elev = torch.nn.Conv2d(self.elev_in_ch,
                                              self.out_channels, 
                                              kernel_size = self.kernel_size, 
                                              stride=1, 
                                              padding=self.padding, 
                                              padding_mode=self.padding_mode, 
                                              device = self.device)
        
        # self.batch_norm_elev = torch.nn.BatchNorm2d(self.out_channels)
        # self.activation_elev = torch.nn.ReLU()
        self.elev_sigmoid = torch.nn.Sigmoid()
        
        
        # self.conv_layer_blended = torch.nn.Conv2d(self.out_channels,
        #                                       self.out_channels, 
        #                                       kernel_size = self.kernel_size, 
        #                                       stride=1, 
        #                                       padding=self.padding, 
        #                                       padding_mode=self.padding_mode, 
        #                                       device = self.device)
        


    
    def forward(self, input_data, elevation_data):
        
        img_conv = self.conv_layer_img(input_data)
        # print("img_conv: ", img_conv.shape)
        
        elev_conv = self.conv_layer_elev(elevation_data)
        # elev_conv = self.batch_norm_elev(elev_conv)
        # elev_conv = self.activation_elev(elev_conv)
        elev_conv = self.elev_sigmoid(elev_conv)
        # print("elev_conv: ", elev_conv.shape)
        
        conv_outs = img_conv*elev_conv
        # conv_outs = img_conv
        # print("conv_outs1: ", conv_outs.shape)
        
        # conv_outs = self.conv_layer_blended(conv_outs)
        # print("conv_outs2: ", conv_outs.shape)
        
        return conv_outs, elev_conv