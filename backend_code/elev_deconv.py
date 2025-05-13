import torch

class ElevationConvTranspose(torch.nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,  
                 kernel_size = 2,
                 stride = 2,
                 padding = 0,
                 dilation = 1,
                 bias = False,
                 padding_mode = 'zeros'):
        
        super(ElevationConvTranspose, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        self.trans_conv_layer_img = torch.nn.ConvTranspose2d(self.in_channels, 
                                                        self.out_channels, 
                                                        self.kernel_size, 
                                                        stride=2, 
                                                        padding=self.padding, 
                                                        output_padding=0, 
                                                        groups=1, 
                                                        bias=True, 
                                                        dilation=1, 
                                                        padding_mode=self.padding_mode, 
                                                        device=self.device)
        
        self.trans_conv_layer_elev = torch.nn.ConvTranspose2d(self.in_channels, 
                                                                self.out_channels, 
                                                                self.kernel_size, 
                                                                stride=2, 
                                                                padding=0, 
                                                                output_padding=0, 
                                                                groups=1, 
                                                                bias=True, 
                                                                dilation=1, 
                                                                padding_mode=self.padding_mode, 
                                                                device=self.device)
        
        # self.batch_norm_elev = torch.nn.BatchNorm2d(self.out_channels)
        # self.activation_elev = torch.nn.ReLU()
        self.elev_sigmoid = torch.nn.Sigmoid()
        
        
        # self.conv_layer_blended = torch.nn.Conv2d(self.out_channels,
        #                                           self.out_channels, 
        #                                           kernel_size = 3, 
        #                                           stride=1, 
        #                                           padding=1, 
        #                                           padding_mode=self.padding_mode, 
        #                                           device = self.device)
        

    
        
    
    def forward(self, input_data, elevation_data):
        
        img_trans_conv = self.trans_conv_layer_img(input_data)
        # print("img_trans_conv: ", img_trans_conv.shape)
        
        elev_trans_conv = self.trans_conv_layer_elev(elevation_data)
        # elev_trans_conv = self.batch_norm_elev(elev_trans_conv)
        # elev_trans_conv = self.activation_elev(elev_trans_conv)
        elev_trans_conv = self.elev_sigmoid(elev_trans_conv)
        # print("elev_trans_conv: ", elev_trans_conv.shape)
        
        trans_conv_outs = img_trans_conv*elev_trans_conv
        # trans_conv_outs = img_trans_conv
        # print("trans_conv_outs1: ", trans_conv_outs.shape)
        
        # trans_conv_outs = self.conv_layer_blended(trans_conv_outs)
        # print("trans_conv_outs2: ", trans_conv_outs.shape)
        
        return trans_conv_outs, elev_trans_conv