import torch.nn as nn
import torch.nn.functional as F

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)  
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, ResidualDilatedConvLayer):
        for submodule in m.modules():
            if isinstance(submodule, nn.Conv1d):
                nn.init.kaiming_normal_(submodule.weight.data, mode='fan_out', 
                                      nonlinearity='leaky_relu', a=0.01)
                submodule.weight.data *= 0.5

class LayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm1d, self).__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        return x
    

class ResidualDilatedConvLayer(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=2):
        super(ResidualDilatedConvLayer, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.ln = LayerNorm1d(channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.ln(out)
        out = self.activation(out)
        return x + out


class EncoderDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=3, num_layers=3):
        super(EncoderDilatedConvBlock, self).__init__()
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = None
        
        self.layers = nn.ModuleList([
            ResidualDilatedConvLayer(out_channels, kernel_size, dilation)
            for _ in range(num_layers)
        ])
        self.pool = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        return x

class DecoderDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=3, num_layers=3):
        super(DecoderDilatedConvBlock, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, 
                                  stride=2, padding=1)
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = None
        
        self.layers = nn.ModuleList([
            ResidualDilatedConvLayer(out_channels, kernel_size, dilation)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.upsample(x)
        if self.proj is not None:
            x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

class AE(nn.Module):
    def __init__(self,  latent_dim=16, num_encoder_blocks=9, num_decoder_blocks=9, dilation=3):
        super(AE, self).__init__()        
        
        self.conv_up = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            LayerNorm1d(4),
            nn.GELU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            LayerNorm1d(8),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            LayerNorm1d(16),
            nn.GELU()
        )
        
        encoder_blocks = []
        for _ in range(num_encoder_blocks):
            encoder_blocks.append(EncoderDilatedConvBlock(16, 16, kernel_size=3, dilation=dilation))
        self.encoder_blocks = nn.Sequential(*encoder_blocks)
        self.to_latent = nn.Conv1d(16, latent_dim, kernel_size=1)
        
        decoder_blocks = []
        for i in range(num_decoder_blocks):
            in_ch = latent_dim if i == 0 else 16
            decoder_blocks.append(DecoderDilatedConvBlock(in_ch, 16, kernel_size=3, dilation=dilation))
        self.decoder_blocks = nn.Sequential(*decoder_blocks)
        
        self.conv_down = nn.Conv1d(16, 1, kernel_size=3, padding=1)
        self.apply(weights_init)
    
    def encode(self, x):
        x = self.conv_up(x)              
        x = self.encoder_blocks(x)         
        latent = self.to_latent(x)                
        return latent
    
    def decode(self, z):
        x = self.decoder_blocks(z)  
        x = self.conv_down(x)       
        return x   
    
    def loss_function(self, x):
        recon = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        return recon_loss
        

