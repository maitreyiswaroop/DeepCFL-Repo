import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layer_sizes, input_size, output_size):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        self._build()

    def _build(self):
        if len(self.layer_sizes)==0:
            self.layers.append(nn.Linear(self.input_size, self.output_size, bias=False))
        else:
            self.layers.append(nn.Linear(self.input_size, self.layer_sizes[0]))
            for i in range(len(self.layer_sizes)-1):
                self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(self.layer_sizes[-1], self.output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

## Encoder-Decoder Network
class Encoder(nn.Module):
    def __init__(self, input_shape, conv_params=None, fc_layers=None, latent_dim=None, is_BYOL=False):
        super(Encoder, self).__init__()
        if not isinstance(input_shape, list):
            if isinstance(input_shape, tuple):
                input_shape = list(input_shape)
            else:
                input_shape = [input_shape]

        # Convolutional layers
        if conv_params is None or conv_params[0] is None:
            self.conv = nn.Identity()
            conv_output_size = torch.prod(torch.tensor(input_shape))
        else:
            conv_filters, conv_kernels, conv_strides, paddings = conv_params
            conv_layers = []
            prev_channels = input_shape[0]
            for i, (filters, kernel_size, stride, padding) in enumerate(zip(conv_filters, conv_kernels,conv_strides, paddings)):
                conv_layers.append(nn.Conv2d(in_channels=prev_channels,
                                            out_channels=filters,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding))
                conv_layers.append(nn.BatchNorm2d(filters))
                conv_layers.append(nn.ReLU())
                prev_channels = filters
            self.conv = nn.Sequential(*conv_layers)
            # Calculate the output size after passing through the convolutional layers
            conv_output_size = self._calculate_conv_output_size(input_shape, conv_filters, conv_kernels, paddings)

        prev_dim = conv_output_size
        fc_layer=[nn.Flatten()]
        for dim in fc_layers:
            fc_layer.append(nn.Linear(prev_dim,dim))
            fc_layer.append(nn.BatchNorm1d(dim))
            fc_layer.append(nn.ReLU())
            prev_dim=dim
        fc_layer.append(nn.Linear(prev_dim,latent_dim))
        if is_BYOL:
            fc_layer.append(nn.BatchNorm1d(latent_dim))
        self.fc = nn.Sequential(*fc_layer)

    def forward(self, x):
        x = self.conv(x)
        # x = nn.Flatten(x),
        x = self.fc(x)
        return x

    def _calculate_conv_output_size(self, input_shape, conv_filters, conv_kernels, paddings):
        x = torch.randn(2, *input_shape)
        with torch.no_grad():
            x = self.conv(x)
        return x.view(x.size(0), -1).shape[1]


class Decoder(nn.Module):
    def __init__(self, latent_dim, conv_params, fc_layers, output_shape,reshape):
        super(Decoder, self).__init__()
        self.reshape=tuple(reshape)
        self.output_shape=list(output_shape)
        if not isinstance(output_shape, list):
            if isinstance(output_shape, tuple):
                output_shape = list(output_shape)
            else:
                output_shape = [output_shape]

        prev_dim = latent_dim
        fc_layer=[]
        for i,dim in enumerate(fc_layers):
            fc_layer.append(nn.Linear(prev_dim,dim))
            fc_layer.append(nn.ReLU())
            prev_dim=dim
        fc_layer.append(nn.Unflatten(1, self.reshape)),
        self.fc = nn.Sequential(*fc_layer)

        # Calculating the input size of the transpose convolutional layers
        self.input_size = [torch.prod(torch.tensor(output_shape)).item(),]

        # Transpose convolutional layers
        if conv_params is None or conv_params[0] is None:
            self.conv = nn.Identity()
        else:
            conv_filters, conv_kernels, conv_strides, paddings, output_paddings=conv_params
            conv_layers = []
            prev_channels = self.reshape[0]#self.input_size[0]
            for i,(filters, kernel_size, stride, padding, output_padding) in enumerate(zip(conv_filters, conv_kernels, conv_strides, paddings, output_paddings)):
                conv_layers.append(nn.ConvTranspose2d(in_channels=prev_channels,
                                                    out_channels=filters,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    output_padding=output_padding))
                conv_layers.append(nn.BatchNorm2d(filters))
                if i<len(conv_filters)-1:
                    conv_layers.append(nn.ReLU())
                prev_channels = filters
            self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.fc(x)
        x = self.conv(x)
        x = x.view([x.shape[0]]+self.output_shape)
        return x

## VAE Class
class VAE(nn.Module):
    def __init__(self, input_shape, conv_filters_in, conv_kernels_in,
                 conv_strides_in, paddings_in,
                 conv_filters_out, conv_kernels_out,
                 conv_strides_out, paddings_out, output_paddings_out,
                 fc_layers_in, fc_layers_out,
                 latent_dim, output_shape,reshape,softmax=False):
        super(VAE, self).__init__()
        input_shape=list(input_shape)
        output_shape=list(output_shape)
        self.encoder = Encoder(input_shape, (conv_filters_in, conv_kernels_in, conv_strides_in,paddings_in) ,fc_layers_in, latent_dim*2)
        self.decoder = Decoder(latent_dim, (conv_filters_out, conv_kernels_out, conv_strides_out,paddings_out,output_paddings_out), fc_layers_out, output_shape,reshape)
        if softmax:
            self.softmax = nn.Softmax(dim=1)
        else:
            self.softmax = nn.Identity()
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu,logvar = self.encoder(x).chunk(2,dim=1)
        z = self.reparameterise(mu,logvar)
        z = self.softmax(z)
        recon_x = self.decoder(z)
        return recon_x, z, mu,logvar

## Combined

class Combined_VAE_Supervised(nn.Module):
    def __init__(self, x_l_shape, dim_x_h, dim_y_h,
                 encoder_specs, decoder_specs,
                 MLP_layers):
        super(Combined_VAE_Supervised, self).__init__()
        self.AE_x = VAE(input_shape=list(x_l_shape),
                        conv_filters_in=encoder_specs[0],
                        conv_kernels_in=encoder_specs[1],
                        conv_strides_in=encoder_specs[2],
                        paddings_in=encoder_specs[3],
                        conv_filters_out=decoder_specs[0],
                        conv_kernels_out=decoder_specs[1],
                        conv_strides_out=decoder_specs[2],
                        paddings_out=decoder_specs[3],
                        output_paddings_out=decoder_specs[4],
                        reshape=decoder_specs[5],
                        fc_layers_in=encoder_specs[4], fc_layers_out=decoder_specs[6],
                        latent_dim=dim_x_h, output_shape=list(x_l_shape))
        self.f_xy = MLP(MLP_layers, dim_x_h, dim_y_h)

    def forward(self, x_l):
        recon_x, x_h, mu_x, logvar_x = self.AE_x(x_l)
        y_h_prime = self.f_xy(x_h)
        return (recon_x, x_h, mu_x, logvar_x), y_h_prime

# Our proposed model 
class Combined_VAE(nn.Module):
    def __init__(self, x_l_shape,y_l_shape, dim_x_h, dim_y_h,
                 x_encoder_specs, x_decoder_specs,
                 y_encoder_specs, y_decoder_specs,
                 MLP_layers=None, softmax = False):
        super(Combined_VAE, self).__init__()
        self.AE_x = VAE(input_shape=x_l_shape,
                        conv_filters_in=x_encoder_specs[0],
                        conv_kernels_in=x_encoder_specs[1],
                        conv_strides_in=x_encoder_specs[2],
                        paddings_in=x_encoder_specs[3],
                        conv_filters_out=x_decoder_specs[0],
                        conv_kernels_out=x_decoder_specs[1],
                        conv_strides_out=x_decoder_specs[2],
                        paddings_out=x_decoder_specs[3],
                        output_paddings_out=x_decoder_specs[4],
                        reshape=x_decoder_specs[5],
                        fc_layers_in=x_encoder_specs[4], fc_layers_out=x_decoder_specs[6],
                        latent_dim=dim_x_h, output_shape=x_l_shape, softmax = softmax)
        self.AE_y = VAE(input_shape=y_l_shape,
                        conv_filters_in=y_encoder_specs[0],
                        conv_kernels_in=y_encoder_specs[1],
                        conv_strides_in=y_encoder_specs[2],
                        paddings_in=y_encoder_specs[3],
                        conv_filters_out=y_decoder_specs[0],
                        conv_kernels_out=y_decoder_specs[1],
                        conv_strides_out=y_decoder_specs[2],
                        paddings_out=y_decoder_specs[3],
                        output_paddings_out=y_decoder_specs[4],
                        reshape=y_decoder_specs[5],
                        fc_layers_in=y_encoder_specs[4], fc_layers_out=y_decoder_specs[6],
                        latent_dim=dim_y_h, output_shape=y_l_shape, softmax=softmax)
        if MLP_layers is not None:
            if MLP_layers==[]:
                self.f_xy = MLP(MLP_layers, dim_x_h, dim_y_h)
            else:
                print("Many one case with predefined weight tensor")
                linear_layer = nn.Linear(dim_x_h, dim_y_h, bias=False)
                linear_layer.weight = nn.Parameter(MLP_layers, requires_grad=False)
                self.f_xy = linear_layer
        else:
            self.f_xy = nn.Identity()

    def forward(self, x_l, y_l):
        recon_x, x_h, mu_x, logvar_x = self.AE_x(x_l)
        recon_y, y_h, mu_y, logvar_y = self.AE_y(y_l)
        y_h_prime = self.f_xy(x_h)
        return (recon_x, x_h, mu_x, logvar_x), (recon_y, y_h, mu_y, logvar_y), y_h_prime