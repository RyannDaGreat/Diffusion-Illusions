# Copyright (c) 2023 Ryan Burgert
# 
# This code was originally from the paper TRITON: Neural Neural Textures make Sim2Real Consistent.
# Please see the project page at https://tritonpaper.github.io - it's quite interesting!
# (That project was also written by me, Ryan Burgert)
#
# Author: Ryan Burgert

import torch
import torch.nn as nn
import numpy as np
import einops
import rp
from PIL import Image

#This file contains three types of learnable images:
#    Raster: A simple RGB pixel grid
#    MLP: A per-pixel MLP that takes in XY and outputs RGB
#    Fourier: An MLP with fourier-feature inputs


##################################
######## HELPER FUNCTIONS ########
##################################

class GaussianFourierFeatureTransform(nn.Module):
    """
    Original authors: https://github.com/ndahlquist/pytorch-fourier-feature-networks
    
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, num_features*2, width, height].
    """

    def __init__(self, num_channels, num_features=256, scale=10):
        #It generates fourier components of Arandom frequencies, not all of them.
        #The frequencies are determined by a random normal distribution, multiplied by "scale"
        #So, when "scale" is higher, the fourier features will have higher frequencies    
        #In learnable_image_tutorial.ipynb, this translates to higher fidelity images.
        #In other words, 'scale' loosely refers to the X,Y scale of the images
        #With a high scale, you can learn detailed images with simple MLP's
        #If it's too high though, it won't really learn anything but high frequency noise
        
        super().__init__()

        self.num_channels = num_channels
        self.num_features = num_features
        
        #freqs are n-dimensional spatial frequencies, where n=num_channels
        self.freqs = nn.Parameter(torch.randn(num_channels, num_features) * scale, requires_grad=False)

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batch_size, num_channels, height, width = x.shape

        assert num_channels == self.num_channels,\
            "Expected input to have {} channels (got {} channels)".format(self.num_channels, num_channels)

        # Make shape compatible for matmul with freqs.
        # From [B, C, H, W] to [(B*H*W), C].
        x = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_channels)

        # [(B*H*W), C] x [C, F] = [(B*H*W), F]
        x = x @ self.freqs

        # From [(B*H*W), F] to [B, H, W, F]
        x = x.view(batch_size, height, width, self.num_features)
        # From [B, H, W, F] to [B, F, H, W 
        x = x.permute(0, 3, 1, 2)

        x = 2 * torch.pi * x
        
        output = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        
        assert output.shape==(batch_size, 2*self.num_features, height, width)
        
        return output       
    

def get_uv_grid(height:int, width:int, batch_size:int=1)->torch.Tensor:
    #Returns a torch cpu tensor of shape (batch_size,2,height,width)
    #Note: batch_size can probably be removed from this function after refactoring this file. It's always 1 in all usages.
    #The second dimension is (x,y) coordinates, which go from [0 to 1) from edge to edge
    #(In other words, it will include x=y=0, but instead of x=y=1 the other corner will be x=y=.999)
    #(this is so it doesn't wrap around the texture 360 degrees)
    assert height>0 and width>0 and batch_size>0,'All dimensions must be positive integers'
    
    y_coords = np.linspace(0, 1, height, endpoint=False)
    x_coords = np.linspace(0, 1, width , endpoint=False)
    
    uv_grid = np.stack(np.meshgrid(y_coords, x_coords), -1)
    uv_grid = torch.tensor(uv_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous()
    uv_grid = uv_grid.repeat(batch_size,1,1,1)
    
    assert tuple(uv_grid.shape)==(batch_size,2,height,width)
    
    return uv_grid


##################################
######## LEARNABLE IMAGES ########
##################################

class LearnableImage(nn.Module):
    def __init__(self,
                 height      :int,
                 width       :int,
                 num_channels:int):

        #This is an abstract class, and is meant to be subclassed before use
        #Upon calling forward(), retuns a tensor of shape (num_channels, height, width)

        super().__init__()
        
        self.height      =height      
        self.width       =width       
        self.num_channels=num_channels
    
    def as_numpy_image(self):
        image=self()
        image=rp.as_numpy_array(image)
        image=image.transpose(1,2,0)
        return image

    
class NoParamsDecoderWrapper(nn.Module):
    #Used in LearnableLatentImage to hide the decoder's params
    #This seems a bit hacky...
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(x)

    def parameters(self):
        return iter(())


class LearnableLatentImage(nn.Module):
    def __init__(self,
                 learnable_image: LearnableImage,
                 decoder        : nn.Module,
                 freeze_decoder : bool = True):

        #Take some representation of a latent space and use it to generate images
    
        super().__init__()
        self.learnable_image=learnable_image
        self.freeze_decoder=freeze_decoder

        if freeze_decoder:
            self.decoder=NoParamsDecoderWrapper(decoder)
        else:
            self.decoder=decoder

    def forward(self):
        return self.decoder(self.learnable_image())

class LearnableImageRasterSigmoided(LearnableImage):
    def __init__(self,
                 height      :int  ,
                 width       :int  ,
                 num_channels:int=3):
        
        super().__init__(height,width,num_channels)
        
        #An image paramterized by pixels

        self.image=nn.Parameter(torch.randn(num_channels,height,width))
        
    def forward(self):
        output = self.image.clone()
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return torch.sigmoid(output) #Can't have values over 1 or less than 0 for peekaboo
    
    
    
class LearnableImageRaster(LearnableImage):
    def __init__(self,
                 height      :int  ,
                 width       :int  ,
                 num_channels:int=3):
        
        super().__init__(height,width,num_channels)
        
        #An image paramterized by pixels

        self.image=nn.Parameter(torch.randn(num_channels,height,width))
        
    def forward(self):
        output = self.image.clone()
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return output
    
    
    
class LearnableImageMLP(LearnableImage):
    def __init__(self,
                 height      :int     , # Height of the learnable images
                 width       :int     , # Width of the learnable images
                 num_channels:int=3   , # Number of channels in the images
                 hidden_dim  :int=256 , # Number of dimensions per hidden layer of the MLP
                ):
        
        super().__init__(height,width,num_channels)
        
        self.hidden_dim  =hidden_dim
        
        # The following Tensor is NOT a parameter, and is not changed while optimizing this class
        self.uv_grid=nn.Parameter(get_uv_grid(height,width,batch_size=1), requires_grad=False)
        
        H=hidden_dim    # Number of hidden features. These 1x1 convolutions act as a per-pixel MLP
        C=num_channels  # Shorter variable names let us align the code better
        self.model = nn.Sequential(
                nn.Conv2d(2, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, C, kernel_size=1),
                nn.Sigmoid(),
            )
            
    def forward(self):
        output = self.model(self.uv_grid).squeeze(0)
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return output
    
    
class LearnableImageFourier(LearnableImage):
    def __init__(self,
                 height      :int=256 , # Height of the learnable images
                 width       :int=256 , # Width of the learnable images
                 num_channels:int=3   , # Number of channels in the images
                 hidden_dim  :int=256 , # Number of dimensions per hidden layer of the MLP
                 num_features:int=128 , # Number of fourier features per coordinate
                 scale       :int=10  , # Magnitude of the initial feature noise
                ):
        #TODO: Make resolution changeable mid-training. Right now it's saved into the state...and it cant be changed mid-training.
        #An image paramterized by a fourier features fed into an MLP
        #The possible output range of these images is between 0 and 1
        #In other words, no pixel will ever have a value <0 or >1
        
        super().__init__(height,width,num_channels)
        
        self.hidden_dim  =hidden_dim
        self.num_features=num_features
        self.scale       =scale
        
        # The following objects do NOT have parameters, and are not changed while optimizing this class
        self.uv_grid=nn.Parameter(get_uv_grid(height,width,batch_size=1), requires_grad=False)
        self.feature_extractor=GaussianFourierFeatureTransform(2, num_features, scale)
        self.features=nn.Parameter(self.feature_extractor(self.uv_grid), requires_grad=False) # pre-compute this if we're regressing on images
        
        H=hidden_dim # Number of hidden features. These 1x1 convolutions act as a per-pixel MLP
        C=num_channels  # Shorter variable names let us align the code better
        M=2*num_features
        self.model = nn.Sequential(
                nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, C, kernel_size=1),
                nn.Sigmoid(),
            )
    
    def get_features(self, condition=None):
        #TODO: Don't keep this! Condition should be CONATENATED! Not replacing features...this is just for testing...

        features = self.features

        assert features.shape == (1, 2*self.num_features, self.height, self.width), features.shape

        if condition is not None:
            #Replace the first n features with the condition, where n = len(condition)
            assert isinstance(condition, torch.Tensor)
            assert condition.device == self.features.device
            assert len(condition.shape)==1, 'Condition should be a vector'
            assert len(condition)<=2*self.num_features
            features = features.clone()
            features = einops.rearrange(features, 'B C H W -> B H W C')
            features[..., :len(condition)] = condition
            features = einops.rearrange(features, 'B H W C -> B C H W')

        assert features.shape == (1, 2*self.num_features, self.height, self.width)

        return features
        

    def forward(self, condition=None):
        # Return all the images we've learned

        features = self.get_features(condition)

        output = self.model(features).squeeze(0)
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return output
    
    #def project(self,uv_maps):
    #    #TODO: Check if this function works well...
    #    #Right now consider it untested
    #    assert len(uv_maps.shape)==(4), 'uv_maps should be BCHW'
    #    assert uv_maps.shape[1]==2, 'Should have two channels: u,v'
    #    return self.model(self.feature_extractor(uv_maps))

class LearnableImageFourierFromPNG(LearnableImageFourier):
    def __init__(self,
                 image_path   :str,
                 height       :int=256,
                 width        :int=256,
                 num_channels :int=3,
                 hidden_dim   :int=256,
                 num_features :int=128,
                 scale        :int=10,
                 num_iters    :int=1000,
                 lr           :float=1e-3,
                 device       :str='cuda:0'):
        """
        Initializes the Fourier feature network and optimizes it to match the given image.

        Parameters:
            image_path   : Path to the PNG image.
            height       : Height of the image.
            width        : Width of the image.
            num_channels : Number of color channels.
            hidden_dim   : Number of dimensions per hidden layer of the MLP.
            num_features : Number of Fourier features per coordinate.
            scale        : Scale of the Fourier features.
            num_iters    : Number of iterations to optimize the network.
            lr           : Learning rate for the optimizer.
            device       : Device to run the computations on ('cuda' or 'cpu').
        """
        super().__init__(height, width, num_channels, hidden_dim, num_features, scale)
        self.device = device
        self.to(self.device)

        # Load and preprocess the target image
        self.target_image = self.load_and_preprocess_image(image_path, height, width).to(self.device)

        # Optimize the network to match the target image
        self.optimize_to_match_image(num_iters, lr)

    def load_and_preprocess_image(self, image_path, height, width):
        # Load the image using PIL
        image = Image.open(image_path).convert('RGB')

        # This is to handle the case where the input image is an output
        # image from a previous training session. The mid-training image 
        # previews are a single PNG with both images side-by-side
        # 
        image = image.crop((0, 0, width, height))

        # Resize the image if necessary
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)

        # Convert to np array and normalize to [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0

        # Convert to torch tensor and rearrange dimensions to [C, H, W]
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        return image_tensor

    def optimize_to_match_image(self, num_iters, lr):
        # Set up the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Define the loss function
        criterion = nn.MSELoss()

        # Training loop
        for iter_num in range(1, num_iters + 1):
            optimizer.zero_grad()
            output = self.forward()  # Output shape: (C, H, W)
            loss = criterion(output, self.target_image)
            loss.backward()
            optimizer.step()

            # Optionally print the loss every 100 iterations
            if iter_num % 100 == 0 or iter_num == num_iters:
                print(f"Optimization Iteration {iter_num}/{num_iters}, Loss: {loss.item():.6f}")

    def forward(self, condition=None):
        # Return the generated image from the network
        features = self.get_features(condition)
        output = self.model(features).squeeze(0)
        return output

###############################
######## TEXTURE PACKS ########
###############################
    

class LearnableTexturePack(nn.Module):
    def __init__(self,
                 height      :int   ,
                 width       :int   ,
                 num_channels:int   ,
                 num_textures:int   ,
                 get_learnable_image):
        
        #This is an abstract class, and is meant to be subclassed before use
        #TODO: Inherit from some list class, such as nn.ModuleList. That way we can access learnable_images by indexing them from self...

        super().__init__()
        
        self.height      =height
        self.width       =width
        self.num_channels=num_channels
        self.num_textures=num_textures
        
        assert callable(get_learnable_image)
        
        learnable_images=[get_learnable_image() for _ in range(num_textures)]
        learnable_images=nn.ModuleList(learnable_images)
        self.learnable_images=learnable_images
    
    def as_numpy_images(self):
        return [x.as_numpy_image() for x in self.learnable_images]
       
    def forward(self):
        #Returns a tensor of size (NT, NC, H, W)
        #Where NT=self.num_textures, NC=self.num_channels, H=self.height, W=self.width
        
        output = torch.stack(tuple(x() for x in self.learnable_images))
        assert output.shape==(self.num_textures, self.num_channels, self.height, self.width), str("WTF? "+str(output.shape)+" IS NOT "+str((self.num_textures, self.num_channels, self.height, self.width)))
        
        return output

    def __len__(self):
        #Returns the number of textures in the texture pack
        return len(self.learnable_images)

    
class LearnableTexturePackRaster(LearnableTexturePack):
    def __init__(self,
                 height      :int=256,
                 width       :int=256,
                 num_channels:int=  3,
                 num_textures:int=  1):
        
        get_learnable_image = lambda: LearnableImageRaster(height      ,
                                                           width       ,
                                                           num_channels)
        
        super().__init__(height             ,
                         width              ,
                         num_channels       ,
                         num_textures       ,
                         get_learnable_image)
        
        
class LearnableTexturePackMLP(LearnableTexturePack):
    def __init__(self,
                 height      :int=256 , 
                 width       :int=256 ,
                 num_channels:int=3   ,
                 hidden_dim  :int=256 ,
                 num_textures:int=1   ):
        
        get_learnable_image = lambda: LearnableImageMLP(height      ,
                                                        width       ,
                                                        num_channels,
                                                        hidden_dim  )
        
        super().__init__(height             ,
                         width              ,
                         num_channels       ,
                         num_textures       ,
                         get_learnable_image)
        
        self.hidden_dim  =hidden_dim
        
        
class LearnableTexturePackFourier(LearnableTexturePack):
    def __init__(self,
                 height      :int=256 , 
                 width       :int=256 ,
                 num_channels:int=3   ,
                 hidden_dim  :int=256 ,
                 num_features:int=128 ,
                 scale       :int=10  ,
                 num_textures:int=1   ):
        
        get_learnable_image = lambda: LearnableImageFourier(height      ,
                                                            width       ,
                                                            num_channels,
                                                            hidden_dim  ,
                                                            num_features,
                                                            scale       )
        
        super().__init__(height             ,
                         width              ,
                         num_channels       ,
                         num_textures       ,
                         get_learnable_image)
        
        self.hidden_dim  =hidden_dim  
        self.num_features=num_features
        self.scale       =scale       


class LearnableImageRasterBilateral(LearnableImageRaster):
    def __init__(self, bilateral_blur, num_channels:int=3):
        _,height,width=bilateral_blur.image.shape
        super().__init__(height,width,num_channels)
        self.bilateral_blur=bilateral_blur
    
    def forward(self):
        output=self.image.clone()
        output=self.bilateral_blur(output)
        output=torch.sigmoid(output)
        return output
    
class LearnableImageFourierBilateral(LearnableImageFourier):
    def __init__(self, bilateral_blur, num_channels:int=3,                  
                 hidden_dim  :int=256 ,
                 num_features:int=128 ,
                 scale       :int=10  ,):
        _,height,width=bilateral_blur.image.shape
        super().__init__(height,width,num_channels,hidden_dim=hidden_dim,num_features=num_features,scale=scale)
        
        H=self.hidden_dim # Number of hidden features. These 1x1 convolutions act as a per-pixel MLP
        C=self.num_channels  # Shorter variable names let us align the code better
        M=2*self.num_features
        self.model = nn.Sequential(
                nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, C, kernel_size=1),
                # nn.Sigmoid(),
            )
        
        self.bilateral_blur=bilateral_blur
    

    def forward(self, condition=None):
        # Return all the images we've learned

        features = self.get_features(condition)

        output = self.model(features).squeeze(0)
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        output = self.bilateral_blur(output)
        output = torch.sigmoid(output)
        
        assert output.shape==(self.num_channels, self.height, self.width)
        
        return output
    
    

        
class LearnableAlphasFourier(LearnableImage):
    
    #Derived from LearnableImageFourier
    
    def __init__(self,
                 height      :int=256 , # Height of the learnable images
                 width       :int=256 , # Width of the learnable images
                 num_channels:int=3   , # Number of channels in the images
                 hidden_dim  :int=256 , # Number of dimensions per hidden layer of the MLP
                 num_features:int=128 , # Number of fourier features per coordinate
                 scale       :int=10  , # Magnitude of the initial feature noise
                ):
        #TODO: Make resolution changeable mid-training. Right now it's saved into the state...and it cant be changed mid-training.
        #An image paramterized by a fourier features fed into an MLP
        #The possible output range of these images is between 0 and 1
        #In other words, no pixel will ever have a value <0 or >1
        
        super().__init__(height,width,num_channels)
        
        self.hidden_dim  =hidden_dim
        self.num_features=num_features
        self.scale       =scale
        
        # The following objects do NOT have parameters, and are not changed while optimizing this class
        self.uv_grid=nn.Parameter(get_uv_grid(height,width,batch_size=1), requires_grad=False)
        self.feature_extractor=GaussianFourierFeatureTransform(2, num_features, scale)
        self.features=nn.Parameter(self.feature_extractor(self.uv_grid), requires_grad=False) # pre-compute this if we're regressing on images
        
        H=hidden_dim # Number of hidden features. These 1x1 convolutions act as a per-pixel MLP
        C=num_channels  # Shorter variable names let us align the code better
        M=2*num_features
        self.model = nn.Sequential(
                nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
                # nn.Sigmoid(),
                # lambda x:nn.Softmax(dim=1)(x) * nn.Sigmoid()(x),
            )
        
        self.gate_net    =nn.Conv2d(H, 1, kernel_size=1) #Chooses how likely it is to be any mask at all
        self.selector_net=nn.Conv2d(H, C, kernel_size=1) #Chooses which mask class it belongs to
    
    def get_features(self, condition=None):
        #TODO: Don't keep this! Condition should be CONATENATED! Not replacing features...this is just for testing...

        features = self.features

        assert features.shape == (1, 2*self.num_features, self.height, self.width), features.shape

        if condition is not None:
            #Replace the first n features with the condition, where n = len(condition)
            assert isinstance(condition, torch.Tensor)
            assert condition.device == self.features.device
            assert len(condition.shape)==1, 'Condition should be a vector'
            assert len(condition)<=2*self.num_features
            features = features.clone()
            features = einops.rearrange(features, 'B C H W -> B H W C')
            features[..., :len(condition)] = condition
            features = einops.rearrange(features, 'B H W C -> B C H W')

        assert features.shape == (1, 2*self.num_features, self.height, self.width)

        return features
        

    def forward(self, condition=None):
        # Return all the images we've learned

        features = self.get_features(condition)

        from icecream import ic
        
        output = self.model(features)
        # ic(output.shape)
        
        gate = self.gate_net(output)
        select = self.selector_net(output)
        # ic(gate.shape,select.shape)
        
        gate = torch.sigmoid(gate)
        
        # select = nn.functional.softmax(select, dim=1)
        
        select = torch.sigmoid(select)
        select = select/select.sum(dim=1,keepdim=True) #Alternative to softmax; just make the sum = 1
        
        # ic(gate.shape,select.shape)
        
        output = gate * select
        # ic(output.shape)
        output = output.squeeze(0)
        # ic(output.shape)
        
        assert output.shape==(self.num_channels, self.height, self.width), '%s != %s'% (output.shape, (self.num_channels, self.height, self.width))
        
        return output
    
