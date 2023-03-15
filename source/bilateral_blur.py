# Copyright (c) 2023 Ryan Burgert
#
# This code implements bilateral filtering in pytorch, with some extra functionality
# Namely, you can use the way the blur changes an RGB image (let's call it 'image') 
#    and apply those same changes to another image (let's call it 'alpha'). 
#    That way, we can ensure that where image has similar colors in RGB space, alpha
#    can have similar alpha values.
#
# Author: Ryan Burgert

import torch
import itertools
import einops
import rp
import icecream

__all__=['BilateralProxyBlur']


def nans_like(tensor: torch.Tensor) -> torch.Tensor:
    # EXAMPLE:
    #     >>> nans_like(torch.Tensor([1,2,3]))
    #    ans = tensor([nan, nan, nan])

    output = torch.ones_like(tensor)
    output[:] = torch.nan

    assert output.shape == tensor.shape
    assert output.dtype == tensor.dtype
    assert output.device == tensor.device

    return output


def shifted_image(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    # image is a torch image
    # the borders will be padded with nans
    # we use nan's intentionally, so we can keep track of which pixels are out-of-bounds throughout future operations
    # this lets us handle edge-cases by ignoring out-of-bounds locations instead of needing to pad the image
    
    assert len(image.shape) == 3  # (num_channels, height, width)
    _, height, width = image.shape

    if dx >= width or dx <= -width or dy >= height or dy <= -height:
        return nans_like(image)

    output = image

    if dx != 0:
        # Create |dx| columns of nan's
        columns = image[:, :, : abs(dx)]
        columns = nans_like(columns)

        if dx > 0:
            # Add the nan columns to the left of the image, and delete equally many columns from the right
            output = torch.concat((columns, output[:, :, :-dx]), dim=2)

        else:
            assert dx < 0
            # Add the nan columns to the right of the image, and delete equally many columns from the left
            output = torch.concat((output[:, :, -dx:], columns), dim=2)

    if dy != 0:
        # Create |dy| rows of nan's
        rows = image[:, : abs(dy), :]
        rows = nans_like(rows)

        if dy > 0:
            # Add the nan rows to the top of the image, and delete equally many rows from the bottom
            output = torch.concat((rows, output[:, :-dy, :]), dim=1)
        else:
            assert dy < 0
            # Add the nan rows to the bottom of the image, and delete equally many rows from the top
            output = torch.concat((output[:, -dy:, :], rows), dim=1)

    assert output.shape == image.shape

    return output


def test_shifted_image():
    import torch, icecream
    from rp import (
        as_float_image,
        as_numpy_image,
        as_torch_image,
        bordered_image_solid_color,
        cv_resize_image,
        display_image,
        get_image_dimensions,
        load_image,
    )

    image = "https://nationaltoday.com/wp-content/uploads/2020/02/doggy-date-night.jpg"
    image = load_image(image)
    image = as_float_image(image)
    image = cv_resize_image(image, (64, 64))
    height, width = get_image_dimensions(image)
    image = as_torch_image(image)
    for dx in [-64, -32, 0, 32, 64]:
        for dy in [-64, -32, 0, 32, 64]:
            im = shifted_image(image, dx, dy)
            im = torch.nan_to_num(im)
            im = as_numpy_image(im)
            im = bordered_image_solid_color(im)

            icecream.ic(dx, dy)
            display_image(im)


def get_weight_matrix(image:torch.Tensor, sigma:float, kernel_size:int, tolerance:float):
    #Return a 4d tensor corresponding to the weights needed to perform a bilateral blur
    
    assert len(image.shape)==3, 'image must be in CHW form'
    assert kernel_size%2, 'We only support kernels with an odd size (so they have a middle pixel), but kernel_size=%i'%kernel_size 
    
    C,H,W = image.shape
    K = kernel_size
    R = K//2 # The kernel radius, not including the center pixel
    
    device = image.device
    dtype = image.dtype
    
    kernel = rp.gaussian_kernel(size=kernel_size, sigma=sigma, dim=2) # A gaussian kernel matrix
    kernel = torch.tensor(kernel).type(dtype).to(device)
    assert kernel.shape==(K,K)

    shifts = torch.empty(K,K,C,H,W).type(dtype).to(device) #All values will be overwritten in upcoming for loop
    for u in range(K):
        for v in range(K):
            shifts[u,v] = shifted_image(image, u-R, v-R)
            
    color_deltas = shifts - image[None, None] # Color deltas between image pixels and neighbouring pixels
    assert color_deltas.shape==shifts.shape==(K,K,C,H,W)
    
    color_dists = (color_deltas**2).sum(dim=2).sqrt() # Euclidean color distances
    assert color_dists.shape==(K,K,H,W)
    
    color_weights = torch.exp(-1/2*(color_dists/tolerance)**2) # Apply a gaussian function, with standard deviation = tolerance
    assert color_weights.shape==color_dists.shape==(K,K,H,W)
    
    weights = kernel[:,:,None,None] * color_weights #Combine spatial weight with color weight
    weights = weights.nan_to_num() # Set all NaN values to 0, corresponding to out-of-bounds pixel locations
    weights = weights/weights.sum((0,1),keepdim=True) #Make sure all weights sum to 1, not including out-of-bounds locations
    assert weights.shape==color_weights.shape==(K,K,H,W)    
    
    return weights


def apply_weight_matrix(image:torch.Tensor, weights:torch.Tensor, iterations:int=1):
    assert len(image.shape)==3, 'image must be in CHW form'
    assert len(weights.shape)==4, 'weights must be in KKHW form'
    assert weights.shape[0]==weights.shape[1], 'weights must be in KKHW form'
    assert weights.device==image.device, 'weights %s and image %s must be on the same device'%(weights.device, image.device)
    assert weights.dtype==image.dtype, 'weights %s and image %s must have the same dtype'%(weights.dtype, image.dtype)
    
    if iterations>1:
        #This is faster than using a large kernel
        for _ in range(iterations):
            image = apply_weight_matrix(image, weights)
        return image
    
    device = image.device
    dtype = image.dtype
    
    C,H,W=image.shape
    K,K,_,_=weights.shape
    R=K//2
    assert weights.shape[2]==H and weights.shape[3]==W, 'the image HW dimensions be the same as the weights'
    
    weighted_colors = torch.empty(K,K,C,H,W).type(dtype).to(device) #All values will be overwritten in upcoming for loop
    for u in range(K):
        for v in range(K):
            shift = shifted_image(image, u-R, v-R)
            shift = shift.nan_to_num() #Replace nans with 0's
            assert shift.shape==image.shape==(C,H,W)
            
            weight=weights[u,v]
            assert weight.shape==(H,W)
            
            weighted_color = shift * weight[None,:,:]
            assert weighted_color.shape==shift.shape==image.shape==(C,H,W)
            
            weighted_colors[u,v] = weighted_color
            
    output_image = weighted_colors.sum((0,1))
    assert output_image.shape==image.shape==(C,H,W)
    
    return output_image  
    

class BilateralProxyBlur:
    def __init__(self, image:torch.Tensor,
                 *,
                 #These parameters are subjectively nice, and are arbitrary! Play around with them.
                 #Large kernel size is SLOOOW. You can get very similar results by using high iterations and small kernel_size
                 kernel_size:int = 5,
                 tolerance = .08,
                 sigma = 5,
                 iterations=10,
                ):
        self.weights = get_weight_matrix(image,sigma,kernel_size,tolerance)
        self.kernel_size=kernel_size
        self.tolerance=tolerance
        self.sigma=sigma
        self.iterations=iterations
        self.image=image
    def __call__(self, image):
        return apply_weight_matrix(image,self.weights,self.iterations)