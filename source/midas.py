import rp
from .utils import SingletonModel
import torch

assert not __name__ == 'midas', 'Please import this module as source.midas, not just midas. The later will conflict with a package from torch hub when initializing MIDAS().'

class MIDAS(SingletonModel):
    devices=[]    

    def __init__(self, device=None):
        super().__init__(device)

        model_type = "DPT_Large"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas = self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate_depth_map(self, image):
        """Estimates the depth map of an input image using the MiDaS model.

        Args:
            image (np.ndarray): The input image to estimate the depth map of.

        Returns:
            np.ndarray: The estimated depth map.
        """
        #TODO: When passed a torch Tensor, it should operate on that and keep gradients. Idk what self.transform(image) does though...problematic...gotta analyze it.
        image = rp.as_byte_image(image)
        image = rp.as_rgb_image(image)

        input_batch = self.transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = rp.as_numpy_array(prediction)
        assert rp.is_image(output)
        return output
    
    def run_live_demo(self,size=512):
        """Runs a live demo to display the estimated depth map of the input from a webcam."""
        while True:
            image = rp.load_image_from_webcam()
            image = rp.resize_image_to_fit(image,size,size)
            depth_map = self.estimate_depth_map(image)
            depth_map = rp.full_range(depth_map)
            rp.display_image(rp.horizontally_concatenated_images(image, depth_map))

    def save_depth_map(self, image_path):
        """Saves the estimated depth map of an input image.

        Args:
            image_path (str): The file path of the input image.

        Returns:
            str: The file path of the saved depth map.
        """
        assert rp.is_image_file(image_path)
        image = rp.load_image(image_path)
        depth_map = self.estimate_depth_map(image)
        depth_map = rp.full_range(depth_map)
        depth_map_name = rp.strip_file_extension(image_path) + '_depth.png'
        return rp.save_image(depth_map, depth_map_name)
    
    def get_rgbd_kernel_image(self, image, rgb_scale=.1, depth_scale=1):
        #Return an RGBA image, where alpha is depth.
        #The color ranges might be over the range [0,1]
        #It's meant to be used as a kernel for a bilateral filter
        assert rp.is_image(image)

        depth=self.estimate_depth_map(image)
        depth=depth/20
        depth=depth*depth_scale

        image=rp.as_float_image(image)
        image=rp.as_rgba_image(image)
        image=image*rgb_scale

        image[:,:,3]=depth
        
        assert rp.is_rgba_image(image)
        
        return image
    