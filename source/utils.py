import rp
import torch

class SingletonModel(rp.CachedInstances):
    """
    A base class to warn if instances of a derived class are created on multiple devices.
    It's effectively used to create singletons, except it doesn't force you to do that.
    
    This class is useful when working with models that allocate resources like VRAM on different devices.
    By inheriting from this class, you can ensure that a warning is displayed when multiple instances of
    the model are created on different devices, which might lead to resource waste (e.g., VRAM on GPUs).

    Example:
        class MyModel(SingletonModel):
            def __init__(self, device=None):
                super().__init__(device)
        
        model1 = MyModel(device="cpu")
        model2 = MyModel(device="gpu0")
        model3 = MyModel(device="gpu1")

        # The following warning messages will be displayed:
        # Warning: MyModel instances are on multiple devices (wasting VRAM): ['cpu', 'gpu0']
        # Warning: MyModel instances are on multiple devices (wasting VRAM): ['cpu', 'gpu0', 'gpu1']

        # Test that instances with the same device are the same object
        assert model1 is MyModel(device="cpu")
        assert model2 is MyModel(device="gpu0")
        assert model3 is MyModel(device="gpu1")
    """
    devices = []

    def __init__(self, device=None):
        if device is None:
            device = rp.select_torch_device(silent=True, prefer_used=True)

        device = torch.device(device)
        
        self.device = device
        self.devices.append(device)

        if len(self.devices) > 1:
            model_name = type(self).__name__
            rp.fansi_print(
                f"Warning: {model_name} instances are on multiple devices (wasting VRAM): {', '.join(map(str,self.devices))}",
                "yellow",
                "bold",
            )

    def __repr__(self):
        model_name = type(self).__name__
        return model_name+'(%s)'%str(self.device)


