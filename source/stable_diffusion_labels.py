import torch
import source.stable_diffusion as stable_diffusion
import rp

class BaseLabel:
    def __init__(self, name:str, embedding:torch.Tensor, device=None):
        #Later on we might have more sophisticated embeddings, such as averaging multiple prompts
        #We also might have associated colors for visualization, or relations between labels
        
        if device is None:
            device=stable_diffusion._get_stable_diffusion_singleton().device

        self.name=name
        self.embedding=embedding.to(device)
        
    def get_sample_image(self):
        s = stable_diffusion._get_stable_diffusion_singleton()
        with torch.no_grad():
            output=s.embeddings_to_imgs(self.embedding)[0]
        assert rp.is_image(output)
        return output
            
    def __repr__(self):
        return '%s(name=%s)'%(type(self).__name__,self.name)
        
class SimpleLabel(BaseLabel):
    def __init__(self, name:str, device=None):
        s = stable_diffusion._get_stable_diffusion_singleton()
        super().__init__(name, s.get_text_embeddings(name).to(device), device=device)

class NegativeLabel(BaseLabel):
    def __init__(self, name:str, negative_prompt='', device=None):
        s = stable_diffusion._get_stable_diffusion_singleton()
        
        if '---' in name:
            #You can use '---' in a prompt to specify the negative part
            name,additional_negative_prompt=name.split('---',maxsplit=1)
            negative_prompt+=' '+additional_negative_prompt
            
        self.negative_prompt=negative_prompt
        old_uncond_text=s.uncond_text
        try:
            s.uncond_text=negative_prompt
            embedding = s.get_text_embeddings(name)
            super().__init__(name, embedding, device=device)
        finally:
            s.uncond_text=old_uncond_text
            
class MeanLabel(BaseLabel):
    #Test: rp.display_image(rp.horizontally_concatenated_images(MeanLabel('Dogcat','dog','cat').get_sample_image() for _ in range(1)))
    def __init__(self, name:str, *prompts, device=None):
        prompts=rp.detuple(prompts)
        super().__init__(name, get_mean_embedding(prompts), device=device)

def get_mean_embedding(prompts:list):        
    s=stable_diffusion._get_stable_diffusion_singleton()
        
    return torch.mean(
        torch.stack(
            [s.get_text_embeddings(prompt) for prompt in prompts]
        ),
        dim=0
    )