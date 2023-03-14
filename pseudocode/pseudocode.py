#Use https://gist.github.com to embed them

import torch, itertools, stable_diffusion as sd

def score_distillation_loss(image, prompt):
    #First, convert the image into latent space
    image_embedding = sd.vqvae.embed(image)

    # This is the same loss proposed in DreamFusion
    # This function is a little oversimplified - read the paper
    timestep = random_int(0, max_diffusion_step)
    noise = sd.get_noise(timestep)
    noised_embedding = sd.add_noise(image_embedding, noise, timestep)

    with torch.no_grad():
        text_embedding = sd.clip.embed(prompt)    
        predicted_noise = sd.unet(noised_embedding, text_embedding, timestep)

    return (torch.abs(noise - predicted_noise)).sum()

class Image(nn.Module):
    def __init__(self, height=256, width=256):
        self.image = nn.Parameter(torch.random(3, height, width))

    def forward(self):
        return torch.sigmoid(self.image)


def make_flippy_illusion(prompt_a:str, prompt_b:str) -> Image:
    #prompt_a is for when we view the image normally
    #prompt_b is for when we view the image upside-down

    num_iterations = 10000 #This is a hyperparameter

    image = Image()

    optim = torch.optim.SGD(image.parameters())

    for _ in range(num_iterations):
        score_distillation_loss(image         , prompt_a).backward()
        score_distillation_loss(image.rot180(), prompt_b).backward()
        optim.step() ; optim.zero_grad()

    return image



def make_rotating_overlays(prompt_a:str, prompt_b:str, prompt_c:str, prompt_d:str) -> (Image, Image):
    #prompt_a, prompt_b, prompt_c and prompt_d are what we see when we overlay the top
    # image over the bottom image and rotate it by 0, 90, 180 and 270 degrees respectively
    #It then returns the bottom and top images
    #
    #We model the light filtering that happens when you overlay two transparencies
    # with a backlight as a pixel-wise multiplication operation.

    num_iterations = 10000 #This is a hyperparameter

    bottom_image = Image() # This image stays still
    top_image    = Image() # This image goes on top, and spins

    optim = torch.optim.SGD(itertools.chain(top_image.parameters(), bottom_image.parameters()))

    for _ in range(num_iterations):
        score_distillation_loss(bottom_image * top_image         , prompt_a).backward()
        score_distillation_loss(bottom_image * top_image.rot90 (), prompt_b).backward()
        score_distillation_loss(bottom_image * top_image.rot180(), prompt_c).backward()
        score_distillation_loss(bottom_image * top_image.rot270(), prompt_d).backward()
        optim.step() ; optim.zero_grad()

    return bottom_image, top_image


def make_hidden_character(prompt_a:str, prompt_b:str, prompt_c:str, prompt_d:str) -> (Image, Image, Image, Image):
    #prompt_a, prompt_b, prompt_c and prompt_d are the four seemingly innocent images
    #prompt_z is the hidden image you get when you overlay all the images on top of each other
    #
    #
    #Like with rotating overlays, we model the light filtering that happens when you overlay
    # two transparencies with a backlight as a pixel-wise multiplication operation.

    num_iterations = 10000 #This is a hyperparameter

    #Initialize the four images
    image_a = Image()
    image_b = Image()
    image_c = Image()
    image_d = Image()

    optim = torch.optim.SGD(
        itertools.chain(
            image_a.parameters(),
            image_b.parameters(),
            image_c.parameters(),
            image_d.parameters(),
        )
    )

    for _ in range(num_iterations):
        image_z = image_a * image_b * image_c * image_d #This is the hidden image!
        score_distillation_loss(image_a, prompt_a).backward()
        score_distillation_loss(image_b, prompt_b).backward()
        score_distillation_loss(image_c, prompt_c).backward()
        score_distillation_loss(image_d, prompt_d).backward()
        score_distillation_loss(image_z, prompt_z).backward()
        optim.step() ; optim.zero_grad()

    return image_a, image_b, image_c, image_d
