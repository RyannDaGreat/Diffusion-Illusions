from rp import *  # pip install rp

def f_initial(Ta, Tb, Tc, Td, Tz, Lz, backlight):
    """
    Closed-form initial estimate of A, B, C, D, Z minimizing weighted mean squared error.

    Parameters
    ----------
    Ta, Tb, Tc, Td : float
        Target values for A, B, C, D.
    Tz : float
        Target value for Z.
    Lz : float
        Weight for how much to prioritize Z reconstruction.
    backlight : float
        Multiplicative constant in the definition of Z = backlight * A * B * C * D.

    Returns
    -------
    list of float
        [A, B, C, D, Z] initial estimates.
    """
    p = Ta * Tb * Tc * Td
    epsilon = Tz - backlight * p
    v = [1 / Ta, 1 / Tb, 1 / Tc, 1 / Td]
    v_dot_v = sum(vi**2 for vi in v)
    scaling = (backlight * Lz * p * epsilon) / (
        1 + (backlight**2) * Lz * p**2 * v_dot_v
    )
    delta = [scaling * vi for vi in v]
    A = Ta + delta[0]
    B = Tb + delta[1]
    C = Tc + delta[2]
    D = Td + delta[3]
    Z = backlight * A * B * C * D
    return [A, B, C, D, Z]


def f(Ta, Tb, Tc, Td, Tz, Lz=2, backlight=2):
    """
    Refined estimate of A, B, C, D, Z by gradient steps starting from closed-form initialization.
    Math done with mathematica + chatGPT: https://chatgpt.com/share/680fbe46-239c-8006-89c7-87f32a381c5c
    
    Note: With a higher backlight value, you can get better accuracy for free!
    HOWEVER: There's a tradeoff: Higher backlight values don't model real-world overlays as well, as innacuracies in the printing process are exacerbated a lot
        A backlight value of 3 is the most you'd really want to use...backlight value of 2 is safe for real-world use
    
    NOTE: Lz is the Loss-coefficient for image Z. Basically, if it's higher - we prioritize the accuracy of Z more than the accuracy of A,B,C,D
    If Lz=0, then it returns exactly A=Ta,B=Tb,C=Tc,D=Td - not useful, resulting in no change. 

    Hidden overlay illusion:
    Given 5 target images, we want to solve for prime images A,B,C,D
    We define derived image Z = A * B * C * D * backlight (where backlight is the brightness of the light behind the overlays)
    This solves for A,B,C,D using least squares, such that:
        |  GIVEN:
        |  Ta Tb Tc Td Tz, Lz (Lz is a coefficient for how much we relatively care about the Z reconstruction)
        |  (Here, Ta for example is an atomic variable - it's not like T * a, it's just T_a shorthand)
        | 
        |  RELATIONSHIPS TO A,B,C,D,Z:
        |  A = Ta
        |  B = Tb
        |  C = Tc
        |  D = Td
        |  Z = 3 * A * B * C * D
        |
        |  GOAL:
        |  Solve for A, B, C, D, Z
        |  Minimize Mean Squared Error:
        |  (Ta - A)^2 +
        |  (Tb - B)^2 +
        |  (Tc - C)^2 +
        |  (Td - D)^2 +
        |  Lz * (Tz - Z)^2

    Parameters
    ----------
    Ta, Tb, Tc, Td : float
        Target values for A, B, C, D.
    Tz : float
        Target value for Z.
    Lz : float, optional
        Weight for Z reconstruction (default is 1).
    backlight : float, optional
        Multiplicative constant in Z = backlight * A * B * C * D (default is 3).

    Returns
    -------
    list of float
        [A, B, C, D, Z] refined estimates.
    """
    
    #We use an initial estimate to speed it up. We could start from just A=Ta, B=Tb, C=Tc, D=Td but
    #   that's slower than using a good first guess, provided with the below function
    A, B, C, D, _ = f_initial(Ta, Tb, Tc, Td, Tz, Lz, backlight=backlight)

    max_iter = 30
    step_size = .01
    for _ in range(max_iter):
        Z = backlight * A * B * C * D
        err = Tz - Z

        loss_grad_A = -2 * (Ta - A) + (-2 * backlight * Lz * err * B * C * D)
        loss_grad_B = -2 * (Tb - B) + (-2 * backlight * Lz * err * A * C * D)
        loss_grad_C = -2 * (Tc - C) + (-2 * backlight * Lz * err * A * B * D)
        loss_grad_D = -2 * (Td - D) + (-2 * backlight * Lz * err * A * B * C)

        A -= step_size * loss_grad_A
        B -= step_size * loss_grad_B
        C -= step_size * loss_grad_C
        D -= step_size * loss_grad_D
        
        #Just make sure there's no NaN pixels, or pixels outside the range [0,1]
        A = np.nan_to_num(np.clip(A,0,1))
        B = np.nan_to_num(np.clip(B,0,1))
        C = np.nan_to_num(np.clip(C,0,1))
        D = np.nan_to_num(np.clip(D,0,1))
        

    Z = backlight * A * B * C * D
    return [A, B, C, D, Z]

#Below we demo the above functions, displaying the target image on the left and the best-fit overlays on the right
images = [
    "https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/bernese-mountain-dog.jpg?crop=1.00xw:0.668xh;0,0.252xh&resize=640:*",
    "https://www.princeton.edu/sites/default/files/styles/1x_full_2x_half_crop/public/images/2022/02/KOA_Nassau_2697x1517.jpg?itok=Bg2K7j7J",
    "https://money.com/wp-content/uploads/2024/03/Best-Small-Dog-Breeds-Pomeranian.jpg?quality=60",
    "https://money.com/wp-content/uploads/2024/03/Best-Small-Dog-Breeds-Maltese.jpg?quality=60",
    "https://www.dogstrust.org.uk/images/800x600/assets/2025-03/toffee%202.jpg",
]
images = load_images(images, use_cache=True)
images = crop_images_to_square(images)
images = resize_images_to_min_size(images)
images = as_float_images(images)
images = as_rgb_images(images)
Ta, Tb, Tc, Td, Tz = images
A, B, C, D, Z = gather_args_call(f)

display_image(
    grid_concatenated_images(
        list_transpose(
            [
                labeled_images([Ta, Tb, Tc, Td, Tz], ["Ta", "Tb", "Tc", "Td", "Tz"]),
                labeled_images([A, B, C, D, Z], ["A", "B", "C", "D", "Z"]),
            ]
        )
    ),
    block=False,
)
