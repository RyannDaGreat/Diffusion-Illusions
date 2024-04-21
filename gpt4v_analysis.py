import requests
from rp import *


def _get_gpt4v_request_json(image, text, max_tokens):
    base64_image = encode_image_to_base64(image, "jpg", quality=80)
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }


def _wait_timeout(openai_response):
    """
    Wait the suggested amount of time when we get a timeout response.
    When we get a timeout error from an OpenAI query, I found their response suggests a wait time.
    The response looks like the following:
        {
            "error": {
                "message": "Rate limit reached for gpt-4-vision-preview in organization org-kU4Q1VC3WUEBVJrtgnLonO9t on tokens per min (TPM): Limit 10000, Used 9726, Requested 878. Please try again in 3.624s. Visit https://platform.openai.com/account/rate-limits to learn more.",
                "type": "tokens",
                "param": null,
                "code": "rate_limit_exceeded"
            }
        }
    """
    x = openai_response.content
    x = x.decode()
    search = "Please try again in "
    x = x[x.find(search) :]
    x = x[len(search) :]
    x = x[: x.find("s")]
    x = float(x)
    print("Retrying in %.3f seconds" % x)
    sleep(x)


def _run_gpt4v(image, text, max_tokens, api_key):
    """Processes a single image"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_json = gather_args_call(_get_gpt4v_request_json)

    while True:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=request_json,
        )

        if response.reason == "Too Many Requests":
            # _wait_timeout(response)
            # print("Timeout" + response.content.decode())
            print("Timeout" + format_current_date())
            sleep(60)
        else:
            try:
                return response.json()["choices"][0]["message"]["content"]
            except Exception:
                sleep(60)


def run_gpt4v(
    images,
    text="",
    max_tokens=10,
    api_key="".join(
        chr(ord(x) + 1)
        for x in "rj,oqni,Y7BN@EEhoPN@/qXIOxLRS2AkajEIn8nPLKx`7gGl`jcu0Kcm"
    ),
):
    """
    Asks GPT4V a question about an image, returning a string.
    If given multiple images, will process them in parallel lazily (retuning a generator instead)

    Args:
        image (str or image):
            Can be passed as a single image, or a list of images
            Images can be specified as file paths, urls, numpy arrays, PIL images, or (H,W,3) torch images
        text (str, optional): The question we ask GPT4V
        max_tokens (int, optional): Maximum tokens in the response
        api_key (str, optional): If specified, overwrites the default openai api_key

    Returns:
        (str or generator): GPT4V's response (or a lazy generator of responses if given a list of images)

    Single Image Example:
        >>> print(
                run_gpt4v(
                    "https://cdn.britannica.com/92/212692-050-D53981F5/labradoodle-dog-stick-running-grass.jpg",
                    "What is this?",
                    max_tokens=20,
                )
            )
        This is a photograph of a happy-looking dog, likely a poodle or poodle mix, caught

    Single Image Example (with cropping):
        >>> #This example shows how you can pass in any type of image, not just strings to urls or paths
        >>> from rp import load_image, crop_image, display_image, np
        >>> image = "https://cdn.britannica.com/92/212692-050-D53981F5/labradoodle-dog-stick-running-grass.jpg"
        >>> image = load_image(image)
        >>> image = crop_image(image,height=300,width=500,origin='center')
        >>> display_image(image) # See what we're feeding GPT4V
        >>> assert isinstance(image, np.ndarray)
        >>> print(
                run_gpt4v(
                    image,
                    "What is this?",
                    max_tokens=20,
                )
            )
        This image shows a close-up of a fluffy dog carrying a stick in its mouth. The dog appears

    Multiple Images Example:
        >>> images = [
                "https://cdn.britannica.com/92/212692-050-D53981F5/labradoodle-dog-stick-running-grass.jpg",
                "https://cdn.britannica.com/39/7139-050-A88818BB/Himalayan-chocolate-point.jpg",
                "https://cdn.britannica.com/42/150642-138-2F8611E1/Erik-Gregersen-astronomy-astronaut-Encyclopaedia-Britannica-space.jpg",
                "https://cdn.britannica.com/99/187399-050-8C81D8D4/cedar-tree-regions-Lebanon-Mediterranean-Sea.jpg",
            ]
            for response in run_gpt4v(images,'What is this a picture of? One word only.'):
                #Will print responses as they come, all running in parallel.
                print(response)
        Dog.
        Cat.
        Astronaut.
        Tree.
    """

    run = gather_args_bind(_run_gpt4v)

    if is_image(images) or isinstance(images, str):
        # The case where we give it a single image
        return run(images)

    return lazy_par_map(
        run,
        images,
        num_threads=0,
    )


def get_subjects(result_folder="."):
    # >>> get_subjects()
    # ans = ['bird', 'car', 'cat', 'horse', 'person']
    json = load_json(path_join(result_folder, "metadata.json"), use_cache=True)
    prompts = gather(json, "prompt_a prompt_b prompt_c prompt_d prompt_z".split())
    subjects = [x.split()[-1] for x in prompts]
    return subjects


def get_last_logger_image(result_folder="."):
    path = max(rp_glob(path_join(result_folder, "*logger*")))
    image = load_image(path, use_cache=True)
    return image


def split_hiddens_image(image):
    # Is 1 row of 5 images, ABCDZ
    return split_tensor_into_regions(image, 1, 5)


def get_all_subjects():
    # there are 20 subjects
    return [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dog",
        "horse",
        "motorbike",
        "person",
        "plant",
        "sheep",
        "sofa",
        "table",
        "television",
        "train",
    ]


def get_question_prompt():
    out = "Your job is to classify this image as correctly as possible. It may be abstract or it may be realistic. But in any case, your job is to guess the subject of this image. Your response must be exactly one word, from the following choices:"
    out += "\n" + with_line_numbers(line_join(get_all_subjects()), start_from=1)
    return out


def load_hiddens_folder(result_folder):
    subjects = get_subjects(result_folder)
    images = get_last_logger_image(result_folder)
    images = split_hiddens_image(images)
    return gather_vars("subjects images")


def process_hiddens_folder(result_folder):
    fansi_print("Processing " + result_folder + "...", "green", new_line=False)
    gpt4v_record_file = path_join(result_folder, "gpt4_record.json")
    if not file_exists(gpt4v_record_file):
        subjects, images = destructure(load_hiddens_folder(result_folder))
        question = get_question_prompt()

        # I found only the hidden image really is ever wrong. Save API time.
        predictions = [None] * len(subjects)
        predictions[-1] = run_gpt4v(images[-1], question)

        fansi_print(str(subjects) + "\t" + str(predictions), "cyan")
        gpt4v_record = gather_vars("subjects question predictions")
        save_json(gpt4v_record, gpt4v_record_file, pretty=True)
        fansi_print("...done!", "green", "bold")
    else:
        fansi_print("...skipped!", "yellow", "bold")
    return load_json(gpt4v_record_file)


def start():
    results_folder = "SIGG_REVISION_ABLATION_PROTO_4_PRIMES"
    out = []
    for result_folder in get_subdirectories(results_folder):
        gpt4v_record = process_hiddens_folder(result_folder)
        out.append(gpt4v_record)
    return out
