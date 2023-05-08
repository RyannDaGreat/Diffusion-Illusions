#THENEWTING        WORKS!!!

def process_result(folder=".",**settings):
    #Expecting a folder like /home/ryan/.../Diffusion-Illusions/untracked/peekaboo_results_coco/2660-COCO_train2014_000000173538.jpg.bird on left.raster_bilateral/000


    #Coco (nocrop vs crop???)

    input_image_path = path_join(folder, "image.png")
    alpha_image_path = path_join(folder, "alphas/0.png")
    input_image = load_image(input_image_path, use_cache=True)
    alpha_image = load_image(alpha_image_path, use_cache=True)
    
    input_image=cv_resize_image(input_image,get_image_dimensions(alpha_image))

    params = path_join(folder, "params.json")
    params = load_json(params)
    from easydict import EasyDict

    params = EasyDict(params)

    fname = params.p_name.split(".")[0] + ".png"
    label_image_path = path_join(label_folder, fname)
    label_image = load_image(label_image_path, use_cache=True)
    
    if DO_CROP:
        crop_filename = crop_file_prefix
        crop_filename += fname.replace("png", "jpg.txt")
        crop_filename = path_join(dataset_folder, crop_filename)
        if not file_exists(crop_filename):
            print(crop_filename)
            return #The dataset doesn't have it???
        x1, x2, y1, y2 = list(map(int, text_file_to_string(crop_filename).strip().split()))
        label_image = label_image[y1:y2, x1:x2]

    prompt = params.label[len("SimpleLabel(name=") : -1]
    
    if 1 or not DO_CROP:
        label_image = crop_image_to_square(label_image)
        label_image = cv_resize_image(label_image, get_image_dimensions(input_image))

    a, b, d = input_image, label_image, alpha_image
    c = blend_images(a, b, alpha=0.5)
    comparison_image = rp.tiled_images([a, b, c, d], length=4)
    comparison_image = labeled_image(comparison_image, prompt)
    
    mode=get_file_extension(get_parent_folder(folder)) #Like raster_bilateral etc

    return gather_vars(
        'folder',
        'prompt',
        'params',
        'label_image',
        'alpha_image',
        'input_image',
        'comparison_image',
        'fname',
        'mode',
    )


from easydict import EasyDict

# COCO
coco_nocrop_settings = EasyDict(
    results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_coco_nocrop",
    label_folder="/nfs/ws1/datasets/RefCOCO/label",
    dataset_folder="/nfs/ws1/datasets/RefCOCO",
    crop_file_prefix="",
    DO_CROP=False,
)

# VOC
voc_settings = EasyDict(
    results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_VOC",
    label_folder="/raid/datasets/pascal_voc/VOC2012/SegmentationClass",
    dataset_folder="/nfs/ws1/datasets/RefVOC/",
    crop_file_prefix="cropped-",
    DO_CROP=True,
)

settings=voc_settings


def quick_test():
    ans=random_batch(get_all_folders(settings.results_folder),20)
    ans=list_pop(get_subfolders(x) for x in ans)

    for x in ans:
        #x=get_subfolders(x)[0]
        x=process_result(x,**settings)
        if x:
            display_image_on_macmax(x.comparison_image)
            input(x.prompt+'  |||  '+x.mode)
            
