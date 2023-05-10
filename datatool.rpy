#GOOGOO
# 2023-05-09 20:05:32.154642
# 2023-05-08 20:34:10.800500
#THENEWTING        WORKS!!!

def process_result(
    folder=".", *,label_folder, dataset_folder, crop_file_prefix, DO_CROP, **_
):
    #Expecting a folder like /home/ryan/.../Diffusion-Illusions/untracked/peekaboo_results_coco/2660-COCO_train2014_000000173538.jpg.bird on left.raster_bilateral/000


    #Coco (nocrop vs crop???)

    input_image_path = path_join(folder, "image.png")
    alpha_image_path = path_join(folder, "alphas/0.png")
    input_image = load_image(input_image_path, use_cache=True)
    alpha_image = load_image(alpha_image_path, use_cache=True)
    
    alpha_image = as_float_image    (alpha_image)
    alpha_image = as_grayscale_image(alpha_image)
    input_image = as_float_image    (input_image)
    input_image = as_rgb_image      (input_image)

    input_image=cv_resize_image(input_image,get_image_dimensions(alpha_image))

    params = path_join(folder, "params.json")
    params = load_json(params)
    from easydict import EasyDict

    params = EasyDict(params)

    fname = params.p_name.split(".")[0] + ".png"
    label_image_path = path_join(label_folder, fname)
    label_image = load_image(label_image_path, use_cache=True)
    label_image = as_grayscale_image(label_image)
    label_image = label_image>0
    label_image = as_float_image(label_image)
    if 'pascal_voc' in label_folder:
        label_image = cv_erode(label_image,diameter=7,circular=True)
    
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

    mode=get_file_extension(get_parent_folder(folder)) #Like raster_bilateral etc

    return gather_vars(
        'folder',
        'prompt',
        'params',
        'label_image',
        'alpha_image',
        'input_image',
        'fname',
        'mode',
    )

def get_comparison_image(result):
    a, b, d = result.input_image, result.label_image, result.alpha_image
    c = blend_images(a, b, alpha=0.5)
    e=alpha_checkerboard(result.input_image,result.alpha_image)
    comparison_image = rp.tiled_images([a, b, c, d, e], length=5)
    iou=result_iou(result)
    comparison_image = labeled_image(comparison_image, result.prompt+'      ||      IOU: %.3f'%iou)
    return comparison_image

def alpha_checkerboard(image,alpha):
    board=get_checkerboard_image(*get_image_dimensions(image),second_color=.75)
    return blend_images(board,image,alpha)
    

def unearth_folder(folder):
    name=get_folder_name(folder)
    subs=get_subfolders(folder)
    sub_names=[get_folder_name(x) for x in subs]
    newfolders=[folder+' '+sub_name for sub_name in sub_names]
    for x,y in zip(subs,newfolders):
        move_folder(x,y)
        
        print(x,y)
    assert folder_is_empty(folder)
    delete_directory(folder)

def get_broken_tv_folders():
    #Returns paths like .../dep_peekaboo_results_VOC/2008_005006.tv
    return [x for x in get_subfolders(settings.results_folder) if x.endswith('.tv')]

def fix_broken_tv_folders():
    #VOC we did tv/monitor as a prompt by accident and that fucked with the folder structure
    for folder in get_broken_tv_folders():
        unearth_folder(folder)

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
    #results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_VOC",
    results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_VOC_singleseed",
    label_folder="/raid/datasets/pascal_voc/VOC2012/SegmentationClass",
    dataset_folder="/nfs/ws1/datasets/RefVOC/",
    crop_file_prefix="cropped-",
    DO_CROP=True,
)

settings=voc_settings
#settings= coco_nocrop_settings


def load_results(settings=None,silent=True):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def process_folder(folder):
        try:
            result = process_result(folder, **settings)
            return (result, None)
        except Exception as error:
            if not silent:
                print_stack_trace(error)
            return (None, folder)

    if not settings:
        settings=globals()['settings']

    folders=get_all_folders(settings.results_folder)
    folders=list_flatten(get_subfolders(x) for x in folders)
    print('Loading %i results'%len(folders))

    results = []
    error_folders = []

    with ThreadPoolExecutor() as executor:
        for result, error_folder in tqdm(executor.map(process_folder, folders), total=len(folders)):
            if result is not None:
                results.append(result)
            if error_folder is not None:
                error_folders.append(error_folder)

    return gather_vars('results settings folders error_folders')

def alpha_filter_1(alpha):
    pred_img=alpha
    alpha=pred_img
    std=alpha.std()
    alpha=alpha-alpha.mean()*.56
    alpha=alpha/std/(2+.1)
    pred_img=alpha
    pred_img=rp.cv_gauss_blur(pred_img,10)
    R=45
    pred_img=rp.cv_dilate(pred_img,R,circular=True)
    pred_img=rp.cv_erode(pred_img,R-6,circular=True)
    R=9
    pred_img=rp.cv_gauss_blur(pred_img,80)
    pred_img=rp.cv_erode(pred_img,R,circular=True)
    pred_img=rp.cv_dilate(pred_img,R,circular=True)
    pred_img=rp.as_float_image(pred_img)
    return pred_img

alpha_filter=identity
#alpha_filter=alpha_filter_1
    
def IOU(a, b):
    a=as_binary_image(a)
    b=as_binary_image(b)
    a=as_grayscale_image(a)
    b=as_grayscale_image(b)
    return np.count_nonzero(np.logical_and(a, b)) / np.count_nonzero(np.logical_or(a, b))

def result_iou(result,threshold=.4,alpha_filter=None):
    if alpha_filter is None:
        alpha_filter=globals()['alpha_filter']
    alpha_image,label_image=destructure(result)
    
    assert is_float_image(alpha_image)
    alpha=alpha_image>threshold
    alpha=alpha_filter(alpha)
    assert is_float_image(alpha_image)
    
    return IOU(alpha,label_image)    

def quick_preview(results):
    #First run results=load_results()
    while True:
        display_image_on_macmax(get_comparison_image(random_element(re.results)))
        input('Press enter to see next image')
