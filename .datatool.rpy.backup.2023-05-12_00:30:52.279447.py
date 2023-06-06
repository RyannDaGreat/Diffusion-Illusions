# 2023-05-11 21:53:35.871249
from tqdm import tqdm

@memoized
def process_result(
    folder=".", *,label_folder, dataset_folder, crop_file_prefix, DO_CROP, **_
):
    #Expecting a folder like /home/ryan/.../Diffusion-Illusions/untracked/peekaboo_results_coco/2660-COCO_train2014_000000173538.jpg.bird on left.raster_bilateral/000
    #Coco (nocrop vs crop???)

    mode=get_file_extension(get_parent_folder(folder)) #Like raster_bilateral etc
    if mode in BLACKLISTED_MODES:
        assert False, 'Dont include this mode. This is for speed during my tests - I already have a good idea of which wont work well.'

    from easydict import EasyDict
    params = path_join(folder, "params.json")
    params = load_json(params)
    params = EasyDict(params)

    input_image_path = path_join(folder, "image.png")
    alpha_image_path = path_join(folder, "alphas/0.png")
    input_image = load_image(input_image_path, use_cache=True)
    alpha_image = load_image(alpha_image_path, use_cache=True)

    alpha_image = as_float_image    (alpha_image)
    alpha_image = as_grayscale_image(alpha_image)
    input_image = as_float_image    (input_image)
    input_image = as_rgb_image      (input_image)

    input_image=cv_resize_image(input_image,get_image_dimensions(alpha_image))

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

###############

from easydict import EasyDict

def get_comparison_image(result):
    a, b, d = result.input_image, result.label_image, result.alpha_image
    dd=d
    d=alpha_filter(d)
    c = blend_images(a, b, alpha=0.5)
    e=alpha_checkerboard(result.input_image,result.alpha_image)
    comparison_image = rp.tiled_images([a, b, c, d, e,dd], length=5)
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


def load_result_bundle(settings=None,silent=True):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from copy import deepcopy
    

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
    settings=deepcopy(settings) #Protect it from any mutations

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

@memoized
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
    # pred_img=rp.cv_erode(pred_img,R-6,circular=True)
    pred_img=rp.cv_erode(pred_img,R,circular=True)
    R=9
    pred_img=rp.cv_gauss_blur(pred_img,80)
    pred_img=rp.cv_erode(pred_img,R,circular=True)
    pred_img=rp.cv_dilate(pred_img,R,circular=True)
    pred_img=rp.as_float_image(pred_img)
    return pred_img

@memoized
def alpha_filter_2(alpha):
    pred_img=alpha
    alpha=pred_img
    std=alpha.std()
    #alpha=alpha-alpha.mean()*.56
    #alpha=alpha/std/(2+.1)
    #pred_img=alpha
    #pred_img=rp.cv_gauss_blur(pred_img,10)
    
    R=40
    pred_img=rp.cv_dilate(pred_img,R,circular=True)
    pred_img=rp.cv_erode(pred_img,R,circular=True)
    # R=10
    # pred_img=rp.cv_gauss_blur(pred_img,10)
    # pred_img=rp.cv_erode(pred_img,R,circular=True)
    # pred_img=rp.cv_dilate(pred_img,R,circular=True)
    pred_img=rp.as_float_image(pred_img)
    return pred_img


def chunkoozle(alpha,r=10,iter=1):
    #Bad name maybe
    alpha=as_grayscale_image(alpha)
    alpha=as_float_image(alpha)
    a=alpha
    for _ in range(iter):
        oa=a
        a=cv_dilate(a,r,circular=True)
        a=cv_erode(a,r,circular=True)
        a=as_float_image(a)
        a=np.maximum(a,oa)
    return a

@memoized
def alpha_filter_3(alpha):
    alpha=chunkoozle(alpha,60,2)
    alpha=rp.as_float_image(alpha)
    return alpha

@memoized
def alpha_filter_4(alpha):
    alpha=chunkoozle(alpha,90,3)
    alpha=rp.as_float_image(alpha)
    return alpha


#@memoized
#def alpha_filter_5(alpha):
    #alpha=chunkoozle(alpha,30,4)
    #alpha=rp.as_float_image(alpha)
    #return alpha



def IOU(a, b):
    a=as_binary_image(a)
    b=as_binary_image(b)
    a=as_grayscale_image(a)
    b=as_grayscale_image(b)
    return np.count_nonzero(np.logical_and(a, b)) / np.count_nonzero(np.logical_or(a, b))

def result_iou(result, thresholds=[.4], alpha_filter=None):
    alpha_filter = alpha_filter or globals()['alpha_filter']
    alpha=result.alpha_image
    label=result.label_image

    assert is_float_image(alpha)
    alpha = alpha_filter(alpha)
    assert is_float_image(alpha)

    thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
    outs = [IOU(alpha > threshold, label) for threshold in thresholds]

    return outs[0] if len(outs) == 1 else outs

def quick_preview(results=None):
    results = results or globals()['results']
    #First run result_bundle=load_results();results=result_bundle.results
    while True:
        display_image_on_macmax(get_comparison_image(random_element(results)))
        input('Press enter to see next image')

def mean_iou(results=None, thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9]):
    results = results or globals()['results']
    thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
    outs = [mean([result_iou(x, t) for x in tqdm(results)]) for t in thresholds]
    return outs

def make_report(results=None,thresholds=[.4]):
    results = results or globals()['results']
    modes=cluster_by_key(results,key= lambda x: x.mode,as_dict=True)
    return {x:mean_iou(y,thresholds) for x,y in modes.items()}

def make_report_per_prompt(results = None,thresholds=[.4]):
    results = results or globals()['results']
    prompts = cluster_by_key(results, key=lambda x: x.prompt, as_dict=True)
    out = {}
    for prompt in prompts:
        print()
        fansi_print(prompt, 'green', 'bold')
        out[prompt] = make_report(prompts[prompt],thresholds)
        fansi_print(out[prompt])
    print_data_as_table(dict_transpose(out))
    return out

def add_means_to_nested_dict(ans):
    from collections import OrderedDict
    ans=OrderedDict(ans)
    
    for key in ans: ans[key]['MEAN']=as_numpy_array(list(ans[key].values())).mean(0).tolist()
    ans=dict_transpose(ans)
    for key in ans: ans[key]['MEAN']=as_numpy_array(list(ans[key].values())).mean(0).tolist()
    ans=dict_transpose(ans)
    return ans

def merged_results(results):
    #Group them by file and average them
    #We'll use this '
    q=list(chunk_by_key(results, lambda x: x.fname))
    w=[x[0] for x in q]
    out=[]
    from copy import deepcopy
    for i,c in enumerate(q):
        alpha_image=np.mean(as_numpy_array([x.alpha_image for x in c]),0)
        r=deepcopy(c[0])
        r.alpha_image=alpha_image
        out.append(r)
    return out
    

def print_data_as_table(data):
    from rich.table import Table
    from rich.console import Console
    from rich.text import Text

    data=add_means_to_nested_dict(data)

    # Initialize a console
    console = Console()

    # Define labels and categories
    labels = list(data.keys())
    categories = list(data[labels[0]].keys())

    num_threshs=len(data[labels[0]][categories[0]])

    # Iterate through each threshold
    for n in range(num_threshs):
        print()
        print('Thresh',n)

        # Create a new table for each threshold
        table = Table(show_header=True, header_style="bold magenta")

        # Add columns to the table (one for each category)
        table.add_column("Label")
        for category in categories:
            table.add_column(category)

        # Define a dictionary to store maximum values of each column
        max_values = {category: float('-inf') for category in categories}

        # First pass to determine the maximum value in each column
        for label in labels:
            for category in categories:
                max_values[category] = max(max_values[category], data[label][category][n])

        # Second pass to add rows to the table (one for each label)
        for label in labels:
            row = [label]
            for category in categories:
                value = data[label][category][n]
                if value == max_values[category]:
                    # If the value is the maximum in its column, format it as bold cyan
                    value_text = Text(f'{value:.3f}', style='underline cyan')
                else:
                    value_text = Text(f'{value:.3f}')
                row.append(value_text)
            table.add_row(*row)

        # Print the table
        console.print(table)

#############

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
    #results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_VOC_singleseed",
    #results_folder="/mnt/md0/nfs/ryan/CleanCode/Projects/Peekaboo/Experiments/Dreams/peekaboo_results",
    label_folder="/raid/datasets/pascal_voc/VOC2012/SegmentationClass",
    dataset_folder="/nfs/ws1/datasets/RefVOC/",
    crop_file_prefix="cropped-",
    DO_CROP=True,
)

settings=voc_settings
#settings= coco_nocrop_settings

alpha_filter=identity
alpha_filter=alpha_filter_1
#alpha_filter=alpha_filter_2
alpha_filter=alpha_filter_3
#alpha_filter=alpha_filter_4
#alpha_filter=alpha_filter_5

BLACKLISTED_MODES=set()
#BLACKLISTED_MODES|=set('clip_raster_bilateral default pure_fourier pure_raster'.split())
#BLACKLISTED_MODES|=set('midas_fourier midas_fourier_low_grav raster_bilateral raster_bilateral_higher_gravity_1 raster_bilateral_higher_gravity_2 raster_bilateral_higher_gravity_3 raster_bilateral_lower_gravity_1 raster_bilateral_lower_gravity_2 raster_bilateral_no_gravity '.split())
BLACKLISTED_MODES|=set('midas_raster_bilateral_lower_grav midas_raster_bilateral_no_grav midas_raster_bilateral_no_grav_100iter midas_raster_bilateral_lower_grav_100iter midas_raster_bilateral_low_grav_100iter'.split()) #Unfinished
#BLACKLISTED_MODES-=set('raster_bilateral raster_bilateral_lower_gravity_2'.split())

result_bundle=load_result_bundle()
results=result_bundle.results
#ans=make_report_per_prompt(thresholds=[.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9])
ans=make_report_per_prompt(thresholds=[.5])

