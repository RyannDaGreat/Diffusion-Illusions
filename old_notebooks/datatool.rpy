# 2023-05-16 03:35:13.467056
# 2023-05-11 21:53:35.871249
#FOR COCO
from tqdm import tqdm
# 2023-05-16 05:02:33.744045
import random
from collections import defaultdict
from typing import List

@memoized
def process_result(
    folder=".", *,label_folder, dataset_folder, crop_file_prefix, DO_CROP, **_
):
    #Expecting a folder like /home/ryan/.../Diffusion-Illusions/untracked/peekaboo_results_coco/2660-COCO_train2014_000000173538.jpg.bird on left.raster_bilateral/000
    #Coco (nocrop vs crop???)

    mode=get_file_extension(get_parent_folder(folder)) #Like raster_bilateral etc
    if mode in BLACKLISTED_MODES:
        assert False, 'Dont include this mode. This is for speed during my tests - I already have a good idea of which wont work well.'
    if WHITELISTED_MODES and mode not in WHITELISTED_MODES:
        assert False

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
            assert False
            return #The dataset doesn't have it???
        x1, x2, y1, y2 = list(map(int, text_file_to_string(crop_filename).strip().split()))
        label_image = label_image[y1:y2, x1:x2]

    prompt = params.label[len("SimpleLabel(name=") : -1]
    
    iou_cache=HandyDict()
    hash_code=rp.random_namespace_hash()

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
        'iou_cache',
        'hash_code',
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
    try:
        return np.count_nonzero(np.logical_and(a, b)) / np.count_nonzero(np.logical_or(a, b))
    except ZeroDivisionError:
        return 0

def result_iou(result, thresholds=[.4], alpha_filter=None):
    
    alpha_filter = alpha_filter or globals()['alpha_filter']

    thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
    outs=[]
    for threshold in thresholds:
        cache_key=handy_hash((threshold,alpha_filter.__name__))
        cache_key=str(cache_key)
        if cache_key in result.iou_cache:
            iou=result.iou_cache[cache_key]
        else:
            alpha=result.alpha_image
            label=result.label_image

            assert is_float_image(alpha)
            alpha = alpha_filter(alpha)
            assert is_float_image(alpha)
            iou=IOU(alpha > threshold, label)
            result.iou_cache[cache_key]=iou
        outs.append(iou)
    #outs = [IOU(alpha > threshold, label) for threshold in thresholds]

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
    try:print_data_as_table(dict_transpose(out))
    except Exception as e:
        print_verbose_stack_trace(e)
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
        
def filter_good_keys(ans):
    keys=get_good_keys(ans)
    ans=dict_transpose(ans)
    ans={x:y for x,y in ans.items() if x in keys}
    return ans
    
def get_good_keys(ans):
    ans=dict_transpose(ans)
    ans={x:len(y) for x,y in ans.items()}
    ans={x for x,y in ans.items() if y>=20}
    return ans

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

    # Create a table
    table = Table(show_header=True, header_style="bold magenta")

    # Add columns to the table (one for each category)
    table.add_column("Label")
    for category in categories:
        table.add_column(category)

    # Iterate through each threshold
    max_values = {category: float('-inf') for category in categories}
    for n in range(num_threshs):
        print('Thresh',n)

        # Define a dictionary to store maximum values of each column

        # First pass to determine the maximum value in each column
        for label in labels:
            for category in categories:
                max_values[category] = max(max_values[category], data[label][category][n])

    for label in labels:
        # Second pass to add rows to the table (one for each label)
        for n in range(num_threshs):
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
        
        separator = " "   # you may adjust the length of the separator
        table.add_row(*(['[bold yellow]' + separator + '[/bold yellow]'] * (len(categories) + 1)))

    # Print the table
    console.print(table)
    
def make_pure_midas_from_results():
    #Make a backup first just to be safe! Hours of work...
    ans=unique(results,key= lambda x: x.input_image)
    ur=list(unique(results,key= lambda x: x.input_image))
    ans=[x.folder for x in ur]
    ans=ans[0]
    ans=get_path_parent(ans,)
    ans=get_path_parent(ans,)
    set_current_directory(ans)
    newfs=[]
    for e in ur:
        folder=e.folder
        fn=get_folder_name(folder)
        foo=folder
        folder=get_path_parent(folder)
        new_name=with_file_extension(folder,'pure_midas',replace=True)
        new_name=path_join(new_name,fn)
        newfs.append(new_name)
        copy_folder(foo,get_path_parent(new_name))
        print(new_name)
        print(foo)
        print()
    
    ###################
    assert all(folder_exists(x) for x in newfs)
    import sys,os;os.chdir('/mnt/md0/nfs/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions');sys.path.append(os.getcwd())# CDH FAST
    import source.midas as midas
    mi=midas.MIDAS()
    #ans=f
    #for f in newfs:
        #alpha_name='alphas/0.png'
        #with SetCurrentDirectoryTemporarily(f):
            #assert is_image_file(alpha_name)
        ########
    for f in newfs:
            with SetCurrentDirectoryTemporarily(f):
                alpha_name='alphas/0.png'
                assert is_image_file(alpha_name)
                
                im=load_image('image.png')
                dep=mi.estimate_depth_map(im)
                dep+=2
                dep/=40
                save_image(dep,alpha_name)
                print(get_absolute_path(alpha_name))
    return newfs

def print_incomplete_analysis(ans):
    if 'mean' not in ans:
        ans=add_means_to_nested_dict(ans)
    Q = ans
    ans = list(Q)
    ans = Q["MEAN"]
    a=ans
    ans = {x: max(y) for x, y in ans.items()}
    argans = {x: max_valued_index(y) for x, y in a.items()}
    ans = invert_dict(ans)
    d = {}
    argd = {}
    for x in sorted(ans):
        d[x] = ans[x]
        argd[d[x]] = argans[d[x]]
    pretty_print(d)
    pretty_print(argd)
    
def balance_results(results: List['result'], N: int = 1000) -> List['result']:
    # Create a dictionary of dictionaries of lists
    organized_results = defaultdict(lambda: defaultdict(list))

    # Iterate over the result objects
    for result in results:
        organized_results[result.mode][result.prompt].append(result)

    # Initialize the balanced_results dictionary
    balanced_results = defaultdict(lambda: defaultdict(list))

    # Iterate over each mode and prompt in the organized_results
    for mode in organized_results:
        for prompt in organized_results[mode]:
            # If the number of results for a mode-prompt pair is less than N,
            # sample with replacement, else sample without replacement
            if len(organized_results[mode][prompt]) < N:
                balanced_results[mode][prompt] = random.choices(organized_results[mode][prompt], k=N)
            else:
                balanced_results[mode][prompt] = random.sample(organized_results[mode][prompt], k=N)

    # Flatten balanced_results into a list
    flattened_balanced_results = [result for mode in balanced_results for prompt in balanced_results[mode] for result in balanced_results[mode][prompt]]

    # Verify conditions:
    #assert all(len(balanced_results[x.mode][x.prompt]) == N for x in results), "Condition 1 failed"
    #assert all(isinstance(balanced_results[x.mode][x.prompt], list) for x in results), "Condition 2 failed"
    #assert all(balanced_results[x.mode][x.prompt][i] in results for x in results for i in range(N)), "Condition 3 failed"

    return flattened_balanced_results

def all_ious(thresholds=[.5]):
    re=balance_results(results,N=100)
    #re=results
    modes=set(x.mode for x in re)
    return {mode:[result_iou(x,thresholds,alpha_filter) for x in re if x.mode==mode] for mode in modes}

def percent_overs(numbers,cutoffs=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]):
    return [sum(x>=c for x in numbers)/len(numbers) for c in cutoffs]

def print_precisions():
    try:
        for thresh in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:
            print()
            print('    THESH',thresh)
            aiou=all_ious([thresh])
            for mode in aiou:
                print('        ',mode,'  %.3f'%mean(aiou[mode]))
                print('            ',as_numpy_array(percent_overs(aiou[mode])))
    except KeyboardInterrupt:
        pass
def view_table(data):
    #Launches a program that lets you view tabular data
    #Kinda like microsoft excel, but in a terminal
    #Can view numpy arrays
    #Can view pandas dataframes
    #Can view .csv files (given a filepath)
    #Can view .csv files (given the contents as a string)
    #Can view lists of lists such as view_table([[1,2,3],['a','b','c'],[[1,2,3],{'key':'value'},None]])
    #Can view multiline strings that look like tables, such as 'a b c\nd e f\nthings stuff things fourth thing six'


    pip_import('pandas')
    pip_import('tabview')
    import tabview
    import pandas as pd


    temp_file=temporary_file_path('csv')
    try:
        #This works for view_table([[1,2,3],[4,5,6]])
        tabview.view(data) 
    except Exception:
        #ERROR: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        if isinstance(data,str):

            if file_exists(data):
                #Perhaps the data is the path to a .csv file...
                tabview.view(data)
                return

            #If data is a string, perhaps it is the contents of some .csv file
            string_to_text_file(temp_file,data)

        else:
            dataframe=pd.DataFrame(data)
            dataframe.to_csv(temp_file,index=True)

        tabview.view(temp_file)
    finally:
        if file_exists(temp_file):
            delete_file(temp_file)
def view_cherrypick(ans):
    ans=add_means_to_nested_dict(ans)
    ans={x:{y:max(z) for y,z in ans[x].items()} for x in ans}
    view_table(ans)
#############

@memoized
def alpha_filter_5(alpha):
    alpha=chunkoozle(alpha,70,1)
    alpha=rp.as_float_image(alpha)
    return alpha
# COCO
coco_nocrop_settings = EasyDict(
    #results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_coco_nocrop",
    results_folder='/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_COCO_NOC',
    label_folder="/nfs/ws1/datasets/RefCOCO/label",
    dataset_folder="/nfs/ws1/datasets/RefCOCO",
    crop_file_prefix="",
    DO_CROP=False,
)

# COCO-C
coco_crop_settings = EasyDict(
    results_folder='/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_COCO_C',
    label_folder="/nfs/ws1/datasets/RefCOCO-C/label",
    dataset_folder="/nfs/ws1/datasets/RefCOCO-C",
    crop_file_prefix="",
    DO_CROP=True,
)


# VOC
voc_settings = EasyDict(
    #results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_VOC_singleseed",
    #results_folder="/mnt/md0/nfs/ryan/CleanCode/Projects/Peekaboo/Experiments/Dreams/peekaboo_results",
    #results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_pure_midas_results",
    #results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dis_results",
    results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_VOC",
    label_folder="/raid/datasets/pascal_voc/VOC2012/SegmentationClass",
    dataset_folder="/nfs/ws1/datasets/RefVOC/",
    crop_file_prefix="cropped-",
    DO_CROP=True,
)

# VOCMO
vocmo_settings = EasyDict(
    results_folder="/home/ryan/CleanCode/Projects/Peekaboo/Experiments/Github/Diffusion-Illusions/untracked/dep_peekaboo_results_VOCMO",
    label_folder="/raid/datasets/pascal_voc/VOC2012/SegmentationClass",
    dataset_folder="/nfs/ws1/datasets/RefVOC-MO/",
    crop_file_prefix="cropped-",
    DO_CROP=True,
)

settings=voc_settings
settings=vocmo_settings
#settings= coco_nocrop_settings
#settings= coco_crop_settings

alpha_filter=identity
#alpha_filter=alpha_filter_1
alpha_filter=alpha_filter_3
#alpha_filter=alpha_filter_5

BLACKLISTED_MODES=set()
WHITELISTED_MODES=set()
#BLACKLISTED_MODES|=set('clip_raster_bilateral default pure_fourier pure_raster'.split())
#BLACKLISTED_MODES|=set('midas_fourier midas_fourier_low_grav raster_bilateral raster_bilateral_higher_gravity_1 raster_bilateral_higher_gravity_2 raster_bilateral_higher_gravity_3 raster_bilateral_lower_gravity_1 raster_bilateral_lower_gravity_2 raster_bilateral_no_gravity '.split())
#BLACKLISTED_MODES|=set('midas_raster_bilateral_lower_grav midas_raster_bilateral_no_grav midas_raster_bilateral_no_grav_100iter midas_raster_bilateral_lower_grav_100iter midas_raster_bilateral_low_grav_100iter'.split()) #Unfinished
#BLACKLISTED_MODES|={'midas_raster_bilateral_no_grav_1.5', 'midas_raster_bilateral_lower_grav_1.5', 'raster_bilateral_1.5', 'default_1.5', 'midas_raster_bilateral_low_grav_1.5'}
#WHITELISTED_MODES|=set('midas_raster_bilateral_lower_grav midas_raster_bilateral_no_grav midas_raster_bilateral_no_grav_100iter midas_raster_bilateral_lower_grav_100iter midas_raster_bilateral_low_grav_100iter'.split()) #Unfinished
#BLACKLISTED_MODES-=set('raster_bilateral raster_bilateral_lower_gravity_2'.split())
#WHITELISTED_MODES|=set('midas_raster_bilateral_low_grav'.split())

fix_broken_tv_folders()

result_bundle=load_result_bundle()
results=result_bundle.results
#ans=make_report_per_prompt(thresholds=[.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9])
#ans=make_report_per_prompt(thresholds=[.5,.52,.54,.56,.58,.6,.62,.64,.66,.68,.7])
ans=make_report_per_prompt(thresholds=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
#ans=make_report_per_prompt(thresholds=[.54])
#ans=make_report_per_prompt(thresholds=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
#ans=make_report_per_prompt(thresholds=[.5])
#
print_incomplete_analysis(ans)
print_precisions()
