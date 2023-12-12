results=get_all_folders('/workspace/Diffusion-Illusions/new_borealis_ROTATOR_pop')
slanko=[]
def do_result(x):
    try:
        json=load_json(path_join(x,'metadata.json'))
        seed=json.SEED
        title=json.title
        outname=str(seed)+' '+title+'.png'
        in_name=get_all_image_files(x)
        in_name=[x for x in in_name if 'logger' in x]
        in_name=sorted(in_name)
        in_name=in_name[-1]
        image=load_image(in_name)
        out='/workspace/poppers/'
        out=path_join(outname,out)
        save_image(image,outname)
        print('Saved',outname)
    except Exception as e:
        fansi_print(e,'red')
par_map(do_result,results)