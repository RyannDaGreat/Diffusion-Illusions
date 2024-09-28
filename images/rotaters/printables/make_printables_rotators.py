@memoized
def load_pair(paths):
        names=get_file_names(paths,include_file_extensions=False)
        names=[longest_common_prefix(*names)]*len(names)
        ims=load_images(paths,use_cache=True)
        t1=10
        T=64
        t2=10
        S=1024
        ims=resize_images(ims,size=(S,S))
        ims=bordered_images_solid_color(ims,thickness=t1)
        #ims=bordered_images_solid_color(ims,thickness=T,top=0)
        ims=labeled_images(ims,names,size=T,background_color=(255,255,255),text_color=(0,0,0),position='top')
        ims=labeled_images(ims,' ',size=T,background_color=(255,255,255),text_color=(0,0,0),position='right')
        ims=labeled_images(ims,' ',size=T,background_color=(255,255,255),text_color=(0,0,0),position='left')
        ims=labeled_images(ims,['Base','Rotator'],size=T,background_color=(255,255,255),text_color=(0,0,0),position='bottom')
        ims=bordered_images_solid_color(ims,thickness=t2)
        #ims=tiled_images(ims,border_color=(0,0,0,1),length=len(ims))
        #display_image(ims)
        #input(names)
        return ims
ans=get_all_image_files()
ans=split_into_sublists(ans,2)
rotators=par_map(load_pair,ans)
ccc=split_into_sublists(rotators,3)
immy=[tiled_images(list_flatten(x),length=2,border_color=(0,0,0,1)) for x in ccc]
import sys,os;os.chdir('printables');sys.path.append(os.getcwd())# TAKE printables
ans=save_images(immy,[x+'.jpg' for x in list(map(str,range(len(immy))))],show_progress=True)
__import__("rp").open_file_with_default_application('.')
__import__("rp").open_file_with_default_application('.')
ans=save_images(immy,[x+'.jpg' for x in list(map(str,range(len(immy))))],show_progress=True)b