def get_end_image_files():
    fai=[]
    for dd in get_all_directories():
        with SetCurrentDirectoryTemporarily(dd):
            for x in get_all_directories():
                try:
                    fi=get_all_image_files(x)
                    fi=[x for x in fi if 'logger' in get_file_name(x)]
                    fi=sorted(fi)
                    fi=fi[-1]
                    fai.append(fi)
                except:pass
    return fai
def copy_outputs():
    #files=get_all_image_files()
    files=rp_glob('trial_*/*logger_0010.png')
    files=get_absolute_paths(files)
    for x in files:
        try:
            q = load_json(get_parent_directory(x) + "/metadata.json")
            subjects=[x.split()[-1] for x in gather(q,'prompt_a prompt_b prompt_c prompt_d prompt_z'.split())]
            structure=' '.join(q.prompt_a.split()[:-1])
            name = "$".join(
                [
                    q.SAVE_DIR,
                    "_".join(subjects),
                    structure,
                ]
            )
            name = with_file_extension(name, get_file_extension(x))
            with SetCurrentDirectoryTemporarily(make_directory("out_images")):
                name = get_unique_copy_path(name)
                copy_file(x, name)
                #os.system("cp "+x+" "+name)
                print("COPIED", name)

        except Exception:
            print_stack_trace()
