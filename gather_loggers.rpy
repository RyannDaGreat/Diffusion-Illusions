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
    for x in get_end_image_files():
        try:
            q = load_json(get_parent_directory(x) + "/metadata.json")
            name = "$".join(
                [
                    q.method_name,
                    "_".join(q.subjects),
                    q.prompt_structure,
                ]
            )
            name = with_file_extension(name, get_file_extension(x))
            with SetCurrentDirectoryTemporarily(make_directory("out_images")):
                name = get_unique_copy_path(name)
                copy_file(x, name)
                print("COPIED", name)

        except Exception:
            print_stack_trace()
