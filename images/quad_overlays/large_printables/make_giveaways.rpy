def process_path(path):
    index, path = path
    border_scale = 2 / 3
    image = bordered_image_solid_color(
        bordered_image_solid_color(
            labeled_image(
                labeled_image(
                    labeled_image(
                        labeled_image(
                            bordered_image_solid_color(
                                cv_resize_image(
                                    load_image(path, use_cache=True),
                                    (1024, 1024),
                                ),
                                thickness=round(32 * border_scale),
                            ),
                            text="Hidden Overlay Prime %i / 4" % (int(strip_file_extension(path)[-1]) + 1),
                            size=round(64 * border_scale),
                            text_color=(0, 0, 0),
                            background_color=(255, 255, 255),
                            position="bottom",
                        ),
                        text="Ryan Burgert et al, SIGGRAPH 2024",
                        size=round(64 * border_scale),
                        text_color=(0, 0, 0),
                        background_color=(255, 255, 255),
                    ),
                    text="diffusionillusions.com",
                    size=round(64 * border_scale),
                    text_color=(0, 0, 0),
                    background_color=(255, 255, 255),
                    position="left",
                ),
                text="Variant %i" % (index // 4 + 1) + ":  " + get_file_name(path, include_file_extension=False)[:-2],
                size=round(64 * border_scale),
                text_color=(0, 0, 0),
                background_color=(255, 255, 255),
                position="right",
            ),
            thickness=round(32 * border_scale),
        ),
        color=(0, 0, 0, 1),
        thickness=2,
    )
    name = get_file_name(path)
    out_folder = "images_giveaway"
    out_path = path_join(out_folder, name)
    save_image(image, out_path)


paths = text_file_to_string("image_list.txt").splitlines()
load_files(process_path, enumerate(paths), show_progress=True)
