arrow = load_image("arrow.png")
arrow = cv_resize_image(arrow, 1 / 8)

for index, image in enumerate(get_all_image_files("inputs")):
    a, b = split_tensor_into_regions(as_rgba_image(load_image(image)), 1, 2)

    out = blend_images(
        1, horizontally_concatenated_images(a, arrow, b, origin="center")
    )
    display_image(out)
    print(
        save_image(
            out,
            "outputs/image%i.png" % index,
        )
    )
