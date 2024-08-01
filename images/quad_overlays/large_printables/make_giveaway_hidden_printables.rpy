strips = [
    horizontally_concatenated_images(
        load_images(paths, use_cache=True, show_progress=True)
    )
    for paths in list_flatten(
        [
            split_into_sublists(
                [
                    x
                    for x in get_all_image_files("images_giveaway", sort_by="number")
                    if not "rpoq" in x
                ],
                4,
            )
        ]
    )
]
pages = [vertically_concatenated_images(x) for x in split_into_sublists(strips, 3)]
save_images(
    pages,
    make_directory("giveaway_hidden_printables") + "/page_%i",
    show_progress=True,
)
