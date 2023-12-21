from rp import *
ans=['2download - 2023-12-20T070001.916.png', '3asdpofaospdfkoasdf.png', '3download - 2023-12-20T064132.411.png', '3download - 2023-12-20T173950.110.png', '3download - 2023-12-20T191843.004.png', '4download - 2023-12-20T071829.750.png', '4download - 2023-12-20T072234.896.png', '4download - 2023-12-20T073415.508.png', '4download - 2023-12-20T073507.944.png', '4download - 2023-12-20T154551.087.png', '4download - 2023-12-20T155029.724.png', '4download - 2023-12-20T160355.352.png', '4download - 2023-12-20T164907.843.png', '4download - 2023-12-20T174023.166.png', '4download - 2023-12-20T190621.610.png', '4download - 2023-12-20T191446.729.png', '4download - 2023-12-20T191628.846.png', '4download - 2023-12-20T192024.887.png', '4download - 2023-12-20T215937.115.png', '4download - 2023-12-20T220006.745.png', '4download - 2023-12-20T220017.204.png', '4download - 2023-12-21T032953.497.png', '4download - 2023-12-21T033034.858.png', '4download - 2023-12-21T033103.777.png', '4download - 2023-12-21T033248.575.png', '5download - 2023-12-21T033212.679.png', '6download - 2023-12-21T033134.555.png']
def f(path):
    # 2023-12-20 05:14:32.028049
    print(path)
    name=get_file_name(path)
    if name[0].isnumeric():
        D=int(name[0])
        fansi_print('Detected D='+str(D),'green','bold')
    else:
        D=4
    def create_checkerboard_animation(img, D=D):
        img = resize_image_to_hold(img, height=1024)
        tiles = split_tensor_into_regions(img, D, D)
        pause=60 #Num frames to pause anim
        frames = crop_images_to_max_size(
            [
                tiled_images(
                    [
                        rotate_image(tile, angle * (1 if (i // D + i % D) % 2 else -1))
                        for i, tile in enumerate(tiles)
                    ],
                    border_thickness=0,
                )
                for angle in [*[0] * pause, *range(91), *[90] * pause]
            ],
            origin="center",
        )
        
        shadow_offset=32
        frames=with_drop_shadows(frames,x=shadow_offset,y=shadow_offset,blur=128,opacity=.25)
        frames = resize_images(frames, size=1 / 2)
        frames = [
            #with_alpha_checkerboard(as_rgba_image(x), first_color=0.45, second_color=0.55, tile_size=16)
            blend_images(1,x)
            for x in frames
        ]
        video_frames = (frames + frames[::-1]) * 1
        return video_frames
    
    
    frames = create_checkerboard_animation(
        split_tensor_into_regions(
            # load_image("/Users/ryan/Downloads/iuiuhiuh]1).png"), 1, 2
            # load_image("/Users/ryan/Downloads/oiiojjioijo.png"), 1, 2#BBAD
            # load_image("/Users/ryan/Downloads/download - 2023-12-20T064123.804.png"),
            # load_image("/Users/ryan/Downloads/download - 2023-12-20T155011.890.png"),
            #load_image("/Users/ryan/Downloads/download - 2023-12-20T164907.843.png"),#YodaWeeper
            #load_image('/Users/ryan/Downloads/download - 2023-12-20T174023.166.png'),
            #load_image('/Users/ryan/Downloads/download - 2023-12-20T161550.531.png'),#WWoman
            load_image(path),#WWoman
            1,
            2,
        )[0],
    )
    
    save_video_mp4(
        frames,
        get_unique_copy_path("short_pvideo.mp4"),
        video_bitrate="medium",
    )
    #display_video(frames, framerate=60)
i=int(sys.argv[1])
print(" i = ",i)
sleep(.1)
path=ans[i]
f(path)

