videos = [
    "puzzle_9.mp4",
    "puzzle_8.mp4",
    "puzzle_7.mp4",
    "puzzle_6.mp4",
    "puzzle_4.mp4",
    "puzzle_3.mp4",
    "puzzle_24.mp4",
    "puzzle_23.mp4",
    "puzzle_22.mp4",
    "puzzle_21.mp4",
    "puzzle_19.mp4",
    "puzzle_18.mp4",
    "puzzle_17.mp4",
    "puzzle_16.mp4",
    "puzzle_15.mp4",
    "puzzle_14.mp4",
    "puzzle_13.mp4",
    "puzzle_12.mp4",
    "puzzle_11.mp4",
    "puzzle_10.mp4",
    "puzzle_1.mp4",
    "puzzle_0.mp4",
]


def process_video(path):
    ans = path
    ans = load_video(ans)
    ans = list(ans)
    ans = ans + [ans[-1]] * 20
    ans = [ans[0]] * 60 + ans + [ans[-1]] * 60
    ans = ans + ans[::-1]
    print(
        save_video_mp4(
            ans,
            with_file_name(
                path, get_file_name(path, include_file_extension=False) + "_loopable"
            ),
            video_bitrate="medium",
        )
    )


par_map(process_video, videos)
