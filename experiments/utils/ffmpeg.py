from pathlib import Path
import subprocess


def make_video(
        image_root: Path,
        video_path: Path,
        image_pattern: str = '%04d.png',
        frame_rate: int = 10):

    subprocess.run([
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
        '-framerate', str(frame_rate),
        '-i', str(image_root / image_pattern),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(video_path)
    ])


def cat_videos(input_videos: list[Path], output_video: Path):

    output_video.parent.mkdir(parents=True, exist_ok=True)

    num_videos = len(input_videos)

    if num_videos <= 1:
        raise ValueError('concatenating <=1 videos')

    video_args = []
    for input_video in input_videos:
        video_args.extend(['-i', str(input_video)])

    subprocess.run([
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
    ] + video_args + [
        '-filter_complex',
        '{}hstack=inputs={}[v]'.format(''.join([f'[{i}:v]' for i in range(num_videos)]), num_videos),
        '-map', '[v]',
        str(output_video)
    ])
