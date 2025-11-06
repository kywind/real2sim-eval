import argparse
import os
import shutil
from glob import glob
from math import ceil
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps


def find_episode_dirs(root: str) -> List[str]:
    eps = [d for d in glob(os.path.join(root, "episode_*")) if os.path.isdir(d)]
    eps = sorted(set(eps))
    return eps


def pick_first_last_rgb(episode_dir: str, cam_idx: int) -> Optional[Tuple[str, str]]:
    if os.path.exists(os.path.join(os.path.dirname(episode_dir), "final_images")):
        assert os.path.exists(os.path.join(os.path.dirname(episode_dir), "start_images"))
        start_image = os.path.join(os.path.dirname(episode_dir), "start_images", os.path.basename(episode_dir) + f"_camera_{cam_idx}.jpg")
        final_image = os.path.join(os.path.dirname(episode_dir), "final_images", os.path.basename(episode_dir) + f'_camera_{cam_idx}.jpg')
    else:
        rgb_dir = os.path.join(episode_dir, f"camera_{cam_idx}", "rgb")
        if not os.path.isdir(rgb_dir):
            return None
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs.extend(glob(os.path.join(rgb_dir, ext)))
        if not imgs:
            return None
        imgs.sort()
        start_image = imgs[0]
        final_image = imgs[-1]
    return start_image, final_image


def load_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 18)
            except Exception:
                pass
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, text: str):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return font.getsize(text)


def make_grid(
    items: List[Tuple[str, str, str]],  # (label, path, start_or_end)
    out_path: str,
    grid_cols: int = 10,
    cell_w: int = 320,
    cell_h: int = 200,
    gutter: int = 8,
):
    """
    items: list of (label, image_path, 'Start'|'End')
    """
    # Preload and letterbox into cells
    cells: List[Image.Image] = []
    for label, path, phase in items:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skip {label} ({phase}): cannot open {path} ({e})")
            continue
        # Preserve aspect: fit within (cell_w, cell_h), then paste centered on cell canvas
        thumb = ImageOps.contain(img, (cell_w, cell_h))
        canvas = Image.new("RGB", (cell_w, cell_h), (20, 20, 20))
        x = (cell_w - thumb.width) // 2
        y = (cell_h - thumb.height) // 2
        canvas.paste(thumb, (x, y))

        # Label
        draw = ImageDraw.Draw(canvas, "RGBA")
        font = load_font()
        text = f"{label} | {phase}"
        tw, th = text_size(draw, font, text)
        pad = 6
        draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(0, 0, 0, 150))
        draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 230))

        cells.append(canvas)

    if not cells:
        raise SystemExit("No images to place in grid.")

    n = len(cells)
    cols = max(1, grid_cols)
    rows = ceil(n / cols)

    grid_w = cols * cell_w + (cols - 1) * gutter
    grid_h = rows * cell_h + (rows - 1) * gutter
    out = Image.new("RGB", (grid_w, grid_h), (10, 10, 10))

    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = c * (cell_w + gutter)
        y = r * (cell_h + gutter)
        out.paste(cell, (x, y))

    out.save(out_path)
    print(f"[OK] Saved grid to: {out_path}")


def main(base_data_dir):

    grid_cols = 10
    cell_w = 320
    cell_h = 200
    gutter = 8
    n_cameras = 2

    data_dir_list = sorted([os.path.join(base_data_dir, d) for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])

    for data_dir in data_dir_list:
        print(f"Processing data directory: {data_dir}")
        episode_dirs = find_episode_dirs(data_dir)
        if not episode_dirs:
            print(f"No episodes under: {data_dir}")
            continue

        for camera_idx in range(n_cameras):
            pairs: List[Tuple[str, str, str]] = []  # (label, path, phase)
            skipped = 0
            for episode_dir in episode_dirs:
                pick = pick_first_last_rgb(episode_dir, camera_idx)
                if pick is None:
                    skipped += 1
                    continue
                first_path, last_path = pick
                episode_name = os.path.basename(episode_dir.rstrip("/"))
                label = f"{episode_name} | cam_{camera_idx}"
                pairs.append((label, first_path, "Start"))
                pairs.append((label, last_path,  "End"))

            if skipped:
                print(f"[INFO] Skipped {skipped} episode(s) without usable RGB for camera_{camera_idx}.")

            make_grid(
                pairs,
                out_path=os.path.join(data_dir, f'grid_camera_{camera_idx}.png'),
                grid_cols=grid_cols,
                cell_w=cell_w,
                cell_h=cell_h,
                gutter=gutter,
            )
    
    out_dir = os.path.join(base_data_dir, 'grid_images')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for camera_idx in range(n_cameras):
        os.makedirs(os.path.join(out_dir, f'grid_camera_{camera_idx}'), exist_ok=True)

    for data_dir in data_dir_list:
        print(f"Processing data directory: {data_dir}")
        episode_dirs = find_episode_dirs(data_dir)
        if not episode_dirs:
            raise SystemExit(f"No episodes under: {data_dir}")

        for camera_idx in range(n_cameras):
            image_from_dir = os.path.join(data_dir, f"grid_camera_{camera_idx}.png")
            image_to_dir = os.path.join(out_dir, f'grid_camera_{camera_idx}', f"{os.path.basename(data_dir)}.png")
            os.system(f"cp {image_from_dir} {image_to_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory containing episode subdirectories.')
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Processing data directory: {data_dir}")
    main(data_dir)
