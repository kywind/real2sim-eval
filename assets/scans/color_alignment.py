import os
from pathlib import Path
import numpy as np
from PIL import Image


def _format_helper(A: np.ndarray, b: np.ndarray) -> str:
    A_round = np.round(A, 3).tolist()
    b_round = np.round(b, 3).tolist()

    def _fmt_matrix(mat):
        rows = [", ".join(f"{v:.3f}" for v in row) for row in mat]
        return "[\n    " + ",\n    ".join(rows) + "\n]"

    def _fmt_vec(vec):
        return "[\n    " + ", ".join(f"{v:.3f}" for v in vec) + "\n]"

    return (
        f"color_A: {_fmt_matrix(A_round)}\n"
        f"color_b: {_fmt_vec(b_round)}\n"
    )


def _write_helper_file(A: np.ndarray, b: np.ndarray, dest: Path) -> None:
    code = _format_helper(A, b)
    dest.write_text(code)


def _get_aug(x: np.ndarray, add_ones: bool = True) -> np.ndarray:
    if add_ones:
        ones = np.ones((x.shape[0], 1), np.float64)
        return np.hstack([x ** 2, x, ones])
    return np.hstack([x ** 2, x])


def _solve_from_samples(S, R):
    S_aug = _get_aug(S)

    weight = np.linalg.norm(R, axis=1) ** 1.0  # NOTE: tunable parameter
    weight = weight / np.max(weight)
    S_aug = S_aug * weight[:, None]
    R = R * weight[:, None]

    # Initial L2 solution
    X, *_ = np.linalg.lstsq(S_aug, R, rcond=None)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("Initial least-squares failed (non-finite values)")

    # Robust IRLS with Tukey bi-weight
    max_iter = 50  # NOTE: tunable parameter
    c = 4.685
    X_prev = X
    w = np.ones((S.shape[0],), np.float64)
    for n_iter in range(max_iter):
        pred = S_aug @ X_prev
        resid = np.linalg.norm(R - pred, axis=1)
        resid = resid / (weight + 1e-10)
        mad = np.median(np.abs(resid - np.median(resid)))
        mad = max(mad, 1e-6)
        scale = c * 1.4826 * mad
        u = resid / scale
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if not np.any(w):
            print(f"No valid weights found, stopping IRLS at iteration {n_iter + 1}")
            break
        sqrt_w = np.sqrt(w)[:, None]
        X_new, *_ = np.linalg.lstsq(S_aug * sqrt_w, R * sqrt_w, rcond=None)
        if not np.all(np.isfinite(X_new)):
            print(f"New solution has non-finite values, stopping IRLS at iteration {n_iter + 1}")
            break
        if np.linalg.norm(X_new - X_prev) < 1e-6:
            X_prev = X_new
            print(f"Converged after IRLS iteration {n_iter + 1}")
            break
        X_prev = X_new
    else:
        print(f"Reached max iterations ({max_iter}) without convergence")    
        print(f"Final error: {np.linalg.norm(R - S_aug @ X_prev)}")

    A = X_prev[:-1, :].T.astype(np.float32)
    b = X_prev[-1, :].T.astype(np.float32)

    return A, b, w


def _apply_transform(img: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    flat = img.reshape(-1, 3).astype(np.float32) / 255.0
    flat_aug = _get_aug(flat, add_ones=False)
    out = flat_aug @ A.T + b
    out = np.clip(out, 0.0, 1.0)
    return (out.reshape(img.shape) * 255.0).astype(np.uint8)


def main():
    src_img_paths = [
        'log/color_alignment/sim/000001.png',
        'log/color_alignment/sim/000002.png',
        'log/color_alignment/sim/000003.png',
        'log/color_alignment/sim/000004.png',
        'log/color_alignment/sim/000005.png',
    ]
    ref_img_paths = [
        'log/color_alignment/real/000001.png',
        'log/color_alignment/real/000002.png',
        'log/color_alignment/real/000003.png',
        'log/color_alignment/real/000004.png',
        'log/color_alignment/real/000005.png',
    ]
    out_dir = 'log/color_alignment/result'

    pixel_src = []
    pixel_ref = []
    image_height = 480
    image_width = 848
    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB")).astype(np.float32) / 255.0
        ref_img = np.array(Image.open(ref_img_path).convert("RGB")).astype(np.float32) / 255.0

        assert src_img.shape == ref_img.shape, "Source and reference images must have the same shape"

        pixel_src.append(src_img.reshape(-1, 3))
        pixel_ref.append(ref_img.reshape(-1, 3))
    
    pixel_src = np.concatenate(pixel_src, axis=0)
    pixel_ref = np.concatenate(pixel_ref, axis=0)

    A, b, w = _solve_from_samples(pixel_src, pixel_ref)

    print("Color correction matrix A:", A)
    print("Color correction bias b:", b)

    os.makedirs(out_dir, exist_ok=True)
    _write_helper_file(A, b, Path(out_dir) / "color_mapping.yaml")

    n_images = len(src_img_paths)
    w_full = w.reshape(n_images, image_height, image_width, -1).repeat(3, axis=-1)
    w_full = w_full * 255.0

    w_mask = (w_full < 1.0).all(axis=-1)  # only zero
    for i in range(n_images):
        w_vis = w_full[i].copy().astype(np.uint8)
        Image.fromarray(w_vis).save(os.path.join(out_dir, f"weights_{i:06d}.png"), quality=95)
        w_mask_vis = (w_mask[i].copy() * 255).astype(np.uint8)
        Image.fromarray(w_mask_vis).save(os.path.join(out_dir, f"weights_mask_{i:06d}.png"), quality=95)

    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB"))
        out_path = os.path.join(out_dir, os.path.basename(src_img_path))
        corr_img = _apply_transform(src_img, A, b)
        Image.fromarray(corr_img).save(out_path, quality=95)

        # concatenate src image, ref image, and corrected image horizontally
        ref_img = np.array(Image.open(ref_img_path).convert("RGB"))
        combined_img = np.hstack((src_img, ref_img, corr_img))
        combined_out_path = os.path.join(out_dir, f"combined_{os.path.basename(src_img_path)}")
        Image.fromarray(combined_img).save(combined_out_path, quality=95)

        print(f"Saved corrected image to {out_path}")


if __name__ == "__main__":
    main()
