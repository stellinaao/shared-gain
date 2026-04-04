# original code by Lukas Oesch, modified by Max Melin, and further modified by Stellina Ao
from wfield import load_stack, approximate_svd, chunk_indices
import gc
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from utils.paths import PROJECT_ROOT
from sg.data import subject_ids, session_ids

LABELS = ["cam0"]  # , "cam1", "cam2"]  # adjust as needed


# ----------------- Helper -----------------
def safe_unlink(file_path, retries=5, delay=0.5):
    """Windows-safe deletion of a file."""
    file_path = Path(file_path)
    for _ in range(retries):
        try:
            file_path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            time.sleep(delay)
    print(f"Warning: could not delete {file_path} after {retries} retries")


# ----------------- Main Pipeline -----------------
def build_video_svd_data(subj_id=None, sess_id=None, subj_idx=None, sess_idx=None):
    """Build SVD for raw video and motion energy."""
    if subj_id is None:
        subj_id = subject_ids[subj_idx]
    if sess_id is None:
        if subj_idx is None:
            subj_idx = np.where(subject_ids == subj_id)[0][0]
        sess_id = session_ids[subj_idx][sess_idx]

    vidfolder = PROJECT_ROOT.parents[0] / "data-np" / subj_id / sess_id

    # Check if all SVD files exist
    missing_npy_file = False
    for label in LABELS:
        svd_path = vidfolder / f"_svd_{label}.npy"
        svd_me_path = vidfolder / f"_me_svd_{label}.npy"
        if not svd_path.exists() or not svd_me_path.exists():
            missing_npy_file = True
            break

    if not missing_npy_file:
        print("All SVD files already exist, skipping video processing.")
        return

    # Load video paths (keep as list-of-lists for backward compatibility)
    vidpaths = []
    for label in LABELS:
        paths = [vidfolder / f"{subj_id}_{sess_id}_{label}_00000000.avi"]
        # Ensure it's a list even if WindowsALFPath is returned
        if not isinstance(paths, (list, tuple, np.ndarray)):
            paths = [paths]
        vidpaths.append(paths)

    # Motion energy videos (preserves list-of-lists structure)
    me_vidpaths = write_motion_energy_video(vidpaths)

    # Compute SVDs
    svd_files = get_video_SVD(vidpaths)
    me_svd_files = get_motion_energy_SVD(me_vidpaths)

    print("SVD processing complete.")
    return svd_files, me_svd_files


# ----------------- Video Processing -----------------
def write_motion_energy_video(files, overwrite=False, grayscale=True):
    """Compute motion energy videos from raw videos."""
    out_files = []
    for group in files:  # keep list-of-lists
        group_out = []
        for current_file in group:
            current_file = Path(current_file)
            me_file = current_file.with_name(
                current_file.stem.replace("raw", "motion_energy") + ".mp4"
            )

            if me_file.exists() and not overwrite:
                group_out.append(me_file)
                continue

            cap = cv2.VideoCapture(str(current_file))
            if not cap.isOpened():
                print(f"Warning: cannot open {current_file}")
                continue

            frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            is_color = not grayscale
            writer = cv2.VideoWriter(
                str(me_file),
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame_rate,
                (w, h),
                is_color,
            )

            success, prev = cap.read()
            if not success:
                cap.release()
                writer.release()
                continue

            if grayscale:
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                writer.write(np.zeros((h, w), dtype=np.uint8))
            else:
                writer.write(np.zeros_like(prev))

            pbar = tqdm(
                total=frame_number - 1,
                desc=f"{current_file.name}",
                unit="frame",
                leave=True,
            )

            while True:
                success, f = cap.read()
                if not success:
                    break

                if grayscale:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

                me = cv2.absdiff(f, prev)  # me calc is here
                writer.write(me)
                prev = f
                pbar.update(1)

            pbar.close()
            cap.release()
            writer.release()
            group_out.append(me_file)
        out_files.append(group_out)

    return out_files


def get_video_SVD(files, overwrite=False):
    """Compute SVD of raw video files (list-of-lists)."""
    out_files = []
    for group in files:
        group_out = []
        for current_file in group:
            current_file = Path(current_file)
            svd_file = current_file.with_name(
                current_file.stem.replace("raw", "svd") + ".npy"
            )

            if svd_file.exists() and not overwrite:
                print(f"SVD already exists at {svd_file}, skipping.")
                safe_unlink(current_file)
                group_out.append(svd_file)
                continue

            print(f"Computing SVD for {current_file}")
            dat = load_stack(str(current_file))

            # n = len(dat)
            # print("len(dat):", n)

            chunkidx = chunk_indices(last_valid_frame(dat), chunksize=256)
            # # probe last 20 frames
            # for i in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
            #     try:
            #         print(i, chunkidx[i])
            #         _ = dat[chunkidx[i][0] : chunkidx[i][1]]
            #     except Exception as e:
            #         print("BAD FRAME:", i, chunkidx[i], e)

            frame_average = []
            for on, off in tqdm(chunkidx, desc="Computing average."):
                frame_average.append(dat[on:off].mean(axis=0))
            frame_average = np.stack(frame_average).mean(axis=0)
            U, SVT = approximate_svd(dat, frame_average, nframes_per_bin=2)
            np.save(svd_file, dict(U=U, SVT=SVT))
            del dat, U, SVT
            gc.collect()
            safe_unlink(current_file)
            group_out.append(svd_file)
        out_files.append(group_out)
    return out_files


def get_motion_energy_SVD(files, overwrite=False):
    """Compute SVD of motion energy videos (list-of-lists)."""
    out_files = []
    for group in files:
        group_out = []
        for current_file in group:
            current_file = Path(current_file)
            svd_file = current_file.with_name(
                current_file.stem.replace("motion_energy", "motion_energy_svd") + ".npy"
            )

            if svd_file.exists() and not overwrite:
                print(f"SVD already exists at {svd_file}, skipping.")
                safe_unlink(current_file)
                group_out.append(svd_file)
                continue

            print(f"Computing motion energy SVD for {current_file}")
            dat = load_stack(str(current_file))
            U, SVT = approximate_svd(dat, np.zeros_like(dat[0]), nframes_per_bin=2)
            np.save(svd_file, dict(U=U, SVT=SVT))
            del dat, U, SVT
            gc.collect()
            safe_unlink(current_file)
            group_out.append(svd_file)
        out_files.append(group_out)
    return out_files


def last_valid_frame(dat, binary=False):
    """
    binary = False because non-sequential probes are costly for VideoStack
    """
    if binary:
        lo = 0
        hi = len(dat) - 1

        while lo <= hi:
            print(lo, hi)
            mid = (lo + hi) // 2
            try:
                _ = dat[mid]
                lo = mid + 1
            except ValueError:
                hi = mid - 1

        return hi + 1
    else:
        true_len = 0
        for i in range(len(dat)):
            if i % 1000 == 0:
                print(i, "/", len(dat))
            try:
                _ = dat[i]
                true_len = i + 1
            except ValueError:
                print(true_len)
                break
        return true_len
