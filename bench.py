import argparse
import hashlib
import random
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from scipy.stats import kendalltau
from solver import (
    build_distance_matrix,
    extract_features,
    multi_fragment_path,
    or_opt_open,
    refine_second_order,
    two_opt_open,
    estimate_direction_score
)

SAMPLES = {
    "flower": "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4",
    "jellyfish": "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4",
    "bunny": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
}


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        check=True,
    )


def prepare_shuffled_frames(
    video_path: Path, work_dir: Path, fps: int, max_seconds: int, seed: int
) -> tuple[Path, list[str]]:
    raw_dir = work_dir / "raw"
    shuf_dir = work_dir / "shuffled"
    for d in (raw_dir, shuf_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    run_ffmpeg([
        "-t", str(max_seconds),
        "-i", str(video_path),
        "-vf", f"fps={fps},scale=640:-2",
        str(raw_dir / "%05d.png"),
    ])

    raw_frames = sorted(raw_dir.glob("*.png"))
    rng = random.Random(seed)
    truth_names: list[str] = []
    for f in raw_frames:
        rand_name = hashlib.sha1(f"{seed}-{f.name}-{rng.random()}".encode()).hexdigest()[:12] + ".png"
        shutil.copy2(f, shuf_dir / rand_name)
        truth_names.append(rand_name)

    return shuf_dir, truth_names


def evaluate(pred: list[str], truth: list[str]) -> dict[str, float | int]:
    rank = {name: i for i, name in enumerate(truth)}
    pred_ranks = [rank[name] for name in pred]
    tau, _ = kendalltau(pred_ranks, list(range(len(truth))))

    adj_hits = sum(
        1 for i in range(len(pred_ranks) - 1) if abs(pred_ranks[i + 1] - pred_ranks[i]) == 1
    )
    breaks = (len(pred_ranks) - 1) - adj_hits
    return {
        "tau": float(tau),
        "abs_tau": abs(float(tau)),
        "adj_acc": adj_hits / max(1, len(pred_ranks) - 1),
        "breaks": breaks,
    }


def rebuild_video(shuf_dir: Path, pred: list[str], out_mp4: Path, fps: int) -> None:
    concat_file = out_mp4.with_suffix(".txt")
    with concat_file.open("w") as f:
        for name in pred:
            f.write(f"file '{(shuf_dir / name).resolve()}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file), "-pix_fmt", "yuv420p", str(out_mp4)])


def benchmark_video(name: str, video_path: Path, fps: int, seconds: int, seed: int, export_video: bool) -> None:
    work_dir = Path("bench_data") / name
    shuf_dir, truth = prepare_shuffled_frames(video_path, work_dir, fps, seconds, seed)

    paths = sorted(shuf_dir.glob("*.png"))
    t0 = time.perf_counter()
    blurs, grads, coarses, smalls = extract_features(paths)
    d = build_distance_matrix(blurs, grads, coarses)
    s2 = or_opt_open(two_opt_open(multi_fragment_path(d), d), d)
    t_s2 = time.perf_counter() - t0

    t1 = time.perf_counter()
    s3 = refine_second_order(s2, d, smalls, alpha=0.15)
    t_s3 = time.perf_counter() - t1

    t2 = time.perf_counter()
    dir_score = estimate_direction_score(paths, s3)
    s4 = s3[::-1] if dir_score < 0.0 else list(s3)
    t_s4 = time.perf_counter() - t2

    pred_s2 = [paths[i].name for i in s2]
    pred_s3 = [paths[i].name for i in s3]
    pred_s4 = [paths[i].name for i in s4]
    m2 = evaluate(pred_s2, truth)
    m3 = evaluate(pred_s3, truth)
    m4 = evaluate(pred_s4, truth)

    print(f"\n=== {name} ({len(paths)} frames @ {fps} fps) ===")
    print(f"Stage 2 (1st-order TSP):   |tau|={m2['abs_tau']:.4f} (tau={m2['tau']:+.4f})  adj={m2['adj_acc']*100:5.1f}%  breaks={m2['breaks']:3d}  time={t_s2:.2f}s")
    print(f"Stage 3 (+2nd-order ref):  |tau|={m3['abs_tau']:.4f} (tau={m3['tau']:+.4f})  adj={m3['adj_acc']*100:5.1f}%  breaks={m3['breaks']:3d}  time=+{t_s3:.2f}s")
    print(f"Stage 4 (+direction):      |tau|={m4['abs_tau']:.4f} (tau={m4['tau']:+.4f})  score={dir_score:+.4f}  flipped={dir_score < 0}  time=+{t_s4:.2f}s")

    if export_video:
        out_mp4 = work_dir / f"{name}_reconstructed.mp4"
        rebuild_video(shuf_dir, pred_s4, out_mp4, fps)
        print(f"Saved visual check -> {out_mp4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, help="Path to a local MP4/MOV/MKV instead of downloading samples")
    parser.add_argument("--fps", type=int, default=12, help="Extraction FPS (lower = harder motion gaps)")
    parser.add_argument("--seconds", type=int, default=15, help="Max seconds to extract")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", action="store_true", help="Re-encode predicted order to MP4 to visually inspect")
    args = parser.parse_args()

    if args.video:
        benchmark_video(args.video.stem, args.video, args.fps, args.seconds, args.seed, args.export)
        return

    dl_dir = Path("bench_data/downloads")
    dl_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLES.items():
        mp4 = dl_dir / f"{name}.mp4"
        if not mp4.exists():
            print(f"Downloading {name}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, mp4.open("wb") as out:
                shutil.copyfileobj(resp, out)
        benchmark_video(name, mp4, args.fps, args.seconds, args.seed, args.export)


if __name__ == "__main__":
    main()