# Notes

## Approach

Consecutive frames in real video are visually close, so recovering frame order maps to finding a **Shortest Hamiltonian Path (Open TSP)** over a pairwise visual distance graph, followed by a decoupled temporal-direction pass.

I used Python (`numpy`, `scipy`, `opencv-python-headless`) instead of the starter Node template for vectorized matrix ops and fast CPU optical flow (`DISOpticalFlow`) inside the offline container.

The solver runs four deterministic CPU stages:

1. **Multi-feature pairwise distance ($N \times N$):** Each frame is resized once and represented by three complementary cues, each normalized by its off-diagonal median before combining ($0.4 / 0.4 / 0.2$):
   - **Blurred 64×64 Lab (L1):** Smooth, near-linear response to small pixel shifts.
   - **Z-scored 64×64 Sobel gradients (L1):** Invariant to auto-exposure and brightness flicker.
   - **Coarse 16×16 Lab grid (L1):** Preserves overlap during fast camera pans.
2. **1st-order Open TSP:** Converts the open path into a closed TSP via a zero-weight dummy node, initializes with **Multi-Fragment** (greedy edge insertion with Union-Find, which naturally builds clean local tracklets first), and refines using candidate-pruned **2-opt** and **Or-opt** (block lengths 1–3, forward and reversed).
3. **2nd-order motion continuity:** Re-optimizes the path with a localized acceleration penalty on blurred 32×32 grayscale triplets, $\|I_i - 2I_j + I_k\|_1 / (\|I_j - I_i\|_1 + \|I_k - I_j\|_1 + \epsilon)$, to discourage figure-8 jumps where oscillating motion revisits similar frames.
4. **Direction heuristic (Arrow of Time):** Samples frame pairs with OpenCV's `DISOpticalFlow`, fits a global affine flow model to measure radial expansion (forward camera/subject motion bias), and measures median vertical flow on foreground residuals (downward/gravity bias). If the combined score is negative, the sequence is reversed.

## What I Tried & Adjusted

- **Skipped deep vision backbones / learned temporal models:** Under CPU-only execution, pure pixel + gradient features already recover exact order on smooth clips in ~1–2s per 120 frames without shipping large model weights.
- **Fixed 2nd-order over-triggering on slow clips:** Initially, using a tiny $\epsilon = 10^{-6}$ and weight $\alpha = 0.5$ caused Stage 3 to *introduce* 2 frame swaps on a slow flower timelapse—when true motion is near zero, compression noise dominates the acceleration ratio. Adding a motion floor ($\epsilon = 0.25 \times \text{median step}$) and lowering $\alpha$ to `0.15` eliminated the regression.
- **Sanity-check benchmarks (shuffled @ 12 fps):** Tested on three public clips (`flower` 60f, `jellyfish` 120f, `bunny` 120f). All three recovered exact order (`tau = +1.0000`, `100%` adjacent-pair accuracy, `0` breaks). These are smooth single-shot clips, so they verify the pipeline mechanics rather than proving real-world generalization across hard edge cases.

## Where It Still Goes Wrong

- **Arrow of time is still weak:** Stage 4 is a macroscopic prior, not a true temporal classifier. Zoom-outs, backward camera tracking, upward motion, or static talking heads can flip the entire video backwards (`tau = -1.0`). Replacing Stage 4 with a lightweight self-supervised temporal-order ONNX model baked into the image at build time would be the clearest next upgrade.
- **Hard scene cuts:** Multi-shot videos decompose into accurate individual shots, but stitching separate shots chronologically is underdetermined by low-level visual distance—TSP will connect unrelated cuts that happen to share similar color palettes or layouts.
- **Large inter-frame gaps / heavy motion blur:** Very low FPS or violent camera shake breaks pixel-grid overlap; adding a small quantized vision embedding (e.g., MobileNetV3-Small via ONNX) as a 4th distance term would help here.
- **Discontinuous / symbolic frames:** Clips without visual continuity between consecutive frames (like `examples/digits/`) are outside what visual similarity can solve and would require task-specific semantics/OCR.
- **Symmetric periodic motion:** If a scene genuinely repeats identical full-frame states (e.g., a pendulum against a static background), 2nd-order smoothness helps at crossings but cannot resolve exact duplicates.
