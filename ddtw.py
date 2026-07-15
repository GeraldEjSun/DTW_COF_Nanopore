import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import resample, find_peaks
from scipy.ndimage import gaussian_filter1d
from tslearn.metrics import dtw_path
from sklearn.metrics import mean_squared_error

# 1. Parameter Settings
input_file = "total_barycenter_600.xlsx"

smooth_sigma = 0

radius_value = 100

# Peak/Valley identification threshold
# Decrease if too few peaks are detected (e.g., 0.08); increase if too many (e.g., 0.15)
prominence_value = 0.001

# Peak/Valley enhanced derivative-DTW weights
signal_weight = 1.0
derivative_weight = 1.6
extrema_weight = 7.4

# =====================================================
# 2. Read Excel Data
# Default format:
# Direction 1: Column B = t, Column C = i
# Direction 2: Column E = t, Column F = i
# =====================================================

df = pd.read_excel(input_file, header=None)
average_signal_1_export = df.iloc[:, [1, 2]].copy()
average_signal_1_export.columns = ["t", "i"]
average_signal_2_export = df.iloc[:, [4, 5]].copy()
average_signal_2_export.columns = ["t", "i"]
average_signal_1_export["t"] = pd.to_numeric(average_signal_1_export["t"], errors="coerce")
average_signal_1_export["i"] = pd.to_numeric(average_signal_1_export["i"], errors="coerce")
average_signal_2_export["t"] = pd.to_numeric(average_signal_2_export["t"], errors="coerce")
average_signal_2_export["i"] = pd.to_numeric(average_signal_2_export["i"], errors="coerce")
average_signal_1_export = average_signal_1_export.dropna().reset_index(drop=True)
average_signal_2_export = average_signal_2_export.dropna().reset_index(drop=True)
time_avg_1 = average_signal_1_export["t"].to_numpy(dtype=float)
signal_avg_1 = average_signal_1_export["i"].to_numpy(dtype=float)
signal_avg_2 = average_signal_2_export["i"].to_numpy(dtype=float)


# 3. Resample Direction 2 to match Direction 1 length
signal_avg_2_resampled = resample(signal_avg_2, len(signal_avg_1))

# 4. Utility Functions
def zscore(y):
    y = np.asarray(y, dtype=float)
    return (y - np.mean(y)) / (np.std(y) + 1e-8)


def fill_nan_by_interp(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    good = np.isfinite(y)

    if good.sum() == 0:
        raise ValueError("Warped signal consists entirely of NaNs. DTW path reconstruction failed.")

    if good.sum() < len(y):
        y[~good] = np.interp(x[~good], x[good], y[good])

    return y

# 5. Use smoothed curves to find peaks/valleys and calculate DTW features
if smooth_sigma > 0:
    signal_1_used = gaussian_filter1d(signal_avg_1, sigma=smooth_sigma)
    signal_2_used = gaussian_filter1d(signal_avg_2_resampled, sigma=smooth_sigma)
else:
    signal_1_used = signal_avg_1.copy()
    signal_2_used = signal_avg_2_resampled.copy()

# Peaks
peaks_1, _ = find_peaks(signal_1_used, prominence=prominence_value)
peaks_2, _ = find_peaks(signal_2_used, prominence=prominence_value)

# Valleys
valleys_1, _ = find_peaks(-signal_1_used, prominence=prominence_value)
valleys_2, _ = find_peaks(-signal_2_used, prominence=prominence_value)

# 6. Construct peak/valley markers
# Peaks = +1, Valleys = -1
extrema_1 = np.zeros_like(signal_1_used)
extrema_2 = np.zeros_like(signal_2_used)

extrema_1[peaks_1] = 1
extrema_1[valleys_1] = -1

extrema_2[peaks_2] = 1
extrema_2[valleys_2] = -1

# Expand single extrema points into local regions for easier capture by DTW
extrema_1 = gaussian_filter1d(extrema_1, sigma=2)
extrema_2 = gaussian_filter1d(extrema_2, sigma=2)

# =====================================================
# 7. Construct peak-valley enhanced derivative-DTW features
# Dimension 1: Original signal
# Dimension 2: First-order derivative
# Dimension 3: Extrema markers
# =====================================================
feature_1 = np.column_stack([
    signal_weight * zscore(signal_1_used),
    derivative_weight * zscore(np.gradient(signal_1_used)),
    extrema_weight * zscore(extrema_1)
])

feature_2 = np.column_stack([
    signal_weight * zscore(signal_2_used),
    derivative_weight * zscore(np.gradient(signal_2_used)),
    extrema_weight * zscore(extrema_2)
])

# =====================================================
# 8. DTW Alignment
# =====================================================
dpath, dtw_distance = dtw_path(
    feature_2,
    feature_1,
    global_constraint="sakoe_chiba",
    sakoe_chiba_radius=radius_value
)

# =====================================================
# 9. Reconstruct Direction 2 based on DTW path
# =====================================================
warped_signal_2 = np.full_like(signal_avg_1, np.nan, dtype=float)

for j in range(len(signal_avg_1)):
    matched_idx = [idx1 for idx1, idx2 in dpath if idx2 == j]

    if matched_idx:
        warped_signal_2[j] = np.mean(signal_avg_2_resampled[matched_idx])

warped_signal_2 = fill_nan_by_interp(warped_signal_2)


# =====================================================
# 10. Calculate MSE
# =====================================================
mse_before = mean_squared_error(signal_avg_1, signal_avg_2_resampled)
mse_after = mean_squared_error(signal_avg_1, warped_signal_2)

print(f"DTW distance = {dtw_distance:.6f}")
print(f"MSE before alignment = {mse_before:.6f}")
print(f"MSE after alignment = {mse_after:.6f}")


# =====================================================
# 11. Plot 1: DTW Aligned Curves
# =====================================================
plt.figure(figsize=(10, 5))

plt.plot(time_avg_1, signal_avg_1, color="black", linewidth=2, label="Average Signal 1")
plt.plot(time_avg_1, warped_signal_2, color="red", linestyle="--", alpha=0.85, label=f"Average Signal 2 after DTW, Radius={radius_value}")

plt.legend()
plt.title("Peak-Valley Enhanced Derivative-DTW Alignment")
plt.xlabel("Original Time of Average Signal 1")
plt.ylabel("Signal Current")
plt.tight_layout()
plt.show()


# =====================================================
# 12. Establish Direction 1 Index -> Direction 2 Index Mapping
# =====================================================
mapping_1_to_2 = {}

for idx1, idx2 in dpath:
    if idx2 not in mapping_1_to_2:
        mapping_1_to_2[idx2] = []
    mapping_1_to_2[idx2].append(idx1)


# =====================================================
# 13. Plot 2: Peak/Valley Mappings on Original Curves
# =====================================================
plt.figure(figsize=(10, 5))

plt.plot(time_avg_1, signal_avg_1, color="black", linewidth=2, label="Average Signal 1")
plt.plot(time_avg_1, signal_avg_2_resampled, color="red", linestyle="--", alpha=0.8, label="Average Signal 2 before warping")

# Direction 1 Peaks/Valleys
plt.scatter(time_avg_1[peaks_1], signal_avg_1[peaks_1], color="black", s=55, marker="o", label="Signal 1 peaks")
plt.scatter(time_avg_1[valleys_1], signal_avg_1[valleys_1], color="black", s=55, marker="v", label="Signal 1 valleys")

# Direction 2 Peaks/Valleys
plt.scatter(time_avg_1[peaks_2], signal_avg_2_resampled[peaks_2], color="red", s=55, marker="o", label="Signal 2 peaks")
plt.scatter(time_avg_1[valleys_2], signal_avg_2_resampled[valleys_2], color="red", s=55, marker="v", label="Signal 2 valleys")

# Draw DTW peak/valley mapping lines originating only from Direction 1 peaks/valleys
extrema_points_1 = np.sort(np.concatenate([peaks_1, valleys_1]))

for p1 in extrema_points_1:
    matched_2_list = mapping_1_to_2.get(p1, [])

    if len(matched_2_list) == 0:
        continue

    p2 = int(np.median(matched_2_list))

    plt.plot([time_avg_1[p1], time_avg_1[p2]], [signal_avg_1[p1], signal_avg_2_resampled[p2]], color="gray", linestyle=":", alpha=0.7)

plt.legend(fontsize=8)
plt.title(f"Peak-Valley Mapping Based on Enhanced DTW, Radius={radius_value}")
plt.xlabel("Normalized Time")
plt.ylabel("Signal Current")
plt.tight_layout()
plt.show()


# =====================================================
# 14. Calculate MSE values with linear correction
# =====================================================
import numpy as np
from sklearn.metrics import mean_squared_error

def linear_correct_mse(template_signal, target_signal):
    """
    Linearly corrects target_signal to fit template_signal, 
    and computes the MSE before and after correction.
    """
    template_signal = np.asarray(template_signal, dtype=float)
    target_signal = np.asarray(target_signal, dtype=float)

    # Original MSE
    mse_raw = mean_squared_error(template_signal, target_signal)

    # Linear correction: template ≈ a * target + b
    A = np.vstack([
        target_signal,
        np.ones_like(target_signal)
    ]).T

    a, b = np.linalg.lstsq(A, template_signal, rcond=None)[0]

    target_signal_corrected = a * target_signal + b

    # Corrected MSE
    mse_corrected = mean_squared_error(
        template_signal,
        target_signal_corrected
    )

    return mse_raw, mse_corrected, a, b, target_signal_corrected


# =====================================================
# 15. Pre-alignment MSE
#    signal_avg_1 vs signal_avg_2_resampled
# =====================================================
mse_before_raw, mse_before_corrected, a_before, b_before, signal_avg_2_resampled_corrected = (
    linear_correct_mse(
        signal_avg_1,
        signal_avg_2_resampled
    )
)


# =====================================================
# 16. Post-alignment MSE
#    signal_avg_1 vs warped_example_2 (warped_signal_2)
# =====================================================
# Note: The code originally referenced 'warped_example_2'. Assuming it points to warped_signal_2:
mse_after_raw, mse_after_corrected, a_after, b_after, warped_example_2_corrected = (
    linear_correct_mse(
        signal_avg_1,
        warped_signal_2
    )
)


# =====================================================
# 17. Output Results
# =====================================================
print("========== Pre-alignment MSE ==========")
print("Pre-alignment raw MSE =", mse_before_raw)
print("Pre-alignment linearly corrected MSE =", mse_before_corrected)
print("Pre-alignment correction parameter a_before =", a_before)
print("Pre-alignment correction parameter b_before =", b_before)

print()

print("========== Post-DTW Alignment MSE ==========")
print("Post-DTW raw MSE =", mse_after_raw)
print("Post-DTW linearly corrected MSE =", mse_after_corrected)
print("Post-DTW correction parameter a_after =", a_after)
print("Post-DTW correction parameter b_after =", b_after)