# DNA Sequencing via Electroporated 2D COF-Lipid Composite Nanopore

## Project Overview

This repository contains a comprehensive signal processing and analysis toolkit for single-molecule DNA detection using a novel **2D Covalent Organic Framework (COF)-Lipid Composite Nanopore**. The nanopore system achieves highly sensitive detection of DNA translocations by integrating COF materials with lipid membranes, overcoming traditional nanopore limitations such as high noise levels and rapid, uncontrolled translocation speeds.

### Key Features

- **Dynamic Time Warping (DTW) Alignment**: Align DNA translocation signals of varying lengths without linear scaling assumptions.
- **Soft-DTW Barycenter Extraction**: Compute representative signal patterns (Fréchet means) from aligned traces to distinguish different DNA translocation behaviors and directions.
- **Comprehensive Signal Processing**: Filter, normalize, and extract meaningful features from raw current-time traces.
- **Statistical Validation**: Implement random subgroup testing to validate the significance of extracted features.
- **Directional Analysis**: Separate and analyze DNA translocation behavior in different directions.

### Scientific Context

DNA translocations through nanopores are inherently stochastic processes, resulting in signals that vary significantly in duration and morphology. This toolkit employs advanced temporal alignment techniques to extract robust, biologically meaningful features despite high signal variability.

---

## Installation

### Requirements

- **Python**: 3.8 or higher
- **Core Dependencies**:
  - NumPy >= 1.19
  - Pandas >= 1.1.0
  - SciPy >= 1.5.0
  - tslearn >= 0.5.2 (for DTW implementations)
  - Matplotlib >= 3.3.0 (for visualization)
  - Seaborn >= 0.11.0 (for enhanced plots)

### Setup Instructions

```bash
# Clone the repository
git clone <repository_url>
cd DTW_COF_Nanopore

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Core Modules

### 1. **Process_Functions.py**

Provides fundamental signal processing operations for nanopore translocation data.

#### Key Functions

| Function | Purpose | Input Data Type | Output Data Type |
|----------|---------|-----------------|------------------|
| `Bessel_Filter` | Low-pass filtering using Bessel filter design | `np.ndarray` (current signal) | `np.ndarray` (filtered signal) |
| `Down_sampling` | Reduce signal sampling rate via decimation | `np.ndarray` (2D: time, current) | `np.ndarray` (downsampled) |
| `interplotation` | Cubic spline interpolation to target length | `pd.DataFrame` (time/current) | `pd.DataFrame` (resampled) |
| `normalized_translocation` | Extract and normalize translocation region | `pd.DataFrame` | `pd.DataFrame` (z-score normalized) |
| `normalized_delta` | Compute relative current change (ΔI/I₀) | `pd.DataFrame` | `pd.DataFrame` (delta-normalized) |
| `Draw_Plot` | Visualize time-series data | `np.ndarray` (time, current) | `None` (plots) |
| `normalized_time_axis` | Min-max normalization of time axis to [0, 1] | `pd.DataFrame` | `pd.DataFrame` |

#### Parameter Reference

##### `Bessel_Filter(data, sampling_frequency=100, order=4, cut_off_frequency=40)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `np.ndarray` | — | Input signal array (current trace) |
| `sampling_frequency` | `float` | 100 | Sampling frequency in Hz |
| `order` | `int` | 4 | Filter order (higher = steeper cutoff) |
| `cut_off_frequency` | `float` | 40 | Cutoff frequency in Hz |

**Returns**: `np.ndarray` — Filtered signal with reduced high-frequency noise.

##### `normalized_translocation(data, threshold=450, drop_head=5, drop_tail=-5)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Input trace with time and current columns |
| `threshold` | `float` | 450 | Current threshold (pA); values above are excluded |
| `drop_head` | `int` | 5 | Number of points to remove from start |
| `drop_tail` | `int` | -5 | Number of points to remove from end (negative indexing) |

**Returns**: `pd.DataFrame` — Normalized trace with z-score standardization.

##### `normalized_delta(data, threshold=450, drop_head=5, drop_tail=-5)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Input trace with time and current columns |
| `threshold` | `float` | 450 | Current threshold for translocation region |
| `drop_head` | `int` | 5 | Points to remove from start |
| `drop_tail` | `int` | -5 | Points to remove from end |

**Returns**: `pd.DataFrame` — Relative current change normalized to baseline I₀.

---

### 2. **soft_DTW_barycenter.py**

Computes representative signal patterns using the Soft-DTW algorithm, producing a Fréchet mean aligned signal.

#### Key Function: `barycenter_plot`

```python
def barycenter_plot(
    aligned_data: List[np.ndarray],
    target_length: int,
    gamma: float = 0.1,
    max_iter: int = 10,
    tol: float = 1e-5
) -> np.ndarray
```

#### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aligned_data` | `List[np.ndarray]` | — | List of 1D aligned current signals (Fréchet mean input) |
| `target_length` | `int` | — | Number of points in the output barycenter |
| `gamma` | `float` | 0.1 | Softness parameter; lower = sharper alignment, higher = smoother |
| `max_iter` | `int` | 10 | Maximum iterations for barycenter convergence |
| `tol` | `float` | 1e-5 | Convergence tolerance (L2 norm difference) |

**Returns**: `np.ndarray` — 1D barycenter signal representing the average aligned translocation profile.

#### Algorithm Notes

- The **Soft-DTW barycenter** is a differentiable approximation of the DTW barycenter.
- The `gamma` parameter controls the smoothness: smaller values produce sharper features, while larger values yield smoother, more robust averages.
- Convergence is typically achieved within 5–10 iterations for nanopore translocation signals.

---

### 3. **Signal_Processing_and_Alignment.py**

Orchestrates the complete signal processing workflow from raw data loading to normalized signal extraction.

#### Function: `Signal_Processing`

```python
def Signal_Processing(
    file_path: str,
    upper_lim: float = 2000,
    lower_lim: float = 100,
    cut_off_frequency: float = 5,
    normalization_method: str = 'standard',
    smooth: bool = False,
    threshold: float = 60
) -> Tuple[List[pd.DataFrame], List[int]]
```

#### Processing Pipeline

```
Raw Excel Data (time, current)
        ↓
    [Loading & Validation]
        ↓
    [Bessel Low-Pass Filtering]
        ↓
    [Translocation Detection & Extraction]
        ↓
    [Threshold-based Feature Selection]
        ↓
    [Time-axis Normalization (0-1 range)]
        ↓
    [Optional Smoothing (Gaussian)]
        ↓
Output: List of DataFrames + Selected Indices
```

#### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | — | Path to Excel file containing raw translocation data |
| `upper_lim` | `float` | 2000 | Upper duration threshold (ms) for signal selection |
| `lower_lim` | `float` | 100 | Lower duration threshold (ms) for signal selection |
| `cut_off_frequency` | `float` | 5 | Bessel filter cutoff frequency (Hz) |
| `normalization_method` | `str` | 'standard' | 'standard' (z-score) or 'delta' (ΔI/I₀) |
| `smooth` | `bool` | False | Apply Gaussian smoothing post-filtering |
| `threshold` | `float` | 60 | Current amplitude threshold (pA) for translocation detection |

**Returns**: 
- `Tuple[List[pd.DataFrame], List[int]]` — (Normalized traces, Selected signal indices)

---

### 4. **dtw_alignment.py**

Performs Dynamic Time Warping alignment of normalized translocation signals and produces aligned averages.

#### Function: `dtw_alignment`

```python
def dtw_alignment(
    time_normalized_data: List[pd.DataFrame],
    signal_index: Optional[List[int]] = None,
    resample_length: Optional[int] = None,
    plot_series: bool = False,
    plot_alignment: bool = False,
    show_matrix: bool = False,
    series_standard: Optional[pd.DataFrame] = None,
    smooth: bool = False,
    constraint: Optional[str] = None,
    sakoe_radius: Optional[float] = 50,
    target_path: Optional[str] = None,
    inverse: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]
```

#### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_normalized_data` | `List[pd.DataFrame]` | — | Normalized traces (time and current columns) |
| `signal_index` | `Optional[List[int]]` | None | Optional labels for traces (used in plot legends) |
| `resample_length` | `Optional[int]` | None | Target length for final resampling; if None, uses longest trace |
| `plot_series` | `bool` | False | Plot the averaged aligned current waveform |
| `plot_alignment` | `bool` | False | Plot all DTW-aligned traces |
| `show_matrix` | `bool` | False | Display normalized pairwise DTW distance matrix |
| `series_standard` | `Optional[pd.DataFrame]` | None | Custom reference trace; if None, selects closest trace to all others |
| `smooth` | `bool` | False | Apply Gaussian smoothing (σ=5) to aligned traces |
| `constraint` | `Optional[str]` | None | 'sakoe_chiba' for constrained DTW, None for unconstrained |
| `sakoe_radius` | `Optional[float]` | 50 | Radius parameter for Sakoe-Chiba constraint |
| `target_path` | `Optional[str]` | None | Directory to save plots; if None, displays interactively |
| `inverse` | `bool` | False | Flip aligned current vertically (useful for directional comparison) |

**Returns**: 
- `Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]` — (Resampled aligned currents, Unique aligned traces [time/current], Average current)

---

### 5. **Specialized DTW Variants**

#### `dtw_alignment_1d_time.py`

Optimized for 1D current vectors with time-axis reconstruction. Extracts current-only signals, performs DTW, and reconstructs aligned time-current pairs.

**Key Difference**: Uses 1D DTW on current profiles instead of full 2D arrays, reducing computational overhead while preserving temporal information.

#### `dtw_alignment_1d.py`

Pure 1D DTW implementation on current vectors. Produces only resampled aligned current traces without explicit time reconstruction.

**Key Difference**: Lightweight 1D optimization; suitable for large signal batches or memory-constrained environments.

#### `dtw_alignment_interplotation.py`

Pre-resamples all input signals to a common length using `TimeSeriesResampler` before DTW. Ensures identical-length inputs to the alignment procedure.

**Key Difference**: Explicit interpolation step before DTW; useful for ensuring uniform sampling across all signals.

---

## Usage Guide

### Example 1: Complete Workflow with Baseline DTW

```python
import os
from Signal_Processing_and_Alignment import Signal_Processing
from dtw_alignment import dtw_alignment
from soft_dtw_barycenter import barycenter_plot

# Step 1: Load and process raw data
file_path = 'data_final/direction_2.xlsx'
time_normalized_data, selected_indices = Signal_Processing(
    file_path,
    upper_lim=2000,
    lower_lim=100,
    cut_off_frequency=5,
    normalization_method='standard',
    smooth=False,
    threshold=60
)

# Step 2: Align signals using DTW
resampled_aligned, aligned_traces, average_current = dtw_alignment(
    time_normalized_data,
    signal_index=selected_indices,
    plot_alignment=True,
    plot_series=True,
    show_matrix=True,
    smooth=False,
    constraint='sakoe_chiba',
    sakoe_radius=None
)

# Step 3: Compute Soft-DTW barycenter
barycenter = barycenter_plot(
    resampled_aligned,
    target_length=len(average_current),
    gamma=0.1,
    max_iter=10,
    tol=1e-5
)

print(f"Processed {len(time_normalized_data)} signals")
print(f"Average current shape: {average_current.shape}")
print(f"Barycenter shape: {barycenter.shape}")
```

### Example 2: Directional Comparison

```python
from dtw_alignment import dtw_alignment
import matplotlib.pyplot as plt

# Process direction 1
data_dir1, idx_dir1 = Signal_Processing(
    'data_final/direction_1.xlsx',
    normalization_method='delta'
)
_, _, avg_dir1 = dtw_alignment(
    data_dir1,
    signal_index=idx_dir1,
    smooth=True
)

# Process direction 2
data_dir2, idx_dir2 = Signal_Processing(
    'data_final/direction_2.xlsx',
    normalization_method='delta'
)
_, _, avg_dir2 = dtw_alignment(
    data_dir2,
    signal_index=idx_dir2,
    smooth=True,
    inverse=True  # Flip for comparison
)

# Visualize comparison
plt.figure(figsize=(10, 6))
plt.plot(avg_dir1, label='Direction 1', linewidth=2)
plt.plot(avg_dir2, label='Direction 2 (Inverted)', linewidth=2)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized Current (pA)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('direction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Example 3: Optimized 1D Processing for Large Datasets

```python
from dtw_alignment_1d import dtw_alignment_1d

# Use lightweight 1D DTW for batch processing
resampled_aligned_1d, average_current_1d = dtw_alignment_1d(
    time_normalized_data,
    plot_alignment=False,  # Skip intermediate plots for speed
    smooth=True,
    resample_length=500
)

print(f"Processed {len(resampled_aligned_1d)} signals with 1D DTW")
```

---

## Mathematical Methodology

### Dynamic Time Warping (DTW)

DTW computes the minimum cumulative distance between two time series by allowing non-linear temporal alignment:

$$\text{DTW}(X, Y) = \min_{\mathbf{p}} \sqrt{\sum_{(i,j) \in \mathbf{p}} d(x_i, y_j)}$$

where $\mathbf{p}$ is a warping path and $d$ is the Euclidean distance. This approach is ideal for comparing nanopore signals of varying durations without assuming linear time scaling.

#### Implementation
- **Algorithm**: Tslearn's `dtw_path` using dynamic programming.
- **Constraint Options**: 
  - **Unconstrained**: Full DP computation, no restrictions on warping path.
  - **Sakoe-Chiba Band**: Restricts warping path to a diagonal band, reducing computation to O(n·r) where r is the radius.

### Soft-DTW Barycenter

The Soft-DTW barycenter is a **differentiable approximation** to the DTW barycenter (Fréchet mean), computed as:

$$\mathbf{b}^* = \arg\min_{\mathbf{b}} \sum_{k=1}^{N} \text{SoftDTW}_{\gamma}(\mathbf{b}, \mathbf{X}_k)$$

where:
- $\text{SoftDTW}_{\gamma}$ is the soft (differentiable) DTW with smoothing parameter $\gamma$
- $\mathbf{X}_k$ are the aligned signal sequences
- Lower $\gamma$ produces sharper, more discriminative features
- Higher $\gamma$ yields smoother, more robust averages

#### Properties
- **Convergence**: Typically 5–10 iterations for nanopore signals
- **Computational Complexity**: O(N·L²·I) where L is signal length and I is iterations
- **Robustness**: Inherently handles variable-length inputs and outliers

---

## Project Structure

```
DTW_COF_Nanopore/
├── Process_Functions.py          # Core signal processing utilities
├── Signal_Processing_and_Alignment.py  # Data loading and preprocessing workflow
├── dtw_alignment.py              # Baseline DTW alignment with full 2D handling
├── dtw_alignment_1d_time.py      # Optimized 1D DTW with time reconstruction
├── dtw_alignment_1d.py           # Lightweight 1D DTW (current-only)
├── dtw_alignment_interplotation.py  # DTW with explicit pre-resampling
├── soft_DTW_barycenter.py        # Soft-DTW barycenter computation
├── main.ipynb                    # Interactive analysis notebook
├── data_final/                   # Processed data directory
│   ├── direction_1.xlsx
│   └── direction_2.xlsx
├── figure_final/                 # Generated visualizations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Key Parameters for Optimal Results

### Filtering
- **cut_off_frequency**: 5–10 Hz for typical nanopore data (removes acquisition noise)
- **order**: 4–6 (higher orders provide steeper rolloff)

### Translocation Detection
- **threshold**: 50–80 pA (depends on nanopore geometry and DNA type)
- **drop_head/drop_tail**: 3–10 points (removes edge artifacts from feature extraction)

### DTW Alignment
- **constraint**: Use 'sakoe_chiba' for speed; set radius to 0.3 × signal_length
- **smooth**: Enable for noisy signals; Gaussian σ=5 is standard

### Soft-DTW Barycenter
- **gamma**: 0.1–1.0 (start with 0.1 for sharp features; increase for robustness)
- **max_iter**: 10–20 (rarely needs >10 for convergence)

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{YourAuthor2024,
  title={DNA Sequencing via Electroporated 2D COF-Lipid Composite Nanopore},
  author={Author, et al.},
  journal={Journal Name},
  year={2024}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Support & Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers at [your-email@institution.edu]

---

## Acknowledgments

This work was supported by [Funding Agency/Institution]. We thank [collaborators] for valuable discussions and technical support.
