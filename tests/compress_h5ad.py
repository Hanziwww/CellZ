#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CELLZ compression script for single-cell sparse matrices (.h5ad files)

Usage:
    # 单文件
    python compress_h5ad.py input.h5ad -o output.cellz
    python compress_h5ad.py input.h5ad --verify --compare-algos
    python compress_h5ad.py input.h5ad --compare-algos --benchmark-runs 5
    python compress_h5ad.py input.h5ad --compare-algos --no-csv  # Disable auto CSV

    # 多文件（独立实验，输出分别生成）
    python compress_h5ad.py a.h5ad b.h5ad --compare-algos --output-dir out/

Features:
- Automatic detection of matrix types (UMI counts, log1p-transformed, or general float)
- Optional log1p inversion for normalized data
- Optional stochastic integerization for float matrices
- Roundtrip verification
- Compression ratio comparison with baseline algorithms (multi-core enabled)
- Experimental-level rigorous measurements with warm-up runs
- Theoretical lower bound computation (Shannon entropy)
- Automatic CSV output of all metrics (enabled by default with --compare-algos)

Compression Metrics:
--------------------
1. Compression Ratio: (Raw Serialized Size) / (Compressed Size)
2. Bits per Nonzero: (Compressed Size × 8) / (Number of Nonzeros)
3. Compression Efficiency: (Theoretical Minimum / Actual) × 100%

Naive Theoretical Bound (Not a True Lower Bound):
-------------------------------------------------
This is a LOOSE/NAIVE bound calculated using basic Shannon entropy:
  - Data entropy: H(data values) = -Σ p(v) log₂ p(v) (global distribution)
  - Structure entropy: Σ_i log₂ C(n_cols, k_i) / nnz (naive enumerative coding)

IMPORTANT: This is NOT a strict lower bound! Advanced compressors like CELLZ
can and should beat this bound by exploiting:
  - Delta/gap encoding of sparse indices
  - Local correlations and patterns within data
  - Special value distributions (e.g., UMI data heavily biased toward 1)
  - Matrix reordering for better spatial locality
  - Multi-stage compression (e.g., xz/LZMA on intermediate bitstreams)

This naive bound serves as a reference baseline, not an unbreakable limit.

Compression Ratio Methodology:
------------------------------
All compression ratios are calculated against a common, well-defined baseline:
the raw serialized CSR matrix representation. This ensures fair comparison across
all algorithms.

The baseline consists of:
  - 4-byte header ("CSR0")
  - 16 bytes for shape (2 × uint64)
  - indptr array (n_rows+1 × int64)
  - indices array (nnz × int32)
  - data array (nnz × dtype)
  - Array length headers (3 × uint64)

This raw serialized format represents the minimum overhead for storing a CSR matrix
and is the same for all algorithms being compared.

This methodology follows best practices for compression benchmarking:
1. Use the same input representation for all algorithms
2. Do not include file format overhead (h5ad metadata, etc.)
3. Include only the matrix data itself
4. Report compression ratio, bits-per-nonzero, and efficiency

Dependencies: cellz, anndata, scipy, numpy, pandas
"""

import argparse
import os
import sys
import time
import io
import struct
import csv
import subprocess
import tempfile
import shutil
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import math

try:
    import cellz
    import anndata as ad
    from scipy import sparse as sp
    import pandas as pd
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install via: pip install cellz anndata scipy pandas")
    sys.exit(1)


def load_h5ad_matrices(path: str) -> Tuple[sp.csr_matrix, Optional[sp.csr_matrix]]:
    """Load X and optionally raw.X from h5ad file as CSR matrices."""
    print(f"Loading {path}...")
    adata = ad.read_h5ad(path, backed=None)
    
    # Main matrix X
    X = adata.X
    if hasattr(X, "toarray") and not isinstance(X, np.ndarray):
        X = X.tocsr()
    else:
        X = np.asarray(X)
        X = sp.csr_matrix(X)
    
    # Raw matrix if available
    raw_X = None
    if hasattr(adata, 'raw') and adata.raw is not None:
        raw_X = adata.raw.X
        if hasattr(raw_X, "toarray") and not isinstance(raw_X, np.ndarray):
            raw_X = raw_X.tocsr()
        else:
            raw_X = np.asarray(raw_X)
            raw_X = sp.csr_matrix(raw_X)
    
    return X, raw_X


def classify_matrix_type(csr: sp.csr_matrix, max_samples: int = 100000) -> dict:
    """Classify matrix data type: 'umi' | 'log1p' | 'float'."""
    data = np.asarray(csr.data)
    if data.size == 0:
        return {"type": "umi", "fraction_integer": 1.0, "fraction_log1p_like": 0.0, "has_negative": False}
    
    # Sample for efficiency
    n_samples = min(len(data), max_samples)
    if len(data) > n_samples:
        idx = np.random.choice(len(data), size=n_samples, replace=False)
        sample = data[idx]
    else:
        sample = data
    
    sample64 = sample.astype(np.float64, copy=False)
    
    # Check integer likeness
    diff_to_int = np.abs(sample64 - np.rint(sample64))
    fraction_integer = float(np.mean(diff_to_int <= 1e-9))
    
    has_negative = bool(np.any(sample64 < 0))
    
    # Check log1p likeness
    with np.errstate(over='ignore', invalid='ignore'):
        k = np.rint(np.expm1(np.clip(sample64, -50.0, 50.0)))
        k = np.clip(k, 0, np.inf)
        recon = np.log1p(k)
        closeness = np.abs(sample64 - recon)
    fraction_log1p_like = float(np.mean(closeness <= 1e-8))
    
    # Classify
    if (not has_negative) and fraction_integer >= 0.999:
        mtype = "umi"
    elif (not has_negative) and (fraction_integer < 0.95) and (fraction_log1p_like >= 0.90):
        mtype = "log1p"
    else:
        mtype = "float"
    
    return {
        "type": mtype,
        "fraction_integer": fraction_integer,
        "fraction_log1p_like": fraction_log1p_like,
        "has_negative": has_negative,
    }


def stochastic_round_float_csr(csr: sp.csr_matrix, invert_log1p: bool = False, seed: int = 12345) -> Tuple[sp.csr_matrix, dict]:
    """Convert float CSR to integer CSR using unbiased stochastic rounding."""
    if csr.nnz == 0:
        return csr, {"nnz": 0, "mae": 0.0, "dropped": 0}
    
    data = csr.data.astype(np.float64, copy=False)
    original = data.copy()
    
    # Invert log1p if requested
    if invert_log1p:
        data = np.expm1(np.maximum(data, 0.0))
    
    # Stochastic rounding: for x = floor(x) + frac, return floor + Bernoulli(frac)
    rng = np.random.default_rng(seed)
    n = np.floor(data)
    f = data - n
    u = rng.random(size=data.shape)
    int_data = (n + (u < f).astype(np.int64)).astype(np.int64)
    
    # Compute MAE in original space
    if invert_log1p:
        approx_log1p = np.log1p(int_data.astype(np.float64))
        mae = float(np.mean(np.abs(approx_log1p - original)))
    else:
        mae = float(np.mean(np.abs(int_data.astype(np.float64) - original)))
    
    # Remove zeros
    keep_mask = int_data > 0
    if keep_mask.all():
        new_csr = sp.csr_matrix((int_data, csr.indices, csr.indptr), shape=csr.shape)
        dropped = 0
    else:
        new_data = int_data[keep_mask]
        new_indices = csr.indices[keep_mask]
        
        # Rebuild indptr
        n_rows = csr.shape[0]
        old_indptr = csr.indptr
        counts_per_row = np.diff(old_indptr)
        kept_counts = np.zeros(n_rows, dtype=np.int64)
        offset = 0
        for i, cnt in enumerate(counts_per_row):
            if cnt > 0:
                kept_counts[i] = int(keep_mask[offset:offset+cnt].sum())
            offset += cnt
        new_indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(kept_counts, out=new_indptr[1:])
        new_csr = sp.csr_matrix((new_data, new_indices, new_indptr), shape=csr.shape)
        dropped = int(int_data.size - new_data.size)
    
    return new_csr, {"nnz": int(new_csr.nnz), "mae": mae, "dropped": dropped}


def process_matrix(csr: sp.csr_matrix, name: str, args, seed_offset: int = 0) -> dict:
    """Process a single matrix: classify, integerize if needed."""
    print(f"\nProcessing matrix: {name}")
    n_rows, n_cols = csr.shape
    print(f"  Shape: {n_rows} x {n_cols}, nnz={csr.nnz}")
    
    # Classify matrix type
    cls = classify_matrix_type(csr)
    print(f"  Detected type: {cls['type']}")
    print(f"    - Integer fraction: {cls['fraction_integer']:.4f}")
    print(f"    - Log1p-like fraction: {cls['fraction_log1p_like']:.4f}")
    print(f"    - Has negative: {cls['has_negative']}")
    
    original_csr = csr.copy()
    processed_csr = csr
    stats = {}
    
    # Decide processing plan
    do_invert = False
    do_integerize = False
    
    if cls['type'] == 'umi':
        # Already integer, no processing needed
        print(f"  Action: No preprocessing (already integer UMI counts)")
    elif cls['type'] == 'log1p':
        if args.invert_log1p and not cls['has_negative']:
            do_invert = True
            do_integerize = True
            print(f"  Action: Invert log1p and stochastic round to integers")
        elif args.float_to_int and not cls['has_negative']:
            do_integerize = True
            print(f"  Action: Stochastic round to integers (keeping log1p space)")
        else:
            print(f"  Warning: log1p data detected; consider --invert-log1p for better compression")
    elif cls['type'] == 'float':
        if args.float_to_int and not cls['has_negative']:
            do_integerize = True
            print(f"  Action: Stochastic round to integers")
        else:
            print(f"  Warning: float data; consider --float-to-int for integer compression")
    
    # Apply integerization if needed
    if do_integerize:
        processed_csr, stats = stochastic_round_float_csr(
            csr, 
            invert_log1p=do_invert,
            seed=args.seed + seed_offset
        )
        print(f"  Integerization: MAE={stats['mae']:.6g}, dropped={stats['dropped']}")
    
    return {
        "name": name,
        "type": cls['type'],
        "original": original_csr,
        "processed": processed_csr,
        "stats": stats,
        "do_invert": do_invert,
    }


def get_cpu_count() -> int:
    """Get number of CPU cores for parallel compression."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def compute_theoretical_lower_bound(csr: sp.csr_matrix) -> dict:
    """
    Compute a LOOSE theoretical lower bound for bits per nonzero.
    
    IMPORTANT: This is an UPPER BOUND on the true entropy, not a strict lower bound!
    Real compressors like CELLZ can and should beat this by exploiting:
      - Delta/gap encoding of indices
      - Local correlations and patterns
      - Special value distributions (e.g., heavy bias toward 1 in UMI data)
      - Matrix reordering for better locality
      - Second-order compression (xz/LZMA on bitstreams)
    
    This bound assumes:
    1. Perfect entropy coding for data values (global Shannon entropy, no delta coding)
    2. Naive encoding of sparse structure via enumerative coding:
       - Structure bits: sum_i log2(C(n_cols, k_i)), where k_i = nnz in row i
       - This ignores gap/delta encoding and local patterns
    3. No container/header overhead
    
    Returns keys compatible with downstream usage:
        - entropy_data_bits: bits per nonzero for data values (global entropy)
        - entropy_indices_bits: bits per nonzero for column structure (naive bound)
        - row_overhead_bits: set to 0.0
        - theoretical_bpnz: total bits per nonzero (LOOSE/NAIVE BOUND)
    """
    if csr.nnz == 0:
        return {
            "entropy_data_bits": 0.0,
            "entropy_indices_bits": 0.0,
            "row_overhead_bits": 0.0,
            "theoretical_bpnz": 0.0,
        }
    
    # 1) Entropy of data values (Shannon entropy over observed symbols)
    data = csr.data.astype(np.int64) if np.issubdtype(csr.data.dtype, np.integer) else csr.data
    unique_vals, counts = np.unique(data, return_counts=True)
    probabilities = counts / csr.nnz
    entropy_data = -np.sum(probabilities * np.log2(probabilities + 1e-100))
    
    # Helpers: stable log2 binomial using log-gamma
    def log2_binom(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        if k == 0 or k == n:
            return 0.0
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)

    n_rows, n_cols = csr.shape
    counts_per_row = np.diff(csr.indptr)

    # 2) Strict row-wise enumerative structure bits: sum_i log2(C(n_cols, k_i))
    # Rows with k_i == 0 contribute 0 bits
    structure_bits_total = 0.0
    for k in counts_per_row.tolist():
        if k > 0:
            structure_bits_total += log2_binom(int(n_cols), int(k))

    # Per-nnz contributions
    entropy_indices_per_nnz = structure_bits_total / csr.nnz
    row_overhead_per_nnz = 0.0  # Avoid double counting; covered by per-row enumerative coding

    # Total theoretical bits per nonzero
    theoretical_bpnz = entropy_data + entropy_indices_per_nnz
    
    return {
        "entropy_data_bits": float(entropy_data),
        "entropy_indices_bits": float(entropy_indices_per_nnz),
        "row_overhead_bits": float(row_overhead_per_nnz),
        "theoretical_bpnz": float(theoretical_bpnz),
    }


def serialize_csr_to_bytes(csr: sp.csr_matrix) -> bytes:
    """Serialize CSR matrix to bytes for compression benchmarking."""
    bio = io.BytesIO()
    bio.write(b"CSR0")
    n_rows, n_cols = csr.shape
    bio.write(struct.pack("<QQ", n_rows, n_cols))
    
    indptr = csr.indptr.astype(np.int64, copy=False)
    indices = csr.indices.astype(np.int32, copy=False)
    data = csr.data
    
    for arr in (indptr.view(np.uint8), indices.view(np.uint8), data.view(np.uint8)):
        bio.write(struct.pack("<Q", len(arr)))
        bio.write(arr.tobytes())
    
    return bio.getvalue()


def measure_compression(
    raw_data: bytes,
    compress_func,
    decompress_func,
    name: str,
    warmup_runs: int = 1,
    measurement_runs: int = 3
) -> Dict:
    """
    Measure compression metrics with experimental rigor.
    
    Returns dict with: name, compressed_size, compress_time_mean, compress_time_std,
                       decompress_time_mean, decompress_time_std, compression_ratio
    """
    # Warm-up runs
    for _ in range(warmup_runs):
        try:
            compressed = compress_func(raw_data)
        except Exception:
            pass
    
    # Compression measurements
    compress_times = []
    compressed_result = None
    for _ in range(measurement_runs):
        t0 = time.perf_counter()
        compressed_result = compress_func(raw_data)
        t1 = time.perf_counter()
        compress_times.append(t1 - t0)
    
    # Decompression measurements
    decompress_times = []
    for _ in range(measurement_runs):
        t0 = time.perf_counter()
        decompressed = decompress_func(compressed_result)
        t1 = time.perf_counter()
        decompress_times.append(t1 - t0)
    
    # Verify correctness
    if decompressed != raw_data:
        raise ValueError(f"{name}: Decompression verification failed!")
    
    compress_times = np.array(compress_times)
    decompress_times = np.array(decompress_times)
    
    return {
        "method": name,
        "compressed_size": len(compressed_result),
        "original_size": len(raw_data),
        "compression_ratio": len(raw_data) / len(compressed_result) if len(compressed_result) > 0 else float('inf'),
        "compress_time_mean": float(np.mean(compress_times)),
        "compress_time_std": float(np.std(compress_times, ddof=1)) if len(compress_times) > 1 else 0.0,
        "compress_time_min": float(np.min(compress_times)),
        "decompress_time_mean": float(np.mean(decompress_times)),
        "decompress_time_std": float(np.std(decompress_times, ddof=1)) if len(decompress_times) > 1 else 0.0,
        "decompress_time_min": float(np.min(decompress_times)),
        # Throughput in MiB/s based on original (raw) size
        "compress_throughput_mib_s": (len(raw_data) / 1048576.0) / float(np.mean(compress_times)) if float(np.mean(compress_times)) > 0 else 0.0,
        "decompress_throughput_mib_s": (len(raw_data) / 1048576.0) / float(np.mean(decompress_times)) if float(np.mean(decompress_times)) > 0 else 0.0,
    }


def compare_baseline_algos(csr: sp.csr_matrix, n_cores: int = None, warmup: int = 1, runs: int = 3) -> Tuple[List[Dict], int]:
    """
    Compare with standard compression algorithms (multi-core enabled, maximum compression).
    Returns tuple of (list of measurement dictionaries, raw_serialized_size in bytes).
    
    All compression ratios are calculated against the same baseline: raw serialized CSR data.
    This ensures fair comparison across all algorithms.
    """
    import gzip
    import bz2
    import lzma
    import zlib
    
    if n_cores is None:
        n_cores = get_cpu_count()
    
    print(f"\nBenchmarking baseline compression algorithms (using {n_cores} cores, {runs} runs)...")
    
    # Serialize CSR to bytes - this is our reference baseline
    raw = serialize_csr_to_bytes(csr)
    raw_size = len(raw)
    
    results = []

    # Configure threads for Blosc-based codecs (if available)
    try:
        import blosc as _blosc
        try:
            _blosc.set_nthreads(n_cores)
        except Exception:
            pass
    except ImportError:
        pass
    try:
        import numcodecs as _numcodecs
        if hasattr(_numcodecs, 'blosc') and hasattr(_numcodecs.blosc, 'set_nthreads'):
            try:
                _numcodecs.blosc.set_nthreads(n_cores)
            except Exception:
                pass
    except ImportError:
        pass
    
    # 1. npz (numpy's compressed format with maximum compression)
    print("\n  Testing npz...")
    try:
        def compress_npz(data):
            bio = io.BytesIO()
            np.savez_compressed(bio, indptr=csr.indptr, indices=csr.indices, data=csr.data, shape=np.array(csr.shape))
            return bio.getvalue()
        
        # For npz, we measure compression only (decompression requires loading back into numpy)
        compress_times = []
        compressed_result = None
        
        # Warm-up
        for _ in range(warmup):
            try:
                _ = compress_npz(raw)
            except Exception:
                pass
        
        # Measurements
        for _ in range(runs):
            t0 = time.perf_counter()
            compressed_result = compress_npz(raw)
            t1 = time.perf_counter()
            compress_times.append(t1 - t0)
        
        compress_times = np.array(compress_times)
        result = {
            "method": "npz",
            "compressed_size": len(compressed_result),
            "original_size": raw_size,
            "compression_ratio": raw_size / len(compressed_result),
            "compress_time_mean": float(np.mean(compress_times)),
            "compress_time_std": float(np.std(compress_times, ddof=1)) if len(compress_times) > 1 else 0.0,
            "compress_time_min": float(np.min(compress_times)),
            "decompress_time_mean": 0.0,
            "decompress_time_std": 0.0,
            "decompress_time_min": 0.0,
            "compress_throughput_mib_s": (raw_size / 1048576.0) / float(np.mean(compress_times)) if float(np.mean(compress_times)) > 0 else 0.0,
            "decompress_throughput_mib_s": 0.0,
        }
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # 2. gzip (level 9 - maximum compression, single-threaded in Python)
    print("\n  Testing gzip (level 9)...")
    try:
        if shutil.which("pigz"):
            result = measure_compression(
                raw,
                lambda data: subprocess.run(["pigz", "-9", "-c", "-p", str(n_cores)], input=data, stdout=subprocess.PIPE, check=True).stdout,
                lambda compressed: subprocess.run(["pigz", "-d", "-c", "-p", str(n_cores)], input=compressed, stdout=subprocess.PIPE, check=True).stdout,
                name=f"pigz-9-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        else:
            result = measure_compression(
                raw,
                lambda data: gzip.compress(data, compresslevel=9),
                lambda compressed: gzip.decompress(compressed),
                name="gzip-9",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # 2b. zlib/deflate (level 1 - fast)
    print("\n  Testing zlib (level 1)...")
    try:
        result = measure_compression(
            raw,
            lambda data: zlib.compress(data, level=1),
            lambda compressed: zlib.decompress(compressed),
            name="zlib-1",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 2c. zlib/deflate (level 9 - maximum)
    print("\n  Testing zlib (level 9)...")
    try:
        result = measure_compression(
            raw,
            lambda data: zlib.compress(data, level=9),
            lambda compressed: zlib.decompress(compressed),
            name="zlib-9",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 3. bz2 (level 9 - maximum compression)
    print("\n  Testing bz2 (level 9)...")
    try:
        if shutil.which("pbzip2"):
            result = measure_compression(
                raw,
                lambda data: subprocess.run(["pbzip2", "-9", "-p", str(n_cores), "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout,
                lambda compressed: subprocess.run(["pbzip2", "-d", "-p", str(n_cores), "-c"], input=compressed, stdout=subprocess.PIPE, check=True).stdout,
                name=f"pbzip2-9-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        else:
            result = measure_compression(
                raw,
                lambda data: bz2.compress(data, compresslevel=9),
                lambda compressed: bz2.decompress(compressed),
                name="bz2-9",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # 3b. bz2 (level 1 - faster)
    print("\n  Testing bz2 (level 1)...")
    try:
        result = measure_compression(
            raw,
            lambda data: bz2.compress(data, compresslevel=1),
            lambda compressed: bz2.decompress(compressed),
            name="bz2-1",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 4. lzma/xz (preset 9 with extreme, multi-threaded)
    print(f"\n  Testing lzma (preset 9 extreme, {n_cores} threads)...")
    try:
        # Prefer xz via subprocess with threads; fallback to Python lzma (single-threaded)
        if shutil.which("xz"):
            result = measure_compression(
                raw,
                lambda data: subprocess.run(["xz", "-9", "--extreme", f"-T{n_cores}", "-c"], input=data, stdout=subprocess.PIPE, check=True, stderr=subprocess.DEVNULL).stdout,
                lambda compressed: subprocess.run(["xz", "-d", "-c"], input=compressed, stdout=subprocess.PIPE, check=True, stderr=subprocess.DEVNULL).stdout,
                name=f"xz-9e-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
            results.append(result)
            bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
            print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}")
        else:
            # Fallback to Python lzma (single-threaded)
            result = measure_compression(
                raw,
                lambda data: lzma.compress(data, preset=9 | lzma.PRESET_EXTREME),
                lambda compressed: lzma.decompress(compressed),
                name="lzma-9e",
                warmup_runs=warmup,
                measurement_runs=runs
            )
            results.append(result)
            bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
            print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # 4b. lzma (preset 3 - faster)
    print("\n  Testing lzma (preset 3)...")
    try:
        result = measure_compression(
            raw,
            lambda data: lzma.compress(data, preset=3),
            lambda compressed: lzma.decompress(compressed),
            name="lzma-3",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 5. zstd (multiple levels, multi-threaded)
    print(f"\n  Testing zstd (level 22, {n_cores} threads)...")
    try:
        try:
            import zstandard as zstd
            result = measure_compression(
                raw,
                lambda data: zstd.ZstdCompressor(level=22, threads=n_cores).compress(data),
                lambda compressed: zstd.ZstdDecompressor().decompress(compressed),
                name=f"zstd-22-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        except ImportError:
            if shutil.which("zstd"):
                result = measure_compression(
                    raw,
                    lambda data: subprocess.run(["zstd", "-22", "--ultra", f"-T{n_cores}", "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout,
                    lambda compressed: subprocess.run(["zstd", "-d", "-c"], input=compressed, stdout=subprocess.PIPE, check=True).stdout,
                    name=f"zstd-22-cli-T{n_cores}",
                    warmup_runs=warmup,
                    measurement_runs=runs
                )
            else:
                raise ImportError("zstandard not installed and zstd CLI not found")
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # 5b. zstd (level 3 - fast, multi-threaded)
    print(f"\n  Testing zstd (level 3, {n_cores} threads)...")
    try:
        try:
            import zstandard as zstd
            result = measure_compression(
                raw,
                lambda data: zstd.ZstdCompressor(level=3, threads=n_cores).compress(data),
                lambda compressed: zstd.ZstdDecompressor().decompress(compressed),
                name=f"zstd-3-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        except ImportError:
            if shutil.which("zstd"):
                result = measure_compression(
                    raw,
                    lambda data: subprocess.run(["zstd", "-3", f"-T{n_cores}", "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout,
                    lambda compressed: subprocess.run(["zstd", "-d", "-c"], input=compressed, stdout=subprocess.PIPE, check=True).stdout,
                    name=f"zstd-3-cli-T{n_cores}",
                    warmup_runs=warmup,
                    measurement_runs=runs
                )
            else:
                raise
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 5c. zstd (level 10 - balanced, multi-threaded)
    print(f"\n  Testing zstd (level 10, {n_cores} threads)...")
    try:
        try:
            import zstandard as zstd
            result = measure_compression(
                raw,
                lambda data: zstd.ZstdCompressor(level=10, threads=n_cores).compress(data),
                lambda compressed: zstd.ZstdDecompressor().decompress(compressed),
                name=f"zstd-10-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        except ImportError:
            if shutil.which("zstd"):
                result = measure_compression(
                    raw,
                    lambda data: subprocess.run(["zstd", "-10", f"-T{n_cores}", "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout,
                    lambda compressed: subprocess.run(["zstd", "-d", "-c"], input=compressed, stdout=subprocess.PIPE, check=True).stdout,
                    name=f"zstd-10-cli-T{n_cores}",
                    warmup_runs=warmup,
                    measurement_runs=runs
                )
            else:
                raise
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 5d. zstd (level 6 - common baseline)
    print(f"\n  Testing zstd (level 6, {n_cores} threads)...")
    try:
        try:
            import zstandard as zstd
            result = measure_compression(
                raw,
                lambda data: zstd.ZstdCompressor(level=6, threads=n_cores).compress(data),
                lambda compressed: zstd.ZstdDecompressor().decompress(compressed),
                name=f"zstd-6-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        except ImportError:
            if shutil.which("zstd"):
                result = measure_compression(
                    raw,
                    lambda data: subprocess.run(["zstd", "-6", f"-T{n_cores}", "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout,
                    lambda compressed: subprocess.run(["zstd", "-d", "-c"], input=compressed, stdout=subprocess.PIPE, check=True).stdout,
                    name=f"zstd-6-cli-T{n_cores}",
                    warmup_runs=warmup,
                    measurement_runs=runs
                )
            else:
                raise
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 5e. zstd (level 19 - high compression)
    print(f"\n  Testing zstd (level 19, {n_cores} threads)...")
    try:
        try:
            import zstandard as zstd
            result = measure_compression(
                raw,
                lambda data: zstd.ZstdCompressor(level=19, threads=n_cores).compress(data),
                lambda compressed: zstd.ZstdDecompressor().decompress(compressed),
                name=f"zstd-19-T{n_cores}",
                warmup_runs=warmup,
                measurement_runs=runs
            )
        except ImportError:
            if shutil.which("zstd"):
                result = measure_compression(
                    raw,
                    lambda data: subprocess.run(["zstd", "-19", f"-T{n_cores}", "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout,
                    lambda compressed: subprocess.run(["zstd", "-d", "-c"], input=compressed, stdout=subprocess.PIPE, check=True).stdout,
                    name=f"zstd-19-cli-T{n_cores}",
                    warmup_runs=warmup,
                    measurement_runs=runs
                )
            else:
                raise
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 5f. zstd (level 19 with long-distance matching)
    print(f"\n  Testing zstd (level 19 + long distance, {n_cores} threads)...")
    try:
        import zstandard as zstd
        def zstd19_long_compress(data):
            try:
                params = zstd.ZstdCompressionParameters.from_level(level=19)
                params = zstd.ZstdCompressionParameters(
                    window_log=params.window_log,
                    chain_log=params.chain_log,
                    hash_log=params.hash_log,
                    search_log=params.search_log,
                    min_match=params.min_match,
                    target_length=params.target_length,
                    strategy=params.strategy,
                    enable_ldm=True,
                    ldm_hash_log=getattr(params, 'ldm_hash_log', 0) or 0,
                    ldm_min_match=64,
                    ldm_bucket_size_log=3,
                    ldm_hash_rate_log=7,
                )
                c = zstd.ZstdCompressor(compression_params=params, threads=n_cores)
            except Exception:
                c = zstd.ZstdCompressor(level=19, threads=n_cores)
            return c.compress(data)
        result = measure_compression(
            raw,
            zstd19_long_compress,
            lambda compressed: zstd.ZstdDecompressor().decompress(compressed),
            name=f"zstd-19-long-T{n_cores}",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 6. Brotli (quality 11 - maximum)
    print("\n  Testing brotli (quality 11)...")
    try:
        try:
            import brotli  # c++ extension
        except ImportError:
            import brotlicffi as brotli  # pure python fallback
        result = measure_compression(
            raw,
            lambda data: brotli.compress(data, quality=11),
            lambda compressed: brotli.decompress(compressed),
            name="brotli-11",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}")
    except ImportError:
        print("    Skipped: brotli/brotlicffi not installed (pip install brotli or brotlicffi)")
    except Exception as e:
        print(f"    Failed: {e}")

    # 7. LZ4 (frame) - fast and high-compression settings
    print("\n  Testing lz4 (frame, level 0)...")
    try:
        import lz4.frame as lz4f
        result = measure_compression(
            raw,
            lambda data: lz4f.compress(data, compression_level=0),
            lambda compressed: lz4f.decompress(compressed),
            name="lz4f-0",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}")
    except ImportError:
        print("    Skipped: lz4.frame not installed (pip install lz4)")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n  Testing lz4 (frame, level 12)...")
    try:
        import lz4.frame as lz4f
        result = measure_compression(
            raw,
            lambda data: lz4f.compress(data, compression_level=12),
            lambda compressed: lz4f.decompress(compressed),
            name="lz4f-12",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 8. Snappy (very fast)
    print("\n  Testing snappy...")
    try:
        import snappy
        result = measure_compression(
            raw,
            lambda data: snappy.compress(data),
            lambda compressed: snappy.decompress(compressed),
            name="snappy",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}")
    except ImportError:
        print("    Skipped: python-snappy not installed (pip install python-snappy)")
    except Exception as e:
        print(f"    Failed: {e}")

    # 9. Blosc (optional, multiple codecs)
    print("\n  Testing blosc (zstd, clevel 9, shuffle)...")
    try:
        import blosc
        result = measure_compression(
            raw,
            lambda data: blosc.compress(data, cname='zstd', clevel=9, shuffle=blosc.SHUFFLE),
            lambda compressed: blosc.decompress(compressed),
            name="blosc-zstd9-shuffle",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except ImportError:
        print("    Skipped: blosc not installed (pip install blosc)")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n  Testing blosc (zstd, clevel 9, bitshuffle)...")
    try:
        import blosc
        result = measure_compression(
            raw,
            lambda data: blosc.compress(data, cname='zstd', clevel=9, shuffle=blosc.BITSHUFFLE),
            lambda compressed: blosc.decompress(compressed),
            name="blosc-zstd9-bitshuffle",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n  Testing blosc (lz4, clevel 9, shuffle)...")
    try:
        import blosc
        result = measure_compression(
            raw,
            lambda data: blosc.compress(data, cname='lz4', clevel=9, shuffle=blosc.SHUFFLE),
            lambda compressed: blosc.decompress(compressed),
            name="blosc-lz4-9-shuffle",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n  Testing blosc (lz4hc, clevel 9, shuffle)...")
    try:
        import blosc
        result = measure_compression(
            raw,
            lambda data: blosc.compress(data, cname='lz4hc', clevel=9, shuffle=blosc.SHUFFLE),
            lambda compressed: blosc.decompress(compressed),
            name="blosc-lz4hc-9-shuffle",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n  Testing blosc (lz4hc, clevel 9, bitshuffle)...")
    try:
        import blosc
        result = measure_compression(
            raw,
            lambda data: blosc.compress(data, cname='lz4hc', clevel=9, shuffle=blosc.BITSHUFFLE),
            lambda compressed: blosc.decompress(compressed),
            name="blosc-lz4hc-9-bitshuffle",
            warmup_runs=warmup,
            measurement_runs=runs
        )
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {result['decompress_throughput_mib_s']:.1f} MiB/s")
    except Exception as e:
        print(f"    Failed: {e}")

    # 10. Container baselines: HDF5 chunked+gzip, Zarr v2/v3 + Blosc(zstd)
    print("\n  Testing HDF5 (chunked + gzip)...")
    try:
        import h5py
        # compression only measurement (like npz)
        compress_times = []
        compressed_bytes = None
        for _ in range(warmup):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as f:
                    tmp = f.name
                with h5py.File(tmp, 'w') as h5:
                    h5.create_dataset('indptr', data=csr.indptr, compression='gzip', chunks=True)
                    h5.create_dataset('indices', data=csr.indices, compression='gzip', chunks=True)
                    h5.create_dataset('data', data=csr.data, compression='gzip', chunks=True)
                    h5.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                with open(tmp, 'rb') as f:
                    _ = f.read()
            finally:
                if 'tmp' in locals() and os.path.exists(tmp):
                    os.unlink(tmp)
        for _ in range(runs):
            t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as f:
                tmp = f.name
            try:
                with h5py.File(tmp, 'w') as h5:
                    h5.create_dataset('indptr', data=csr.indptr, compression='gzip', chunks=True)
                    h5.create_dataset('indices', data=csr.indices, compression='gzip', chunks=True)
                    h5.create_dataset('data', data=csr.data, compression='gzip', chunks=True)
                    h5.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                with open(tmp, 'rb') as f:
                    compressed_bytes = f.read()
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            t1 = time.perf_counter()
            compress_times.append(t1 - t0)
        compress_times = np.array(compress_times)
        result = {
            'method': 'HDF5-gzip',
            'compressed_size': len(compressed_bytes),
            'original_size': raw_size,
            'compression_ratio': raw_size / len(compressed_bytes),
            'compress_time_mean': float(np.mean(compress_times)),
            'compress_time_std': float(np.std(compress_times, ddof=1)) if len(compress_times) > 1 else 0.0,
            'compress_time_min': float(np.min(compress_times)),
            'decompress_time_mean': 0.0,
            'decompress_time_std': 0.0,
            'decompress_time_min': 0.0,
            'compress_throughput_mib_s': (raw_size / 1048576.0) / float(np.mean(compress_times)) if float(np.mean(compress_times)) > 0 else 0.0,
            'decompress_throughput_mib_s': 0.0,
        }
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s")
    except ImportError:
        print("    Skipped: h5py not installed (pip install h5py)")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n  Testing Zarr v2 (Zip/DirStore + Blosc(zstd))...")
    try:
        import zarr
        import numcodecs
        compress_times = []
        compressed_bytes = None
        compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)
        for _ in range(warmup):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zarr.zip') as f:
                    zip_path = f.name
                # Prefer ZipStore if available; otherwise, write to DirectoryStore and zip the directory
                if hasattr(zarr, 'storage') and hasattr(zarr.storage, 'ZipStore'):
                    store = zarr.storage.ZipStore(zip_path, mode='w')
                    root = zarr.group(store=store)
                    root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                    root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                    root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                    root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                    store.close()
                    with open(zip_path, 'rb') as f:
                        _ = f.read()
                else:
                    # DirectoryStore fallback
                    tmpdir = tempfile.mkdtemp(prefix='zarr_v2_')
                    try:
                        store = zarr.DirectoryStore(tmpdir)
                        root = zarr.group(store=store)
                        root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                        root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                        root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                        root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                        # Zip the directory to emulate zip store
                        base = os.path.splitext(zip_path)[0]
                        archive_path = shutil.make_archive(base, 'zip', tmpdir)
                        with open(archive_path, 'rb') as f:
                            _ = f.read()
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        base = os.path.splitext(zip_path)[0]
                        if os.path.exists(base + '.zip'):
                            os.unlink(base + '.zip')
            finally:
                if 'zip_path' in locals() and os.path.exists(zip_path):
                    os.unlink(zip_path)
        for _ in range(runs):
            t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zarr.zip') as f:
                zip_path = f.name
            try:
                if hasattr(zarr, 'storage') and hasattr(zarr.storage, 'ZipStore'):
                    store = zarr.storage.ZipStore(zip_path, mode='w')
                    root = zarr.group(store=store)
                    root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                    root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                    root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                    root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                    store.close()
                    with open(zip_path, 'rb') as f:
                        compressed_bytes = f.read()
                else:
                    tmpdir = tempfile.mkdtemp(prefix='zarr_v2_')
                    try:
                        store = zarr.DirectoryStore(tmpdir)
                        root = zarr.group(store=store)
                        root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                        root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                        root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                        root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                        base = os.path.splitext(zip_path)[0]
                        archive_path = shutil.make_archive(base, 'zip', tmpdir)
                        with open(archive_path, 'rb') as f:
                            compressed_bytes = f.read()
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        base = os.path.splitext(zip_path)[0]
                        if os.path.exists(base + '.zip'):
                            os.unlink(base + '.zip')
            finally:
                if os.path.exists(zip_path):
                    os.unlink(zip_path)
            t1 = time.perf_counter()
            compress_times.append(t1 - t0)
        compress_times = np.array(compress_times)
        result = {
            'method': 'Zarr-v2-zip-blosc-zstd',
            'compressed_size': len(compressed_bytes),
            'original_size': raw_size,
            'compression_ratio': raw_size / len(compressed_bytes),
            'compress_time_mean': float(np.mean(compress_times)),
            'compress_time_std': float(np.std(compress_times, ddof=1)) if len(compress_times) > 1 else 0.0,
            'compress_time_min': float(np.min(compress_times)),
            'decompress_time_mean': 0.0,
            'decompress_time_std': 0.0,
            'decompress_time_min': 0.0,
            'compress_throughput_mib_s': (raw_size / 1048576.0) / float(np.mean(compress_times)) if float(np.mean(compress_times)) > 0 else 0.0,
            'decompress_throughput_mib_s': 0.0,
        }
        results.append(result)
        bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
        print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s")
    except ImportError:
        print("    Skipped: zarr/numcodecs not installed (pip install zarr numcodecs)")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n  Testing Zarr v3 (Zip/DirStore + Blosc(zstd))...")
    try:
        import zarr
        import numcodecs
        major = 0
        try:
            major = int(getattr(zarr, '__version__', '0').split('.')[0])
        except Exception:
            major = 0
        if major >= 3:
            compress_times = []
            compressed_bytes = None
            compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)
            for _ in range(warmup):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zarr3.zip') as f:
                        zip_path = f.name
                    if hasattr(zarr, 'storage') and hasattr(zarr.storage, 'ZipStore'):
                        store = zarr.storage.ZipStore(zip_path, mode='w')
                        root = zarr.group(store=store)
                        root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                        root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                        root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                        root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                        store.close()
                        with open(zip_path, 'rb') as f:
                            _ = f.read()
                    else:
                        tmpdir = tempfile.mkdtemp(prefix='zarr_v3_')
                        try:
                            store = zarr.DirectoryStore(tmpdir)
                            root = zarr.group(store=store)
                            root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                            root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                            root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                            root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                            base = os.path.splitext(zip_path)[0]
                            archive_path = shutil.make_archive(base, 'zip', tmpdir)
                            with open(archive_path, 'rb') as f:
                                _ = f.read()
                        finally:
                            shutil.rmtree(tmpdir, ignore_errors=True)
                            base = os.path.splitext(zip_path)[0]
                            if os.path.exists(base + '.zip'):
                                os.unlink(base + '.zip')
                finally:
                    if 'zip_path' in locals() and os.path.exists(zip_path):
                        os.unlink(zip_path)
            for _ in range(runs):
                t0 = time.perf_counter()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zarr3.zip') as f:
                    zip_path = f.name
                try:
                    if hasattr(zarr, 'storage') and hasattr(zarr.storage, 'ZipStore'):
                        store = zarr.storage.ZipStore(zip_path, mode='w')
                        root = zarr.group(store=store)
                        root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                        root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                        root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                        root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                        store.close()
                        with open(zip_path, 'rb') as f:
                            compressed_bytes = f.read()
                    else:
                        tmpdir = tempfile.mkdtemp(prefix='zarr_v3_')
                        try:
                            store = zarr.DirectoryStore(tmpdir)
                            root = zarr.group(store=store)
                            root.create_dataset('indptr', data=csr.indptr, chunks=True, compressor=compressor)
                            root.create_dataset('indices', data=csr.indices, chunks=True, compressor=compressor)
                            root.create_dataset('data', data=csr.data, chunks=True, compressor=compressor)
                            root.create_dataset('shape', data=np.array(csr.shape, dtype=np.int64))
                            base = os.path.splitext(zip_path)[0]
                            archive_path = shutil.make_archive(base, 'zip', tmpdir)
                            with open(archive_path, 'rb') as f:
                                compressed_bytes = f.read()
                        finally:
                            shutil.rmtree(tmpdir, ignore_errors=True)
                            base = os.path.splitext(zip_path)[0]
                            if os.path.exists(base + '.zip'):
                                os.unlink(base + '.zip')
                finally:
                    if os.path.exists(zip_path):
                        os.unlink(zip_path)
                t1 = time.perf_counter()
                compress_times.append(t1 - t0)
            compress_times = np.array(compress_times)
            result = {
                'method': 'Zarr-v3-zip-blosc-zstd',
                'compressed_size': len(compressed_bytes),
                'original_size': raw_size,
                'compression_ratio': raw_size / len(compressed_bytes),
                'compress_time_mean': float(np.mean(compress_times)),
                'compress_time_std': float(np.std(compress_times, ddof=1)) if len(compress_times) > 1 else 0.0,
                'compress_time_min': float(np.min(compress_times)),
                'decompress_time_mean': 0.0,
                'decompress_time_std': 0.0,
                'decompress_time_min': 0.0,
                'compress_throughput_mib_s': (raw_size / 1048576.0) / float(np.mean(compress_times)) if float(np.mean(compress_times)) > 0 else 0.0,
                'decompress_throughput_mib_s': 0.0,
            }
            results.append(result)
            bpnz = (result['compressed_size'] * 8) / csr.nnz if csr.nnz > 0 else 0
            print(f"    Size: {result['compressed_size']:,} bytes, Ratio: {result['compression_ratio']:.3f}x, Time: {result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {result['compress_throughput_mib_s']:.1f} MiB/s")
        else:
            print("    Skipped: zarr v3 API not available")
    except ImportError:
        print("    Skipped: zarr/numcodecs not installed (pip install zarr numcodecs)")
    except Exception as e:
        print(f"    Failed: {e}")

    return results, raw_size


def verify_roundtrip(original_csr: sp.csr_matrix, decompressed_csr: sp.csr_matrix, name: str = "X") -> bool:
    """Verify that decompression matches the original."""
    orig = original_csr.copy()
    decomp = decompressed_csr.copy()
    orig.sort_indices()
    decomp.sort_indices()
    
    if orig.shape != decomp.shape:
        print(f"  [{name}] Shape mismatch: {orig.shape} vs {decomp.shape}")
        return False
    
    if orig.nnz != decomp.nnz:
        print(f"  [{name}] NNZ mismatch: {orig.nnz} vs {decomp.nnz}")
        return False
    
    ok = (
        np.array_equal(orig.indptr, decomp.indptr) and
        np.array_equal(orig.indices, decomp.indices) and
        np.array_equal(orig.data, decomp.data)
    )
    
    if ok:
        print(f"  [{name}] Roundtrip verification: PASSED")
    else:
        print(f"  [{name}] Roundtrip verification: FAILED (data mismatch)")
    
    return ok


def measure_cellz_compression(
    csr: sp.csr_matrix,
    raw_size: int,
    args,
    warmup_runs: int = 1,
    measurement_runs: int = 3
) -> Dict:
    """
    Measure CELLZ compression with experimental rigor.
    
    Args:
        csr: CSR matrix to compress
        raw_size: Size of raw serialized CSR data (baseline for compression ratio)
        args: Command-line arguments containing CELLZ parameters
        warmup_runs: Number of warm-up runs
        measurement_runs: Number of measurement runs
    
    Returns:
        Dictionary with compression metrics, with compression_ratio calculated against raw_size baseline
    """
    print(f"\n  Testing CELLZ...")
    
    # Warm-up runs
    for _ in range(warmup_runs):
        try:
            _ = cellz.compress_csr_to_cellz(
                csr,
                target_block_nnz=args.target_nnz,
                block_xz=args.block_xz,
                block_xz_level=args.block_xz_level,
                block_xz_min_bytes=args.block_xz_min_bytes,
                block_xz_min_ratio=args.block_xz_min_ratio,
            )
        except Exception:
            pass
    
    # Compression measurements
    compress_times = []
    compressed_result = None
    for _ in range(measurement_runs):
        t0 = time.perf_counter()
        compressed_result = cellz.compress_csr_to_cellz(
            csr,
            target_block_nnz=args.target_nnz,
            block_xz=args.block_xz,
            block_xz_level=args.block_xz_level,
            block_xz_min_bytes=args.block_xz_min_bytes,
            block_xz_min_ratio=args.block_xz_min_ratio,
        )
        t1 = time.perf_counter()
        compress_times.append(t1 - t0)
    
    # Decompression measurements
    decompress_times = []
    for _ in range(measurement_runs):
        t0 = time.perf_counter()
        decompressed = cellz.decompress_cellz_to_csr(compressed_result)
        t1 = time.perf_counter()
        decompress_times.append(t1 - t0)
    
    # Verify correctness
    orig = csr.copy()
    orig.sort_indices()
    decomp = decompressed.copy()
    decomp.sort_indices()
    
    if not (orig.shape == decomp.shape and orig.nnz == decomp.nnz and
            np.array_equal(orig.indptr, decomp.indptr) and
            np.array_equal(orig.indices, decomp.indices) and
            np.array_equal(orig.data, decomp.data)):
        raise ValueError("CELLZ: Decompression verification failed!")
    
    compress_times = np.array(compress_times)
    decompress_times = np.array(decompress_times)
    
    return {
        "method": "CELLZ",
        "compressed_size": len(compressed_result),
        "original_size": raw_size,
        "compression_ratio": raw_size / len(compressed_result) if len(compressed_result) > 0 else float('inf'),
        "compress_time_mean": float(np.mean(compress_times)),
        "compress_time_std": float(np.std(compress_times, ddof=1)) if len(compress_times) > 1 else 0.0,
        "compress_time_min": float(np.min(compress_times)),
        "decompress_time_mean": float(np.mean(decompress_times)),
        "decompress_time_std": float(np.std(decompress_times, ddof=1)) if len(decompress_times) > 1 else 0.0,
        "decompress_time_min": float(np.min(decompress_times)),
        # Throughput based on original baseline bytes
        "compress_throughput_mib_s": (raw_size / 1048576.0) / float(np.mean(compress_times)) if float(np.mean(compress_times)) > 0 else 0.0,
        "decompress_throughput_mib_s": (raw_size / 1048576.0) / float(np.mean(decompress_times)) if float(np.mean(decompress_times)) > 0 else 0.0,
    }


def save_results_to_csv(results: List[Dict], metadata: Dict, output_path: str) -> None:
    """
    Save benchmark results to CSV with metadata.
    
    The CSV includes:
    - Metadata columns (input file, matrix dimensions, preprocessing, theoretical bounds, etc.)
    - Compression metrics for each algorithm
    - Bits per nonzero for each method
    - Compression efficiency relative to theoretical minimum
    - All compression ratios are calculated against the same baseline (raw serialized CSR)
    """
    df = pd.DataFrame(results)
    
    # Calculate bits per nonzero
    nnz = metadata.get("matrix_nnz", 1)
    df["bits_per_nnz"] = (df["compressed_size"] * 8) / nnz
    
    # Calculate compression efficiency (how close to theoretical minimum)
    theoretical_bpnz = metadata.get("theoretical_bpnz", 0)
    if theoretical_bpnz > 0:
        df["compression_efficiency_pct"] = (theoretical_bpnz / df["bits_per_nnz"]) * 100
    else:
        df["compression_efficiency_pct"] = 0.0
    
    # Add metadata columns
    for key, value in metadata.items():
        df[key] = value
    
    # Reorder columns - metadata first, then metrics
    meta_cols = list(metadata.keys())
    metric_cols = [
        "method", "compressed_size", "original_size", "compression_ratio", 
        "bits_per_nnz", "compression_efficiency_pct",
        "compress_time_mean", "compress_time_std", "compress_time_min",
        "decompress_time_mean", "decompress_time_std", "decompress_time_min",
        "compress_throughput_mib_s", "decompress_throughput_mib_s",
    ]
    df = df[meta_cols + metric_cols]
    
    # Sort by compressed size (best compression first)
    df = df.sort_values("compressed_size")
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"  - Baseline size: {metadata.get('original_size', 'N/A'):,} bytes")
    print(f"  - Naive bound: {theoretical_bpnz:.2f} bits/nnz (reference, can be beaten)")
    print(f"  - Best achieved: {df['bits_per_nnz'].min():.2f} bits/nnz ({df['method'].iloc[0]})")
    best_ratio = df['compression_efficiency_pct'].max()
    if best_ratio > 100:
        print(f"  - Efficiency: {best_ratio:.1f}% (BEATS naive bound via advanced encoding!)")
    else:
        print(f"  - Efficiency: {best_ratio:.1f}% of naive bound")


def main():
    parser = argparse.ArgumentParser(
        description="CELLZ compression for single-cell sparse matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic compression
  python compress_h5ad.py input.h5ad -o output.cellz
  
  # Compress log1p-transformed data by inverting to UMI counts
  python compress_h5ad.py normalized.h5ad -o output.cellz --invert-log1p
  
  # Compress with verification and comparison (CSV auto-generated)
  python compress_h5ad.py input.h5ad --verify --compare-algos
  
  # Stochastic integerization for float matrices
  python compress_h5ad.py float_data.h5ad --float-to-int
  
  # Rigorous benchmarking with custom CSV output
  python compress_h5ad.py input.h5ad --compare-algos --csv-output results.csv --benchmark-runs 5
  
  # Benchmarking without CSV output
  python compress_h5ad.py input.h5ad --compare-algos --no-csv
        """
    )
    
    parser.add_argument("inputs", nargs="+", type=str, help="Input .h5ad file path(s)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .cellz file path (single input only)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to place outputs for multiple inputs")
    parser.add_argument("--verify", action="store_true", help="Verify roundtrip compression")
    parser.add_argument("--compare-algos", action="store_true", help="Compare with baseline algorithms (gzip, bz2, lzma, etc.)")
    parser.add_argument("--invert-log1p", action="store_true", help="Invert log1p transformation before compression (for normalized data)")
    parser.add_argument("--float-to-int", action="store_true", help="Apply stochastic integerization to float data")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for stochastic rounding")
    
    # Benchmarking parameters
    parser.add_argument("--csv-output", type=str, default=None, help="Save benchmark results to CSV file (default: auto-generated from input filename)")
    parser.add_argument("--no-csv", action="store_true", help="Disable automatic CSV output")
    parser.add_argument("--benchmark-runs", type=int, default=1, help="Number of measurement runs for benchmarking (default: 1)")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warm-up runs before measurements (default: 1)")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of CPU cores for parallel compression (default: auto-detect)")
    
    # CELLZ compression parameters
    parser.add_argument("--target-nnz", type=int, default=131072, help="Target nonzeros per block for column tiling")
    parser.add_argument("--block-xz", action="store_true", help="Enable block-level xz compression for bitstreams")
    parser.add_argument("--block-xz-level", type=int, default=9, help="xz compression level (0-9)")
    parser.add_argument("--block-xz-min-bytes", type=int, default=4096, help="Minimum bytes threshold to attempt xz")
    parser.add_argument("--block-xz-min-ratio", type=float, default=0.99, help="Only keep xz if compressed < min_ratio * original")
    
    args = parser.parse_args()
    
    inputs = args.inputs
    multi_inputs = len(inputs) > 1
    if multi_inputs and args.output:
        print("Warning: -o/--output is ignored in multi-input mode; use --output-dir")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for input_path in inputs:
        # Resolve per-file output directory
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_dir_for_file = args.output_dir if args.output_dir else os.path.dirname(input_path)
        if out_dir_for_file:
            os.makedirs(out_dir_for_file, exist_ok=True)

        # Determine per-file output paths
        if (not multi_inputs) and args.output:
            output_path = args.output
        else:
            output_path = os.path.join(out_dir_for_file, base + ".cellz")

        # Determine per-file CSV path (only used in compare-algos and if CSV enabled)
        csv_path = None
        if args.compare_algos and not args.no_csv:
            if (not multi_inputs) and args.csv_output:
                csv_path = args.csv_output
            else:
                csv_path = os.path.join(out_dir_for_file, base + "_benchmark.csv")

        # Setup paths
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            continue
        
        # Load matrices
        X, raw_X = load_h5ad_matrices(input_path)
        
        matrices_to_process = [("X", X)]
        if raw_X is not None:
            print(f"Found raw.X matrix")
            matrices_to_process.append(("raw.X", raw_X))
        
        # Process each matrix
        processed_results = []
        for idx, (name, mat) in enumerate(matrices_to_process):
            result = process_matrix(mat, name, args, seed_offset=idx)
            processed_results.append(result)
        
        # Prepare metadata for CSV (will be updated with original_size later)
        n_cores = args.n_cores if args.n_cores else get_cpu_count()
        metadata = {
            "input_file": os.path.basename(input_path),
            "matrix_shape_rows": processed_results[0]["processed"].shape[0],
            "matrix_shape_cols": processed_results[0]["processed"].shape[1],
            "matrix_nnz": processed_results[0]["processed"].nnz,
            "matrix_type": processed_results[0]["type"],
            "preprocessing": "invert_log1p" if processed_results[0]["do_invert"] else "none",
            "n_cores": n_cores,
            "benchmark_runs": args.benchmark_runs,
            "warmup_runs": args.warmup_runs,
            "original_size": 0,  # Will be updated during benchmarking
        }
        
        # Benchmarking mode or simple compression mode
        if args.compare_algos:
            print("\n" + "="*60)
            print("RIGOROUS COMPRESSION BENCHMARKING")
            print("="*60)
            print(f"Matrix: {processed_results[0]['processed'].shape[0]} x {processed_results[0]['processed'].shape[1]}, nnz={processed_results[0]['processed'].nnz}")
            print(f"CPU cores: {n_cores}")
            print(f"Benchmark runs: {args.benchmark_runs}")
            print(f"Warmup runs: {args.warmup_runs}")
            
            # First, establish the baseline size (raw serialized CSR)
            csr_to_benchmark = processed_results[0]["processed"]
            raw_serialized = serialize_csr_to_bytes(csr_to_benchmark)
            raw_size = len(raw_serialized)
            
            # Compute theoretical lower bound
            theoretical_bound = compute_theoretical_lower_bound(csr_to_benchmark)
            
            print(f"\nBaseline (raw serialized CSR): {raw_size:,} bytes")
            print(f"  - Matrix shape: {csr_to_benchmark.shape[0]:,} x {csr_to_benchmark.shape[1]:,}")
            print(f"  - Nonzeros: {csr_to_benchmark.nnz:,}")
            print(f"  - Density: {csr_to_benchmark.nnz / (csr_to_benchmark.shape[0] * csr_to_benchmark.shape[1]) * 100:.4f}%")
            print(f"  - Bits per nonzero (uncompressed): {(raw_size * 8) / csr_to_benchmark.nnz:.2f}")
            
            print(f"\nNaive Theoretical Bound (reference only, NOT a strict lower bound):")
            print(f"  - Data entropy: {theoretical_bound['entropy_data_bits']:.2f} bits/nnz (global Shannon)")
            print(f"  - Structure entropy: {theoretical_bound['entropy_indices_bits']:.2f} bits/nnz (naive enumerative)")
            print(f"  - NAIVE BOUND: {theoretical_bound['theoretical_bpnz']:.2f} bits/nnz")
            print(f"    (Good compressors should BEAT this by exploiting delta coding, patterns, etc.)")
            
            # Benchmark CELLZ first (as requested)
            cellz_result = measure_cellz_compression(
                csr_to_benchmark,
                raw_size=raw_size,
                args=args,
                warmup_runs=args.warmup_runs,
                measurement_runs=args.benchmark_runs
            )
            bpnz = (cellz_result['compressed_size'] * 8) / csr_to_benchmark.nnz if csr_to_benchmark.nnz > 0 else 0
            print(f"    Size: {cellz_result['compressed_size']:,} bytes, Ratio: {cellz_result['compression_ratio']:.3f}x, Time: {cellz_result['compress_time_mean']*1000:.1f}ms, Bits/NNZ: {bpnz:.2f}, Comp: {cellz_result['compress_throughput_mib_s']:.1f} MiB/s, Decomp: {cellz_result['decompress_throughput_mib_s']:.1f} MiB/s")
            
            # Then benchmark baseline algorithms
            baseline_results, _ = compare_baseline_algos(
                csr_to_benchmark,
                n_cores=n_cores,
                warmup=args.warmup_runs,
                runs=args.benchmark_runs
            )
            
            # Combine all results (CELLZ first)
            all_results = [cellz_result] + baseline_results
            
            # Update metadata with baseline size and theoretical bound
            metadata["original_size"] = raw_size
            metadata["theoretical_bpnz"] = theoretical_bound["theoretical_bpnz"]
            metadata["entropy_data_bits"] = theoretical_bound["entropy_data_bits"]
            metadata["entropy_indices_bits"] = theoretical_bound["entropy_indices_bits"]
            metadata["row_overhead_bits"] = theoretical_bound["row_overhead_bits"]
            
            # Print summary table
            print("\n" + "="*90)
            print("BENCHMARK SUMMARY (sorted by compressed size)")
            print("="*90)
            print(f"Baseline: {raw_size:,} bytes (raw serialized CSR)")
            print(f"Naive bound: {theoretical_bound['theoretical_bpnz']:.2f} bits/nnz (reference, not a hard limit)")
            print(f"All compression ratios calculated against baseline for fair comparison")
            print("="*90)
            print(f"{'Method':<20} {'Size (bytes)':<15} {'Ratio':<10} {'Bits/NNZ':<12} {'Eff %':<10} {'Comp (ms)':<15} {'Decomp (ms)':<15}")
            print("-" * 105)
            nnz = processed_results[0]["processed"].nnz
            theoretical_min = theoretical_bound["theoretical_bpnz"]
            for r in sorted(all_results, key=lambda x: x['compressed_size']):
                comp_time = f"{r['compress_time_mean']*1000:.1f}±{r['compress_time_std']*1000:.1f}"
                decomp_time = f"{r['decompress_time_mean']*1000:.1f}±{r['decompress_time_std']*1000:.1f}"
                bits_per_nnz = (r['compressed_size'] * 8) / nnz if nnz > 0 else 0
                efficiency = (theoretical_min / bits_per_nnz * 100) if bits_per_nnz > 0 else 0
                print(f"{r['method']:<20} {r['compressed_size']:<15,} {r['compression_ratio']:<10.3f} {bits_per_nnz:<12.2f} {efficiency:<10.1f} {comp_time:<15} {decomp_time:<15}  C {r.get('compress_throughput_mib_s',0):>7.1f} MiB/s  D {r.get('decompress_throughput_mib_s',0):>7.1f} MiB/s")
            
            # Save to CSV (per-file path determined earlier)
            if csv_path:
                save_results_to_csv(all_results, metadata, csv_path)
            
            # Write CELLZ output file if path specified
            if (not multi_inputs) and args.output:
                with open(output_path, "wb") as f:
                    # Compress one final time for output
                    final_compressed = cellz.compress_csr_to_cellz(
                        processed_results[0]["processed"],
                        target_block_nnz=args.target_nnz,
                        block_xz=args.block_xz,
                        block_xz_level=args.block_xz_level,
                        block_xz_min_bytes=args.block_xz_min_bytes,
                        block_xz_min_ratio=args.block_xz_min_ratio,
                    )
                    f.write(final_compressed)
                print(f"\nCELLZ output written to: {output_path}")
        
        else:
            # Simple compression mode (no benchmarking)
            print("\n" + "="*60)
            print("Compressing with CELLZ...")
            print("="*60)
            
            t0 = time.perf_counter()
            
            if len(processed_results) == 1:
                # Single matrix
                compressed_bytes = cellz.compress_csr_to_cellz(
                    processed_results[0]["processed"],
                    target_block_nnz=args.target_nnz,
                    block_xz=args.block_xz,
                    block_xz_level=args.block_xz_level,
                    block_xz_min_bytes=args.block_xz_min_bytes,
                    block_xz_min_ratio=args.block_xz_min_ratio,
                )
            else:
                # Multiple matrices (not yet supported in current API, compress separately)
                # For now, just compress the main matrix
                print("Note: Multi-matrix compression not yet fully supported; compressing X only")
                compressed_bytes = cellz.compress_csr_to_cellz(
                    processed_results[0]["processed"],
                    target_block_nnz=args.target_nnz,
                    block_xz=args.block_xz,
                    block_xz_level=args.block_xz_level,
                    block_xz_min_bytes=args.block_xz_min_bytes,
                    block_xz_min_ratio=args.block_xz_min_ratio,
                )
            
            compression_time = time.perf_counter() - t0
            
            # Write output
            with open(output_path, "wb") as f:
                f.write(compressed_bytes)
            
            # Report results
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            ratio = input_size / output_size if output_size > 0 else float('inf')
            
            print(f"\n" + "="*60)
            print("Compression Results")
            print("="*60)
            print(f"Input file:  {input_path}")
            print(f"Output file: {output_path}")
            print(f"Input size:  {input_size:,} bytes")
            print(f"Output size: {output_size:,} bytes")
            print(f"Compression ratio: {ratio:.2f}x")
            print(f"Compression time: {compression_time*1000:.1f} ms")
            
            nnz_total = sum(r["processed"].nnz for r in processed_results)
            if nnz_total > 0:
                bits_per_nnz = (output_size * 8) / nnz_total
                print(f"Bits per nonzero: {bits_per_nnz:.2f}")
            
            # Verify if requested
            if args.verify:
                print("\n" + "="*60)
                print("Verification")
                print("="*60)
                
                # Decompress
                t0 = time.perf_counter()
                decompressed = cellz.decompress_cellz_to_csr(compressed_bytes)
                decompression_time = time.perf_counter() - t0
                print(f"Decompression time: {decompression_time*1000:.1f} ms")
                
                # Verify
                verify_roundtrip(processed_results[0]["processed"], decompressed, name=processed_results[0]["name"])
    
    print("\nDone!")


if __name__ == "__main__":
    main()
