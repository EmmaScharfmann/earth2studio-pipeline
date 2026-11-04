"""Seasonal evaluation of AI ensemble forecasts.

Everything here runs in the BASE environment (numpy / xarray / pandas, no
torch): forecasts are produced by the per-model environments through
scripts/run_benchmark.py, and this package only reads the zarr stores they
write.
"""
