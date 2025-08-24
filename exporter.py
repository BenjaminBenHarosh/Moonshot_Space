from __future__ import annotations
from typing import Dict, Any, List
import os
import json
import gzip
import pandas as pd


# ---- utilities --------------------------------------------------------------

def _ensure_parent_dir(path: str) -> None:
    """Create parent directories if they don't exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _to_dataframe(result) -> pd.DataFrame:
    """
    Convert simulation result to pandas DataFrame with standard schema.

    Accepts:
      - SimulationResult (from simulator.py) with .to_dataframe() method
      - dict with keys time/position/velocity/acceleration
      - pandas DataFrame already in correct schema
    """
    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()

    if isinstance(result, pd.DataFrame):
        required_cols = ["time", "position", "velocity", "acceleration"]
        return result[required_cols].copy()

    # Assume dict-like interface
    return pd.DataFrame({
        "time": result["time"],
        "position": result["position"],
        "velocity": result["velocity"],
        "acceleration": result["acceleration"],
    })


def _get_events_and_metadata(result) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract events and metadata from simulation result."""
    events = []
    metadata = {}

    # Try SimulationResult attributes
    if hasattr(result, "events"):
        events = getattr(result, "events", [])
    if hasattr(result, "metadata"):
        metadata = getattr(result, "metadata", {})

    # Fallback to dict-like access
    if isinstance(result, dict):
        events = result.get("events", events)
        metadata = result.get("metadata", metadata)

    return events, metadata


# ---- public API -------------------------------------------------------------

def export_timeseries(
        result,
        base_path: str,
        fmt: str = "parquet",
        *,
        json_compress: bool = True,
        parquet_compression: str = "snappy",
        hdf_key_timeseries: str = "timeseries",
        hdf_key_events: str = "coil_events",
) -> str:
    """
    Export simulation time series to structured file format.

    Args:
        result: SimulationResult object or compatible dict/DataFrame
        base_path: Base path for output file (extension added if needed)
        fmt: Output format - 'json', 'json.gz', 'parquet', or 'hdf5'
        json_compress: Whether to compress JSON output
        parquet_compression: Compression algorithm for Parquet
        hdf_key_timeseries: HDF5 key for main timeseries data
        hdf_key_events: HDF5 key for events data

    Returns:
        str: Path to the main output file
    """
    fmt = fmt.lower()
    df = _to_dataframe(result)
    events, metadata = _get_events_and_metadata(result)

    # Auto-compress JSON if requested
    if fmt == "json" and json_compress:
        fmt = "json.gz"

    if fmt == "json.gz":
        path = base_path if base_path.endswith(".json.gz") else base_path + ".json.gz"
        _ensure_parent_dir(path)

        payload = {
            "timeseries": {
                "time": df["time"].tolist(),
                "position": df["position"].tolist(),
                "velocity": df["velocity"].tolist(),
                "acceleration": df["acceleration"].tolist(),
            },
            "events": events,
            "metadata": metadata,
        }

        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        return path

    elif fmt == "json":
        path = base_path if base_path.endswith(".json") else base_path + ".json"
        _ensure_parent_dir(path)

        payload = {
            "timeseries": {
                "time": df["time"].tolist(),
                "position": df["position"].tolist(),
                "velocity": df["velocity"].tolist(),
                "acceleration": df["acceleration"].tolist(),
            },
            "events": events,
            "metadata": metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    elif fmt == "parquet":
        path = base_path if base_path.endswith(".parquet") else base_path + ".parquet"
        _ensure_parent_dir(path)

        # Export main timeseries
        df.to_parquet(path, compression=parquet_compression, index=False)

        # Export events as separate parquet if they exist
        if events:
            events_path = path.replace(".parquet", "__events.parquet")
            events_df = pd.DataFrame(events)
            events_df.to_parquet(events_path, compression=parquet_compression, index=False)

        # Export metadata as JSON sidecar
        if metadata:
            metadata_path = path.replace(".parquet", "__metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        return path

    elif fmt == "hdf5":
        path = base_path if base_path.endswith(".h5") else base_path + ".h5"
        _ensure_parent_dir(path)

        with pd.HDFStore(path, mode="w", complevel=9, complib="blosc") as store:
            # Store main timeseries data
            store.put(hdf_key_timeseries, df, format="table", data_columns=True)

            # Store events if they exist
            if events:
                events_df = pd.DataFrame(events)
                store.put(hdf_key_events, events_df, format="table", data_columns=True)

            # Store metadata as attributes
            if metadata:
                store.get_storer(hdf_key_timeseries).attrs.metadata_json = json.dumps(metadata)

        return path

    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'json', 'json.gz', 'parquet', or 'hdf5'")


def load_timeseries(path: str) -> Dict[str, Any]:
    """
    Load simulation results from exported file.

    Args:
        path: Path to the exported file

    Returns:
        dict: Loaded data with keys 'timeseries', 'events', 'metadata'
    """
    path_lower = path.lower()

    if path_lower.endswith('.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    elif path_lower.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    elif path_lower.endswith('.parquet'):
        # Load main timeseries
        df = pd.read_parquet(path)
        result = {"timeseries": df.to_dict('list'), "events": [], "metadata": {}}

        # Try to load events sidecar
        events_path = path.replace('.parquet', '__events.parquet')
        if os.path.exists(events_path):
            events_df = pd.read_parquet(events_path)
            result["events"] = events_df.to_dict('records')

        # Try to load metadata sidecar
        metadata_path = path.replace('.parquet', '__metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                result["metadata"] = json.load(f)

        return result

    elif path_lower.endswith('.h5'):
        result = {"timeseries": {}, "events": [], "metadata": {}}

        with pd.HDFStore(path, mode='r') as store:
            # Load timeseries
            if 'timeseries' in store:
                df = store['timeseries']
                result["timeseries"] = df.to_dict('list')

                # Try to load metadata from attributes
                try:
                    metadata_json = store.get_storer('timeseries').attrs.metadata_json
                    result["metadata"] = json.loads(metadata_json)
                except (AttributeError, KeyError):
                    pass

            # Load events if they exist
            if 'coil_events' in store:
                events_df = store['coil_events']
                result["events"] = events_df.to_dict('records')

        return result

    else:
        raise ValueError(f"Unsupported file format for path: {path}")