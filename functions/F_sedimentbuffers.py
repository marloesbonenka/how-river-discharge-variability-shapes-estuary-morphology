"""Shared helper functions for sediment buffer postprocessing scripts."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from FUNCTIONS.F_loaddata import load_and_cache_scenario, get_stitched_his_paths
from FUNCTIONS.F_general import find_variability_model_folders


def tidal_avg(arr, window):
    """Centered moving average over `window` samples."""
    return np.convolve(arr, np.ones(window) / window, mode="same")


def load_sedimentbuffer_runs(
    base_path,
    cache_dir,
    discharge,
    variability_map,
    boxes,
    var_name,
    analyze_noisy,
    scenario_filter=None,
):
    """
    Load sediment-buffer time series for all matching run folders.

    Returns dict:
      folder_name -> {
          'time': array,
          'buffers': {(start, end): array},
      }
    """
    base_path = Path(base_path)
    cache_dir = Path(cache_dir)

    timed_out_dir = base_path / "timed-out"
    if not timed_out_dir.exists():
        timed_out_dir = None

    folders = find_variability_model_folders(
        base_path=base_path,
        discharge=discharge,
        scenarios_to_process=scenario_filter,
        analyze_noisy=analyze_noisy,
    )

    cache_dir.mkdir(exist_ok=True)
    runs = {}
    for folder in folders:
        his_paths = get_stitched_his_paths(
            base_path=base_path,
            folder_name=folder,
            timed_out_dir=timed_out_dir,
            variability_map=variability_map,
            analyze_noisy=analyze_noisy,
        )
        if not his_paths:
            print(f"[WARNING] No HIS files found for {folder.name}, skipping.")
            continue

        scenario_num = folder.name.split("_")[0]
        run_id = "_".join(folder.name.split("_")[1:])
        cache_file = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

        _, data = load_and_cache_scenario(
            scenario_dir=folder,
            his_file_paths=his_paths,
            cache_file=cache_file,
            boxes=boxes,
            var_name=var_name,
        )

        runs[folder.name] = {
            "time": data["t"],
            "buffers": data["buffer_volumes"],
        }

    return runs


def align_runs_to_common_time(base_time, runs_dict, box_key):
    """
    Interpolate all runs in runs_dict onto base_time for one box key.

    Returns array of shape (n_runs, len(base_time)).
    """
    aligned = []
    for run_data in runs_dict.values():
        t = run_data["time"]
        buf = run_data["buffers"].get(box_key)
        if buf is None:
            continue

        if np.issubdtype(t.dtype, np.datetime64):
            t_f = t.astype("datetime64[ns]").astype(np.float64)
            t_base_f = base_time.astype("datetime64[ns]").astype(np.float64)
        else:
            t_f = t.astype(np.float64)
            t_base_f = base_time.astype(np.float64)

        aligned.append(np.interp(t_base_f, t_f, buf))

    return np.array(aligned)


def trim_to_end(time_array, t_end_min):
    """Boolean mask for values at or before the shared simulation end time."""
    return time_array <= t_end_min

def run_envelope_workflow(
    base_directory,
    config,
    discharge,
    output_dirname,
    boxes,
    sed_var,
    variability_map,
    scenario_labels,
    scenario_colors,
    base_scenario='1',
    envelope_plot_mode='all',
    show_noisy_envelope=True,
):
    """Run noisy-envelope plots for sediment buffer volumes."""
    # Derive scenario keys from variability_map so all runs are included
    _all_keys = sorted({str(int(k)) for k in variability_map}, key=lambda x: int(x))
    _auto_colors = ["#56B4E9", "#E69F00", "#009E73", "#D55E00",
                    "#CC79A7", "#0072B2", "#F0E442", "#000000"]
    scenario_config = {
        k: {
            "color": (scenario_colors or {}).get(k,
                      _auto_colors[i % len(_auto_colors)]),
            "label": (scenario_labels or {}).get(k, f"Scenario {k}"),
        }
        for i, k in enumerate(_all_keys)
    }

    base_directory = Path(base_directory)
    variability_base_path = base_directory / config
    variability_cache_dir = variability_base_path / "cached_data"
    noisy_base_path = variability_base_path / f"0_Noise_Q{discharge}"
    noisy_cache_dir = noisy_base_path / "cached_data"

    valid_modes = {'noise_only', 'all', 'scenarios_only'}
    if envelope_plot_mode not in valid_modes:
        raise ValueError(
            f"Unknown envelope_plot_mode='{envelope_plot_mode}'. "
            f"Choose one of {sorted(valid_modes)}."
        )
    if envelope_plot_mode == 'noise_only' and not show_noisy_envelope:
        raise ValueError("show_noisy_envelope=False is incompatible with envelope_plot_mode='noise_only'.")

    if envelope_plot_mode == 'noise_only':
        envelope_output_dir = noisy_base_path / "output_plots" / output_dirname
    else:
        envelope_output_dir = variability_base_path / "output_plots" / output_dirname
    envelope_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Envelope output dir: {envelope_output_dir}")

    base_runs = load_sedimentbuffer_runs(
        base_path=variability_base_path,
        cache_dir=variability_cache_dir,
        discharge=discharge,
        variability_map=variability_map,
        boxes=boxes,
        var_name=sed_var,
        analyze_noisy=False,
        scenario_filter={int(base_scenario)},
    )
    if not base_runs:
        raise FileNotFoundError(
            f"No run found for base scenario {base_scenario} in {variability_base_path}."
        )

    base_data = list(base_runs.values())[0]
    base_time = base_data['time']
    base_buffers = base_data['buffers']
    base_cfg = scenario_config[base_scenario]

    variability_runs = {}
    if envelope_plot_mode in {'all', 'scenarios_only'}:
        all_var_runs = load_sedimentbuffer_runs(
            base_path=variability_base_path,
            cache_dir=variability_cache_dir,
            discharge=discharge,
            variability_map=variability_map,
            boxes=boxes,
            var_name=sed_var,
            analyze_noisy=False,
            scenario_filter=None,
        )
        # Extend scenario_config with any scenarios found on disk not in variability_map
        for folder_name in sorted(all_var_runs, key=lambda x: int(x.split('_')[0])):
            snum = str(int(folder_name.split('_')[0]))
            if snum not in scenario_config:
                scenario_config[snum] = {
                    'color': (scenario_colors or {}).get(snum,
                              _auto_colors[len(scenario_config) % len(_auto_colors)]),
                    'label': (scenario_labels or {}).get(snum, f'Scenario {snum}'),
                }
        _all_keys = sorted(scenario_config, key=lambda x: int(x))

        for folder_name, run_data in all_var_runs.items():
            scenario_num = str(int(folder_name.split('_')[0]))
            if scenario_num == base_scenario:
                continue
            variability_runs[scenario_num] = {
                'time': run_data['time'],
                'buffers': run_data['buffers'],
                'label': scenario_config[scenario_num]['label'],
                'color': scenario_config[scenario_num]['color'],
            }

    noisy_runs = {}
    if show_noisy_envelope and envelope_plot_mode in {'all', 'noise_only'} and noisy_base_path.exists():
        noisy_runs = load_sedimentbuffer_runs(
            base_path=noisy_base_path,
            cache_dir=noisy_cache_dir,
            discharge=discharge,
            variability_map=variability_map,
            boxes=boxes,
            var_name=sed_var,
            analyze_noisy=True,
            scenario_filter=None,
        )
    elif show_noisy_envelope and envelope_plot_mode in {'all', 'noise_only'}:
        noisy_runs = {}
        print(f"[INFO] No noisy base path found: {noisy_base_path}")

    print(f"Found {len(noisy_runs)} noisy runs")

    if envelope_plot_mode == 'noise_only' and not noisy_runs:
        raise FileNotFoundError(
            f"No noisy runs found in {noisy_base_path}. "
            f"Set envelope_plot_mode='all' or add noisy runs."
        )
    if envelope_plot_mode == 'all' and show_noisy_envelope and not noisy_runs:
        print("[INFO] No noisy runs found; plotting base + variability scenarios without noisy envelope.")

    all_times = [base_time]
    all_times += [r['time'] for r in noisy_runs.values()]
    all_times += [r['time'] for r in variability_runs.values()]
    t_end_min = min(t[-1] for t in all_times)
    print(f"Shortest simulation end time across all runs: {t_end_min}")
    line_width = 1.6

    ref_time = base_time[0]

    def _to_morph_time(time_array):
        """Convert time to morphological-time axis: (hydrodynamic year + 1) * 100."""
        t = np.asarray(time_array)
        if np.issubdtype(t.dtype, np.datetime64):
            hydro_years = (t - ref_time) / np.timedelta64(365, 'D')
            return (hydro_years + 1.0) * 100.0

        # Fallback for numeric time arrays; assumes values are in hydrodynamic years.
        t0 = np.asarray(ref_time).astype(np.float64)
        hydro_years = t.astype(np.float64) - t0
        return (hydro_years + 1.0) * 100.0

    for box_key in boxes:
        box_start, box_end = box_key
        fig, ax = plt.subplots()

        if show_noisy_envelope and noisy_runs:
            base_time_trimmed = base_time[trim_to_end(base_time, t_end_min)]
            base_time_trimmed_morph = _to_morph_time(base_time_trimmed)
            noisy_stack = align_runs_to_common_time(base_time_trimmed, noisy_runs, box_key)

            base_buf_for_env = base_buffers.get(box_key)
            if base_buf_for_env is not None:
                base_row = base_buf_for_env[trim_to_end(base_time, t_end_min)][np.newaxis, :]
                if noisy_stack.size > 0:
                    noisy_stack = np.vstack([noisy_stack, base_row])
                else:
                    noisy_stack = base_row

            if noisy_stack.size > 0:
                env_min = np.nanmin(noisy_stack, axis=0)
                env_max = np.nanmax(noisy_stack, axis=0)

                for i, run_data in enumerate(noisy_runs.values()):
                    buf = run_data['buffers'].get(box_key)
                    if buf is None:
                        continue
                    mask = trim_to_end(run_data['time'], t_end_min)
                    t_morph = _to_morph_time(run_data['time'][mask])
                    ax.plot(
                        t_morph,
                        buf[mask],
                        color='grey',
                        alpha=0.35,
                        linewidth=line_width,
                        label='Noisy runs' if i == 0 else None,
                    )

                ax.fill_between(
                    base_time_trimmed_morph,
                    env_min,
                    env_max,
                    color='grey',
                    alpha=0.2,
                    label='Noisy envelope',
                )

        if envelope_plot_mode in {'all', 'scenarios_only'}:
            for run_data in variability_runs.values():
                buf = run_data['buffers'].get(box_key)
                if buf is None:
                    continue
                mask = trim_to_end(run_data['time'], t_end_min)
                t_morph = _to_morph_time(run_data['time'][mask])
                ax.plot(
                    t_morph,
                    buf[mask],
                    color=run_data['color'],
                    linewidth=line_width,
                    label=run_data['label'],
                )

        base_buf = base_buffers.get(box_key)
        if base_buf is not None:
            mask = trim_to_end(base_time, t_end_min)
            t_morph = _to_morph_time(base_time[mask])
            ax.plot(
                t_morph,
                base_buf[mask],
                color=base_cfg['color'],
                linewidth=line_width,
                label=base_cfg['label'],
            )

        ax.set_xlabel('time [years]')
        ax.set_ylabel('sediment buffer volume [m$^3$]')
        ax.set_title(f'{box_start}-{box_end}km')

        handles, labels = ax.get_legend_handles_labels()
        desired_order = [
            scenario_config[k]['label'] for k in _all_keys
        ]
        ordered_handles = []
        ordered_labels = []
        for name in desired_order:
            if name in labels:
                idx = labels.index(name)
                ordered_handles.append(handles[idx])
                ordered_labels.append(labels[idx])

        # Keep any non-scenario legend entries (e.g., noisy context) at the end.
        for h, l in zip(handles, labels):
            if l not in ordered_labels:
                ordered_handles.append(h)
                ordered_labels.append(l)

        ax.legend(ordered_handles, ordered_labels)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = envelope_output_dir / f"Q{discharge}_sedimentbuffer_box_{box_start}_{box_end}km.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.show()