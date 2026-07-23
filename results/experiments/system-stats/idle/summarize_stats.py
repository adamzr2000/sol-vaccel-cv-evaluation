import os
import csv
import json
import statistics
import glob


def _load_energy_totals(stem):
    """
    Load the energy_totals_<stem>.json sidecar the collector writes next to
    the CSV, if present. cpu and gpu are always monitored in separate
    sessions (see collect_idle_stats.sh), so each CSV has its own 1:1-named
    totals file - no candidate/fallback naming needed. Returns None for
    older runs predating this sidecar.
    """
    name = f"energy_totals_{stem}.json"
    if not os.path.exists(name):
        return None
    try:
        with open(name, "r") as f:
            return json.load(f)
    except Exception:
        return None


def generate_summary():
    output_file = "summary.csv"
    results = []

    # Find all CSV files in the current directory ending with 'idle.csv'
    csv_files = glob.glob("*-idle.csv")

    if not csv_files:
        print("No '-idle.csv' files found in the current directory.")
        return

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        # Use the filename (without extension) as the host identifier
        host = filename.replace(".csv", "")

        power_values = []
        timestamps_ms = []
        kind = None

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Detect the correct column based on whether it's a CPU or GPU file
                if 'cpu_watts' in row:
                    kind = 'cpu'
                    val = row['cpu_watts']
                elif 'power_draw_w' in row:
                    kind = 'gpu'
                    val = row['power_draw_w']
                else:
                    # If neither column exists, skip to the next row
                    continue

                try:
                    power_values.append(float(val))
                except ValueError:
                    # Skip rows with missing or malformed data
                    pass

                ts = row.get('timestamp')
                if ts:
                    try:
                        timestamps_ms.append(int(float(ts)))
                    except ValueError:
                        pass

        if not power_values:
            print(f"Warning: No valid power data found in {filename}")
            continue

        power_w_mean = statistics.mean(power_values)
        # stdev requires at least two data points
        power_w_std = statistics.stdev(power_values) if len(power_values) > 1 else 0.0

        totals = _load_energy_totals(host)

        # Exact energy total from the collector's hardware counters (RAPL for
        # cpu, NVML total-energy for gpu) - the authoritative figure.
        energy_j_from_counters = None
        if totals:
            if kind == 'cpu':
                energy_j_from_counters = totals.get('cpu_energy_j_total')
            elif kind == 'gpu' and totals.get('gpu'):
                vals = [g.get('gpu_energy_j_total') for g in totals['gpu'] if g.get('gpu_energy_j_total') is not None]
                if vals:
                    energy_j_from_counters = round(sum(vals), 3)

        # Monitored window: prefer the collector's own exact duration (same
        # window the counter total above covers); fall back to the CSV's own
        # first/last timestamp span if the sidecar is missing.
        duration_sec = None
        if totals and totals.get('duration_sec') is not None:
            duration_sec = totals['duration_sec']
        elif len(timestamps_ms) >= 2:
            duration_sec = round((max(timestamps_ms) - min(timestamps_ms)) / 1000.0, 3)

        # For comparison/sanity-checking only (log output + summary.csv) -
        # NOT used anywhere else in the project. Reconstructing energy this
        # way is biased by sampling interval/cadence; energy_j_from_counters
        # above is the one to actually use.
        energy_j_from_power = round(power_w_mean * duration_sec, 3) if duration_sec else None

        results.append([
            host,
            round(power_w_mean, 3),
            round(power_w_std, 3),
            duration_sec if duration_sec is not None else '',
            energy_j_from_power if energy_j_from_power is not None else '',
            energy_j_from_counters if energy_j_from_counters is not None else '',
        ])

    header = ['host', 'power_w_mean', 'power_w_std', 'duration_sec', 'energy_j_from_power', 'energy_j_from_counters']

    # Write out the results to summary.csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print(f"Success! Parsed {len(results)} files. Summary saved to {output_file}")

    # Print it to the console as well for a quick preview
    print("\n--- PREVIEW ---")
    print(",".join(header))
    for row in results:
        print(",".join(str(v) for v in row))

if __name__ == "__main__":
    generate_summary()
