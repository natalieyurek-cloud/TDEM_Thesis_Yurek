import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

"""
Parse a .usf file representing a station's TEM soundings, and put the results into a pandas dataframe.
This function's output dataframe will contain all variables for each entry, including the metadata 
details about the sweep.
"""
def parse_usf_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove the header section (starts with //)
    data_lines = [line.strip() for line in lines if not line.startswith("//") and line.strip()]

    all_rows = []
    current_sweep = {}
    data_buffer = []
    reading_data = False

    def save_current_sweep():
        """Helper to save the current sweep to all_rows."""
        nonlocal data_buffer, current_sweep
        if data_buffer:
            df_sweep = pd.DataFrame(data_buffer, columns=["TIME", "VOLTAGE", "QUALITY"])
            for k, v in current_sweep.items():
                df_sweep[k] = v
            all_rows.append(df_sweep)
            data_buffer = []

    for line in data_lines:
        # Start of a new sweep
        if line.startswith("/SWEEP_NUMBER:"):
            # Save the previous sweep before starting a new one
            save_current_sweep()
            # Reset metadata for new sweep
            current_sweep = {"SWEEP_NUMBER": int(line.split(":", 1)[1].strip())}
            reading_data = False
            continue
        
        # Read metadata lines starting with /
        if line.startswith("/"):
            # Metadata lines have a colon, other lines (like /END) do not
            if ":" in line:
                key, val = line[1:].split(":", 1)
                current_sweep[key.strip()] = val.strip()
            continue
        
        # Detect start of the numeric table
        if re.match(r"^\s*TIME", line, re.IGNORECASE):
            reading_data = True
            continue
        
        # Detect end of sweep
        if line.startswith("/END"):
            save_current_sweep()
            reading_data = False
            continue
        
        # If we are in a data section, parse numeric lines
        if reading_data:
            parts = re.split(r"[, ]+", line.strip())
            parts = [p for p in parts if p]
            if len(parts) >= 3:
                try:
                    time = float(parts[0])
                    voltage = float(parts[1])
                    quality = int(parts[2])
                    data_buffer.append([time, voltage, quality])
                except ValueError:
                    # Skip malformed lines
                    continue

    # Ensure final sweep is saved
    save_current_sweep()

    # Combine all sweeps
    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["TIME", "VOLTAGE", "QUALITY"])

    return df

"""
Return new dataframe with only the quality entries
"""
def filter_quality_one(df):
    if "QUALITY" not in df.columns:
        raise ValueError("DataFrame must contain a 'QUALITY' column.")
    return df[df["QUALITY"] == 1].copy()

"""
Calculates summary statistics for TIME and VOLTAGE columns.
Returns a 2D list in the format:
[
["Variable", "Mean", "Median", "Std", "Min", "Max"],
["TIME", ...],
["VOLTAGE", ...]
]
"""
def summarize_time_voltage(df, precision=6):
    stats = [["Variable", "Mean", "Median", "Std", "Min", "Max"]]

    for col in ["TIME", "VOLTAGE"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
        values = df[col].dropna()
        formatted = [f"{x:.{precision}g}" for x in 
            [ np.mean(values), 
             np.median(values), 
             np.std(values), 
             np.min(values), 
             np.max(values) 
            ]] 
        stats.append([col] + formatted)

    return stats

"""
Plots TIME vs VOLTAGE with error bars representing 1 standard deviation.
"""
def plot_time_voltage_error_by_bin(df, bins=30, log_bins=True):
    # Filter out invalid (<=0) times or voltages for log scale
    df = df[(df["TIME"] > 0) & (df["VOLTAGE"].notna())]

    # Create time bins
    if log_bins:
        bin_edges = np.logspace(np.log10(df["TIME"].min()), np.log10(df["TIME"].max()), bins + 1)
    else:
        bin_edges = np.linspace(df["TIME"].min(), df["TIME"].max(), bins + 1)

    # Assign each row to a time bin
    df["time_bin"] = pd.cut(df["TIME"], bins=bin_edges, include_lowest=True)

    # Compute mean and std of voltage within each bin
    grouped = df.groupby("time_bin").agg({
        "TIME": "mean",
        "VOLTAGE": ["mean", "std"]
    }).dropna()

    # Flatten multi-index columns
    grouped.columns = ["TIME_MEAN", "VOLTAGE_MEAN", "VOLTAGE_STD"]

    # Plot
    plt.figure(figsize=(8,6))
    plt.errorbar(
        grouped["TIME_MEAN"],
        grouped["VOLTAGE_MEAN"],
        yerr=grouped["VOLTAGE_STD"],
        fmt='o-',
        ecolor='gray',
        elinewidth=1,
        capsize=3,
        markersize=4,
        label="Mean ± 1 Std"
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs Time (Binned with Error Bars)")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.show()

###---------------------------------------------------------------------###
###                            DATA PROCESSING                          ###           
###---------------------------------------------------------------------###

# parse USF, input your specific file path in place of this one
df = parse_usf_file("D:/Project36/Station2/20260423_175421_378_Station2.usf")

#Original path example given C:/Users/yetfl/Space Grant/HoverTEM/USFParsing/20251009_095621_409_Station1.usf

# Sanity check of the columns
print(df.columns)

print('Parsing complete')

# output to excel
df.to_excel("output.xlsx", index=False)

print('Excel write complete')

# Create the scatterplot
plt.figure(figsize=(8,6))
plt.scatter(df["TIME"], df["VOLTAGE"], s=10, alpha=0.7)

# Apply logarithmic scales to both axes
plt.xscale('log')
plt.yscale('log')

# Label and style
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Voltage over Time")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.show()

print("Plot1 complete")

quality_df = filter_quality_one(df)

# Create the scatterplot
plt.figure(figsize=(8,6))
plt.scatter(quality_df["TIME"], quality_df["VOLTAGE"], s=10, alpha=0.7)

# Apply logarithmic scales to both axes
plt.xscale('log')
plt.yscale('log')

# Label and style
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Voltage over Time")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.show()

print("Plot2 complete")

summary = summarize_time_voltage(df)

for row in summary:
    print(row)

print("Summary stats complete")

non_noise_df = df[df["SWEEP_IS_NOISE"] == "0"].copy()

plot_time_voltage_error_by_bin(non_noise_df)

low_moment_df = non_noise_df[non_noise_df["CURRENT"].astype(float) < 2.0].copy()
high_moment_df = non_noise_df[non_noise_df["CURRENT"].astype(float) >= 2.0].copy()

plot_time_voltage_error_by_bin(low_moment_df)
plot_time_voltage_error_by_bin(high_moment_df)

###---------------------------------------------------------------------###
###             ORGANIZE DATA INTO SWEEP x TIME VOLTAGE MATRIX          ###
###---------------------------------------------------------------------###

# Keep only quality=1 and not flagged as noise
clean_df = df[(df["QUALITY"] == 1) & (df["SWEEP_IS_NOISE"] == "0")].copy()

# Ensure numeric types
clean_df["TIME"] = clean_df["TIME"].astype(float)
clean_df["VOLTAGE"] = clean_df["VOLTAGE"].astype(float)
clean_df["SWEEP_NUMBER"] = clean_df["SWEEP_NUMBER"].astype(int)

# Pivot: rows = sweeps, columns = time gates
sweep_matrix = clean_df.pivot_table(
    index="SWEEP_NUMBER",
    columns="TIME",
    values="VOLTAGE",
    aggfunc="mean"  # in case of duplicate times
)

# Sort by time axis
sweep_matrix = sweep_matrix.sort_index(axis=1)

print("Sweep x Time Matrix Created:")
print(sweep_matrix)

# Save to CSV for external tools (e.g., SimPEG)
sweep_matrix.to_csv("tem_sweep_matrix.csv")
print("Matrix exported to tem_sweep_matrix.csv")

# Heatmap for quick visualization
plt.figure(figsize=(10, 6))
plt.imshow(sweep_matrix, aspect='auto', interpolation='nearest')
plt.colorbar(label="Voltage (V)")
plt.title("Sweep vs. Time Voltage Matrix")
plt.xlabel("Time Index")
plt.ylabel("Sweep Number")
plt.show()

print("Matrix plot complete")

# Calculate the average voltage at each time gate across all sweeps
avg_voltage = sweep_matrix.mean(axis=0)  # mean across rows (sweeps)

# Convert to a clean DataFrame for export
avg_df = avg_voltage.reset_index()
avg_df.columns = ["time", "avg_voltage"]

# Save to CSV
avg_df.to_csv("average_voltage_by_time.csv", index=False)

print("Saved average voltage file: average_voltage_by_time.csv")






