import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches


# Simulation setup
shape = (256, 64)

spu_params = {
    "mean": 10e3,
    "sigma_pct": 0.05,
    "gradient_x_pct": 0.05,
    "gradient_y_pct": 0.07
}

spd_params = {
    "mean": 10.5e3,
    "sigma_pct": 0.07,
    "gradient_x_pct": 0.10,
    "gradient_y_pct": 0.08
}

spu_params_no_variation = {
    "mean": 10e3,
    "sigma_pct": 0.0,
    "gradient_x_pct": 0.0,
    "gradient_y_pct": 0.0
}

spd_params_no_variation = {
    "mean": 10.5e3,
    "sigma_pct": 0.0,
    "gradient_x_pct": 0.0,
    "gradient_y_pct": 0.0
}

# Generate a 2D resistor matrix with Gaussian noise and optional gradients in X and Y directions
def generate_resistor_matrix(shape, mean, sigma_pct, gradient_x_pct, gradient_y_pct):
    rows, cols = shape
    gradient_x = np.linspace(0, 1, cols) * gradient_x_pct
    gradient_y = np.linspace(0, 1, rows)[:, None] * gradient_y_pct
    gradients = gradient_x + gradient_y
    noise = np.random.normal(loc=mean, scale=mean * sigma_pct, size=(rows, cols))
    variation = (1 + gradient_x) * (1 + gradient_y)
    return variation * noise

# Compute the equivalent resistance of resistors connected in parallel
def parallel_resistance_sum(resistances):
    return 1 / np.sum(1 / resistances)

# Simulate the VRBL outputs of an SRAM-based IMC system for a given input and weight matrix
def simulate_sram_imc(weight_matrix, input_matrix, spu_params, spd_params, vdd=1.0):
    rows, cols = weight_matrix.shape
    assert input_matrix.shape == weight_matrix.shape, "Weight and input shape mismatch"

    SPU = generate_resistor_matrix(weight_matrix.shape, **spu_params)
    SPD = generate_resistor_matrix(weight_matrix.shape, **spd_params)
    WPU = generate_resistor_matrix(weight_matrix.shape, **spu_params)
    WPD = generate_resistor_matrix(weight_matrix.shape, **spd_params)

    vrbls = []
    for col in range(cols):
        total_pu = []
        total_pd = []
        for row in range(rows):
            act = input_matrix[row, col]
            wt = weight_matrix[row, col]

            if act == 0:
                if row % 2 == 0:
                    total_pu.append(SPU[row, col])
                    total_pd.append(SPD[row, col])
                else:
                    total_pu.append(SPU[row, col])
                    total_pd.append(SPD[row, col])
            elif act == 1:
                if wt == 1:
                    total_pu.append(SPU[row, col])
                elif wt == -1:
                    total_pd.append(SPD[row, col])
            elif act == -1:
                if wt == 1:
                    total_pd.append(SPD[row, col])
                elif wt == -1:
                    total_pu.append(SPU[row, col])

        RPU_eq = parallel_resistance_sum(np.array(total_pu)) if total_pu else 1e9
        RPD_eq = parallel_resistance_sum(np.array(total_pd)) if total_pd else 1e9
        vrbl = vdd * RPD_eq / (RPD_eq + RPU_eq)
        vrbls.append(vrbl)

    return np.array(vrbls)

# Quantize the VRBL values into discrete ADC levels using uniform step size
def adc_quantize(vrbls, vdd=1.0, levels=12):
    step = vdd / (levels - 1)
    return np.clip(np.round(vrbls / step), 0, levels - 1).astype(int)

# Redistribute a list of ternary input values (1, 0, -1) in a balanced manner to reduce PU/PD variations
def redistribute_inputs_balanced(inputs):
    left_pointer = 0
    right_pointer = len(inputs) - 1
    middle_pointer = len(inputs) // 2
    middle_list = []

    positions_by_value = defaultdict(list)
    for val in inputs:
        positions_by_value[val].append(val)

    result = [None] * len(inputs)
    for val in [1, -1, 0]:
        values = positions_by_value[val]
        if not values:
            continue

        n = len(values)
        i = n - 1
        while i > 0:
            result[left_pointer] = values[i]
            result[right_pointer] = values[i-1]
            i = i - 2
            left_pointer += 1
            right_pointer -= 1

        if i == 0:
            middle_list.append(val)

    for i, middle in enumerate(middle_list):
        result[middle_pointer] = middle
        if i == 0:
            if result[middle_pointer + 1] is None:
                middle_pointer += 1
            else:
                middle_pointer -= 1
        else:
            if result[middle_pointer + 2] is None:
                middle_pointer += 2
            else:
                middle_pointer -= 2

    return result

# Perform column wise balancing of input activations to reduce variation effects
def pre_processing_inputs(weight_matrix, input_matrix):
    rows, cols = weight_matrix.shape
    processed_inputs = input_matrix.copy()

    for col in range(cols):
        col_weights = weight_matrix[:, col]
        col_inputs = input_matrix[:, col]

        for weight_value in [1, -1]:
            same_weight_indices = np.where(col_weights == weight_value)[0]
            if len(same_weight_indices) > 1:
                inputs_to_shuffle = col_inputs[same_weight_indices]
                redistributed_inputs = redistribute_inputs_balanced(inputs_to_shuffle)[::-1]
                for idx, new_val in zip(same_weight_indices, redistributed_inputs):
                    processed_inputs[idx, col] = new_val

    return processed_inputs

# Run a full simulation including input redistribution, SRAM IMC evaluation,
# ADC quantization, and correction error estimation
def run_simulation(seed, shape, spu_params, spd_params, spu_params_no_variation, spd_params_no_variation, vdd=1.0):
    np.random.seed(seed)

    inputs = np.ones(shape)
    weights = np.empty(shape, dtype=int)
    weights[::2, :] = 1  # Even-indexed rows: 0, 2, 4, ...
    weights[1::2, :] = -1  # Odd-indexed rows: 1, 3, 5, ...

    weights = np.random.choice([1, -1], size=shape)
    inputs = np.random.choice([1, 0, -1], size=shape)

    inputs_raw = inputs
    inputs_balanced = pre_processing_inputs(weights, inputs)

    # Simulations
    vrbls_raw = simulate_sram_imc(weights, inputs_raw, spu_params, spd_params)
    vrbls_balanced = simulate_sram_imc(weights, inputs_balanced, spu_params, spd_params)
    vrbls_balanced_ideal = simulate_sram_imc(weights, inputs_balanced, spu_params_no_variation, spd_params_no_variation)
    vrbls_ideal = simulate_sram_imc(weights, inputs, spu_params_no_variation, spd_params_no_variation)

    # True ideal for X learning
    ideal_spu_params_true = {"mean": 10e3, "sigma_pct": 0.0, "gradient_x_pct": 0.0, "gradient_y_pct": 0.0}
    ideal_spd_params_true = {"mean": 10e3, "sigma_pct": 0.0, "gradient_x_pct": 0.0, "gradient_y_pct": 0.0}
    vrbls_ideal_true = simulate_sram_imc(weights, inputs, ideal_spu_params_true, ideal_spd_params_true)

    # ADC quantization
    adc_raw = adc_quantize(vrbls_raw)
    adc_balanced = adc_quantize(vrbls_balanced)
    adc_ideal = adc_quantize(vrbls_ideal)

    # Difference metrics
    abs_diff_vrbl_raw = np.mean(np.abs(vrbls_raw - vrbls_ideal))
    abs_diff_vrbl_balanced = np.mean(np.abs(vrbls_balanced - vrbls_ideal))
    abs_diff_adc_raw = np.mean(np.abs(adc_raw - adc_ideal))
    abs_diff_adc_balanced = np.mean(np.abs(adc_balanced - adc_ideal))

    std_vrbl_raw = np.std(vrbls_raw)
    std_vrbl_balanced = np.std(vrbls_balanced)
    std_vrbl_ideal = np.std(vrbls_ideal)

    # Learn X from "ideal" setup (variation between SPU and SPD)
    X = learning_X(shape, spu_params, spd_params, vdd=1.0)

    # Correct raw VRBL using X and formula
    correction_factor = (vdd / vrbls_raw - 1) / X
    vrbls_corrected = vdd / (1 + correction_factor)

    # Compare corrected to true ideal
    correction_error = np.mean(np.abs(vrbls_corrected - vrbls_ideal_true))

    return (
        abs_diff_vrbl_raw,
        abs_diff_vrbl_balanced,
        abs_diff_adc_raw,
        abs_diff_adc_balanced,
        std_vrbl_raw,
        std_vrbl_balanced,
        std_vrbl_ideal,
        correction_error
    )

# Generate multiple weight matrix patterns with a fixed number of ones per column.
# Supports deterministic (top, bottom, middle) and random distribution of '1's for multiple simulation runs.
def create_weight_patterns(num_ones, total_rows, total_cols, runs_per_config=10):
    patterns = []
    for run in range(runs_per_config):
        matrix = np.full((total_rows, total_cols), -1)
        if num_ones == 0:
            patterns.append(matrix)
            continue
        for col in range(total_cols):
            if run == 0:
                matrix[:num_ones, col] = 1
            elif run == 1:
                matrix[-num_ones:, col] = 1
            elif run == 2:
                start = (total_rows - num_ones) // 2
                matrix[start:start + num_ones, col] = 1
            else:
                ones_indices = np.random.choice(total_rows, num_ones, replace=False)
                matrix[:, col] = -1
                matrix[ones_indices, col] = 1
        patterns.append(matrix)
    return patterns


# Run the simulation 100 times with randomized inputs and weights.
# Compare raw vs balanced inputs against an ideal reference and display metrics and plots
def run_summary_simulation_comparison(shape, spu_params, spd_params, spu_params_no_variation, spd_params_no_variation):
    results = []
    for seed in range(100):
        result = run_simulation(seed, shape, spu_params, spd_params, spu_params_no_variation, spd_params_no_variation)
        results.append(result)

    results = np.array(results)
    mean_results = np.mean(results[:, :4], axis=0)  # Means for difference metrics
    std_results = np.std(results[:, :4], axis=0)

    mean_std_vrbls = np.mean(results[:, 4:7], axis=0)  # [raw, balanced, ideal]
    correction_errors = results[:, 7]
    correction_error_mean = np.mean(correction_errors)
    correction_error_std = np.std(correction_errors)

    print("=== Average over 20 Simulations ===")
    print(f"Avg Absolute VRBL Diff (Raw vs Ideal):      {mean_results[0]:.6f} V ± {std_results[0]:.6f}")
    print(f"Avg Absolute VRBL Diff (Balanced vs Ideal): {mean_results[1]:.6f} V ± {std_results[1]:.6f}")
    print(f"Avg Absolute ADC Diff (Raw vs Ideal):       {mean_results[2]:.6f} ± {std_results[2]:.6f}")
    print(f"Avg Absolute ADC Diff (Balanced vs Ideal):  {mean_results[3]:.6f} ± {std_results[3]:.6f}")
    print()
    print(f"Std of VRBL outputs:")
    print(f"Raw Input:      {mean_std_vrbls[0]:.6f} V")
    print(f"Balanced Input: {mean_std_vrbls[1]:.6f} V")
    print(f"Ideal (No Var): {mean_std_vrbls[2]:.6f} V")
    print()

    better_vrbl = "Balanced" if mean_results[1] < mean_results[0] else "Raw"
    better_adc = "Balanced" if mean_results[3] < mean_results[2] else "Raw"

    print(f"\nBetter VRBL method (closer to ideal): {better_vrbl}")
    print(f"Better ADC method  (closer to ideal): {better_adc}")

    # --- Bar plot for Difference ---
    labels = ['VRBL Difference', 'ADC Difference']
    raw_diffs = [mean_results[0], mean_results[2]]
    balanced_diffs = [mean_results[1], mean_results[3]]
    raw_stds = [std_results[0], std_results[2]]
    balanced_stds = [std_results[1], std_results[3]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, raw_diffs, width, label='Raw Input', yerr=raw_stds, capsize=5)
    bars2 = ax.bar(x + width/2, balanced_diffs, width, label='Balanced Input', yerr=balanced_stds, capsize=5)

    ax.set_ylabel('Average Absolute Difference')
    ax.set_title('Raw vs Balanced Input: Mean ± STD (VRBL and ADC)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # --- Plot Standard Deviation of Analog Outputs ---
    plt.figure(figsize=(6, 4))
    std_labels = ['Raw', 'Balanced', 'Ideal']
    plt.bar(std_labels, mean_std_vrbls, color=['lightblue', 'lightgreen', 'lightgray'])
    plt.ylabel('VRBL Output Std (V)')
    plt.title('Standard Deviation of Analog Outputs (VRBL)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- Plot Correction Error Bar ---
    plt.figure(figsize=(5, 4))
    plt.bar(["Corrected Raw"], [correction_error_mean], yerr=[correction_error_std], capsize=8, color="lightcoral")
    plt.ylabel("Mean Abs Error (V)")
    plt.title("Correction Error: Raw → Corrected vs True Ideal")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Plot the distribution of analog outputs (VRBLs) for different numbers of ones per column
# across multiple weight patterns. Also plot summary statistics: mean, std, range/mean.
def plot_vrbls_distribution(shape, spu_params, spd_params):
    input_matrix = np.ones(shape)
    vrbls_outputs_by_num_ones = {}

    for num_ones in range(0, 257, 16):
        patterns = create_weight_patterns(num_ones, *shape)
        if num_ones == 16:
            with open("patterns_16_ones.txt", "w") as f:
                for idx, matrix in enumerate(patterns):
                    f.write(f"--- Pattern {idx} ---\n")
                    for row in matrix:
                        row_str = " ".join(str(int(val)) for val in row)
                        f.write(row_str + "\n")
                    f.write("\n")
        vrbls_runs = []
        for weight_matrix in patterns:
            vrbls = simulate_sram_imc(weight_matrix, input_matrix, spu_params, spd_params)
            vrbls_runs.append(vrbls.flatten())
        vrbls_outputs_by_num_ones[num_ones] = np.concatenate(vrbls_runs)

    labels, data, medians, means, stds, ratios = [], [], [], [], [], []

    for num_ones, vrbls in vrbls_outputs_by_num_ones.items():
        labels.append(num_ones)
        data.append(vrbls)
        medians.append(np.median(vrbls))
        means.append(np.mean(vrbls))
        stds.append(np.std(vrbls))
        ratios.append((np.max(vrbls) - np.min(vrbls)) / np.mean(vrbls) if np.mean(vrbls) != 0 else 0)

    plt.figure(figsize=(14, 6))
    plt.boxplot(data, tick_labels=[str(lbl) for lbl in labels], showfliers=False)
    plt.plot(range(1, len(medians) + 1), medians, color='red', marker='o', linestyle='-', linewidth=2, label="Median vrbls")
    plt.title("Distribution of Analog Outputs (vrbls) per Number of Ones in Weight Matrix")
    plt.xlabel("Number of Ones per Column")
    plt.ylabel("vrbls [V]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for ydata, title, ylabel in zip([means, stds, ratios],
                                    ["Mean of Analog Outputs", "Standard Deviation of Analog Outputs", "(Max - Min) / Mean Ratio"],
                                    ["Mean of vrbls", "Std of vrbls", "Range / Mean Ratio"]):
        plt.figure(figsize=(10, 5))
        plt.plot(labels, ydata, marker='o')
        plt.title(f"{title} vs Number of Ones")
        plt.xlabel("Number of Ones per Column")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Plot and compare VRBL distributions and statistics between raw and balanced inputs
# for different numbers of ones per column across weight patterns.
def plot_vrbls_distribution_raw_vs_balanced(shape, spu_params, spd_params):
    #input_matrix_raw = np.ones(shape)
    input_matrix_raw = np.random.choice([1, 0, -1], size=shape)
    vrbls_raw_outputs = {}
    vrbls_balanced_outputs = {}

    for num_ones in range(0, 257, 16):
        patterns = create_weight_patterns(num_ones, *shape)
        vrbls_raw_runs = []
        vrbls_balanced_runs = []

        for weight_matrix in patterns:
            inputs_raw = input_matrix_raw
            inputs_balanced = pre_processing_inputs(weight_matrix, input_matrix_raw)

            vrbls_raw = simulate_sram_imc(weight_matrix, inputs_raw, spu_params, spd_params)
            vrbls_balanced = simulate_sram_imc(weight_matrix, inputs_balanced, spu_params, spd_params)

            vrbls_raw_runs.append(vrbls_raw.flatten())
            vrbls_balanced_runs.append(vrbls_balanced.flatten())

        vrbls_raw_outputs[num_ones] = np.concatenate(vrbls_raw_runs)
        vrbls_balanced_outputs[num_ones] = np.concatenate(vrbls_balanced_runs)

    labels = list(vrbls_raw_outputs.keys())

    # --- Extract stats ---
    def compute_stats(output_dict):
        data, means, stds, medians, ratios = [], [], [], [], []
        for vrbls in output_dict.values():
            data.append(vrbls)
            means.append(np.mean(vrbls))
            stds.append(np.std(vrbls))
            medians.append(np.median(vrbls))
            ratios.append((np.max(vrbls) - np.min(vrbls)) / np.mean(vrbls) if np.mean(vrbls) != 0 else 0)
        return data, means, stds, medians, ratios

    raw_data, raw_means, raw_stds, raw_medians, raw_ratios = compute_stats(vrbls_raw_outputs)
    balanced_data, balanced_means, balanced_stds, balanced_medians, balanced_ratios = compute_stats(vrbls_balanced_outputs)

    # --- Boxplot Comparison ---
    plt.figure(figsize=(14, 6))
    positions = np.arange(len(labels))
    width = 0.35

    plt.boxplot(raw_data, positions=positions - width/2, widths=width, patch_artist=True,
                boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"), showfliers=False)
    plt.boxplot(balanced_data, positions=positions + width/2, widths=width, patch_artist=True,
                boxprops=dict(facecolor="lightgreen"), medianprops=dict(color="green"), showfliers=False)

    plt.xticks(positions, [str(l) for l in labels])
    plt.title("VRBL Distribution: Raw vs Balanced Inputs per Number of Ones")
    plt.xlabel("Number of Ones per Column")
    plt.ylabel("Analog Output (vrbls)")
    plt.grid(True)
    plt.legend(["Raw", "Balanced"])
    plt.tight_layout()
    plt.show()

    # --- Line plots for comparison ---
    for raw_y, balanced_y, title, ylabel in zip(
        [raw_means, raw_stds, raw_ratios],
        [balanced_means, balanced_stds, balanced_ratios],
        ["Mean of Analog Outputs", "Standard Deviation of Analog Outputs", "(Max - Min) / Mean Ratio"],
        ["Mean of vrbls", "Std of vrbls", "Range / Mean Ratio"]
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(labels, raw_y, label="Raw Input", marker='o')
        plt.plot(labels, balanced_y, label="Balanced Input", marker='x')
        plt.title(f"{title} vs Number of Ones")
        plt.xlabel("Number of Ones per Column")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Compare analog output (VRBL) distributions for:
# 1. Ideal true hardware with no variation
# 2. Raw outputs with variation
# 3. Corrected outputs (using learned X factor for compensation)
# Plot the distributions and print a detailed correction summary.
def plot_vrbls_distribution_ideal_true_vs_corrected(shape, spu_params, spd_params, spu_params_ideal_true, spd_params_ideal_true, vdd=1.0):
    input_matrix = np.ones(shape)
    vrbls_ideal_true_outputs = {}
    vrbls_before_correction_outputs = {}
    vrbls_corrected_outputs = {}

    for num_ones in range(0, 257, 16):
        patterns = create_weight_patterns(num_ones, *shape)
        ideal_runs, corrected_runs, before_runs = [], [], []

        for weight_matrix in patterns:
            # Simulate all 3
            vrbls_ideal_true = simulate_sram_imc(weight_matrix, input_matrix, spu_params_ideal_true, spd_params_ideal_true)
            vrbls_before_correction = simulate_sram_imc(weight_matrix, input_matrix, spu_params, spd_params)

            # Learn X from variation params
            X = learning_X(shape, spu_params, spd_params)
            print("X ",X)

            # Apply correction
            correction_factor = (vdd / vrbls_before_correction - 1) / X
            vrbls_corrected = vdd / (1 + correction_factor)

            # Store flattened
            ideal_runs.append(vrbls_ideal_true.flatten())
            before_runs.append(vrbls_before_correction.flatten())
            corrected_runs.append(vrbls_corrected.flatten())

        vrbls_ideal_true_outputs[num_ones] = np.concatenate(ideal_runs)
        vrbls_before_correction_outputs[num_ones] = np.concatenate(before_runs)
        vrbls_corrected_outputs[num_ones] = np.concatenate(corrected_runs)

    labels = list(vrbls_ideal_true_outputs.keys())

    # --- Extract stats ---
    def compute_stats(output_dict):
        data, means, stds, medians, ratios = [], [], [], [], []
        for vrbls in output_dict.values():
            data.append(vrbls)
            means.append(np.mean(vrbls))
            stds.append(np.std(vrbls))
            medians.append(np.median(vrbls))
            ratios.append((np.max(vrbls) - np.min(vrbls)) / np.mean(vrbls) if np.mean(vrbls) != 0 else 0)
        return data, means, stds, medians, ratios

    ideal_data, ideal_means, ideal_stds, _, ideal_ratios = compute_stats(vrbls_ideal_true_outputs)
    before_data, before_means, before_stds, _, before_ratios = compute_stats(vrbls_before_correction_outputs)
    corrected_data, corrected_means, corrected_stds, _, corrected_ratios = compute_stats(vrbls_corrected_outputs)

    # --- Boxplot Comparison ---
    plt.figure(figsize=(16, 6))
    positions = np.arange(len(labels))
    width = 0.25

    plt.boxplot(ideal_data, positions=positions - width, widths=width, patch_artist=True,
                boxprops=dict(facecolor="lightgray"), medianprops=dict(color="black"), showfliers=False)
    plt.boxplot(before_data, positions=positions, widths=width, patch_artist=True,
                boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"), showfliers=False)
    plt.boxplot(corrected_data, positions=positions + width, widths=width, patch_artist=True,
                boxprops=dict(facecolor="lightcoral"), medianprops=dict(color="darkred"), showfliers=False)

    plt.xticks(positions, [str(l) for l in labels])
    plt.title("VRBL Distribution: Ideal True vs Before vs Corrected per Number of Ones")
    plt.xlabel("Number of Ones per Column")
    plt.ylabel("Analog Output (vrbls)")
    plt.grid(True)
    plt.legend(["Ideal True", "Before Correction", "Corrected"])
    plt.tight_layout()
    plt.show()

    # --- Line plots for comparison ---
    for y_ideal, y_before, y_corrected, title, ylabel in zip(
        [ideal_means, ideal_stds, ideal_ratios],
        [before_means, before_stds, before_ratios],
        [corrected_means, corrected_stds, corrected_ratios],
        ["Mean of Analog Outputs", "Standard Deviation of Analog Outputs", "(Max - Min) / Mean Ratio"],
        ["Mean of vrbls", "Std of vrbls", "Range / Mean Ratio"]
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(labels, y_ideal, label="Ideal True", marker='o')
        plt.plot(labels, y_before, label="Before Correction", marker='s')
        plt.plot(labels, y_corrected, label="Corrected", marker='x')
        plt.title(f"{title} vs Number of Ones")
        plt.xlabel("Number of Ones per Column")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("\n=== Mean VRBL Comparison ===")
    print(f"{'Num Ones':>10} | {'Ideal':>10} | {'Before':>10} | {'Corrected':>10} | {'Closer To Ideal'}")
    print("-" * 65)
    for i, num_ones in enumerate(labels):
        diff_before = abs(before_means[i] - ideal_means[i])
        diff_corrected = abs(corrected_means[i] - ideal_means[i])
        closer = "Corrected" if diff_corrected < diff_before else "Before"
        print(f"{num_ones:>10} | {ideal_means[i]:>10.4f} | {before_means[i]:>10.4f} | {corrected_means[i]:>10.4f} | {closer}")

    print("\n=== Correction Improvement Summary ===")
    print(f"{'Num Ones':>10} | {'Before Err':>12} | {'Corrected Err':>14} | {'Improvement':>12} | {'% Better':>10}")
    print("-" * 70)

    total_before_error = 0
    total_corrected_error = 0

    for i, num_ones in enumerate(labels):
        err_before = abs(before_means[i] - ideal_means[i])
        err_corrected = abs(corrected_means[i] - ideal_means[i])
        improvement = err_before - err_corrected
        percent = (improvement / err_before * 100) if err_before != 0 else 0

        total_before_error += err_before
        total_corrected_error += err_corrected

        print(
            f"{num_ones:>10} | {err_before:>12.6f} | {err_corrected:>14.6f} | {improvement:>12.6f} | {percent:>9.2f}%")

    total_improvement = total_before_error - total_corrected_error
    total_percent = (total_improvement / total_before_error * 100) if total_before_error != 0 else 0

    print("-" * 70)
    print(f"{'Total':>10} | {total_before_error:>12.6f} | {total_corrected_error:>14.6f} | {total_improvement:>12.6f} | {total_percent:>9.2f}%")


# Generate a 2x mirrored version of the given weight matrix:
# - Top left: original
# - Top right: flipped horizontally
# - Bottom left: flipped vertically
# - Bottom right: flipped both vertically and horizontally
def generate_mirrored_matrix(weight_matrix):
    w = weight_matrix
    w_r = np.fliplr(w)
    w_d = np.flipud(w)
    w_rd = np.flip(w, (0, 1))
    top = np.hstack([w, w_r])
    bottom = np.hstack([w_d, w_rd])
    return np.vstack([top, bottom])

# Compute the symmetric average of VRBL outputs from a mirrored matrix.
# Averages values from opposite ends of each column (index i and 63 - i).
def compute_mirrored_avg_vrbls(vrbls_big):
    vrbls_big = vrbls_big.flatten()
    # Average across symmetric outputs (i + 63 - i)/4 → for i in 0 to 31
    return np.array([(vrbls_big[i] + vrbls_big[63 - i]) / 2.0 for i in range(32)])


# Compare raw, raw corrected, mirrored averaged, and mirrored corrected VRBLs
# against an ideal setup with no variation. This is done across varying numbers
# of ones per column.
# Outputs include boxplots, line plots of mean/std/MAE, and summary of improvements.
def plot_raw_vs_mirrored_distribution(w_shape, spu_params, spd_params,
                                      spu_params_no_variation, spd_params_no_variation, X, vdd=1.0):
    input_matrix_w = np.ones(w_shape)
    input_matrix_big = np.ones((256, 64))

    vrbls_outputs_raw = {}
    vrbls_outputs_raw_corrected = {}
    vrbls_outputs_mirrored = {}
    vrbls_outputs_corrected = {}
    vrbls_outputs_ideal = {}

    raw_to_ideal_diffs = {}
    raw_corrected_to_ideal_diffs = {}
    mirrored_to_ideal_diffs = {}
    corrected_to_ideal_diffs = {}

    for num_ones in range(0, 129, 16):
        patterns = create_weight_patterns(num_ones, *w_shape)
        raw_runs, raw_corr_runs = [], []
        mirrored_runs, corrected_runs = [], []
        ideal_runs = []

        for w in patterns:
            vrbls_raw = simulate_sram_imc(w, input_matrix_w, spu_params, spd_params)
            raw_runs.append(vrbls_raw.flatten())

            raw_corr = correct_vrbls(vrbls_raw, X)
            raw_corr_runs.append(raw_corr.flatten())

            vrbls_ideal = simulate_sram_imc(w, input_matrix_w, spu_params_no_variation, spd_params_no_variation)
            ideal_runs.append(vrbls_ideal.flatten())

            w_big = generate_mirrored_matrix(w)
            vrbls_big = simulate_sram_imc(w_big, input_matrix_big, spu_params, spd_params)
            vrbls_big_corr = correct_vrbls(vrbls_big, X)

            mirrored_avg = compute_mirrored_avg_vrbls(vrbls_big)
            corrected_avg = compute_mirrored_avg_vrbls(vrbls_big_corr)

            mirrored_runs.append(mirrored_avg)
            corrected_runs.append(corrected_avg)

        raw_all = np.concatenate(raw_runs)
        raw_corr_all = np.concatenate(raw_corr_runs)
        ideal_all = np.concatenate(ideal_runs)
        mirrored_all = np.concatenate(mirrored_runs)
        corrected_all = np.concatenate(corrected_runs)

        vrbls_outputs_raw[num_ones] = raw_all
        vrbls_outputs_raw_corrected[num_ones] = raw_corr_all
        vrbls_outputs_ideal[num_ones] = ideal_all
        vrbls_outputs_mirrored[num_ones] = mirrored_all
        vrbls_outputs_corrected[num_ones] = corrected_all

        raw_to_ideal_diffs[num_ones] = np.abs(raw_all - ideal_all)
        raw_corrected_to_ideal_diffs[num_ones] = np.abs(raw_corr_all - ideal_all)
        mirrored_to_ideal_diffs[num_ones] = np.abs(mirrored_all - ideal_all)
        corrected_to_ideal_diffs[num_ones] = np.abs(corrected_all - ideal_all)

    # -------------------- Collect Stats --------------------
    labels = []
    raw_data, raw_corr_data, mirrored_data, corrected_data = [], [], [], []
    raw_means, raw_corr_means, mirrored_means, corrected_means = [], [], [], []
    raw_stds, raw_corr_stds, mirrored_stds, corrected_stds = [], [], [], []
    raw_mae, raw_corr_mae, mirrored_mae, corrected_mae = [], [], [], []

    for num_ones in vrbls_outputs_raw:
        labels.append(num_ones)

        raw_v = vrbls_outputs_raw[num_ones]
        raw_corr_v = vrbls_outputs_raw_corrected[num_ones]
        mirrored_v = vrbls_outputs_mirrored[num_ones]
        corrected_v = vrbls_outputs_corrected[num_ones]

        raw_data.append(raw_v)
        raw_corr_data.append(raw_corr_v)
        mirrored_data.append(mirrored_v)
        corrected_data.append(corrected_v)

        raw_means.append(np.mean(raw_v))
        raw_corr_means.append(np.mean(raw_corr_v))
        mirrored_means.append(np.mean(mirrored_v))
        corrected_means.append(np.mean(corrected_v))

        raw_stds.append(np.std(raw_v))
        raw_corr_stds.append(np.std(raw_corr_v))
        mirrored_stds.append(np.std(mirrored_v))
        corrected_stds.append(np.std(corrected_v))

        raw_mae.append(np.mean(raw_to_ideal_diffs[num_ones]))
        raw_corr_mae.append(np.mean(raw_corrected_to_ideal_diffs[num_ones]))
        mirrored_mae.append(np.mean(mirrored_to_ideal_diffs[num_ones]))
        corrected_mae.append(np.mean(corrected_to_ideal_diffs[num_ones]))

    # -------------------- Boxplot --------------------
    plt.figure(figsize=(18, 6))
    positions = np.arange(len(labels))
    width = 0.18

    # Boxplot colors
    colors = {
        "Raw": ("lightblue", "blue"),
        "Raw Corrected": ("skyblue", "darkblue"),
        "Mirrored": ("lightgreen", "green"),
        "Corrected": ("lightcoral", "darkred"),
    }

    # Plot each boxplot with respective color
    plt.boxplot(raw_data, positions=positions - 1.5 * width, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors["Raw"][0]), medianprops=dict(color=colors["Raw"][1]), showfliers=False)
    plt.boxplot(raw_corr_data, positions=positions - 0.5 * width, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors["Raw Corrected"][0]), medianprops=dict(color=colors["Raw Corrected"][1]),
                showfliers=False)
    plt.boxplot(mirrored_data, positions=positions + 0.5 * width, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors["Mirrored"][0]), medianprops=dict(color=colors["Mirrored"][1]),
                showfliers=False)
    plt.boxplot(corrected_data, positions=positions + 1.5 * width, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors["Corrected"][0]), medianprops=dict(color=colors["Corrected"][1]),
                showfliers=False)

    plt.xticks(positions, labels)
    plt.title("VRBL Distribution: Raw vs Raw Corrected vs Mirrored vs Corrected")
    plt.xlabel("Number of Ones per Column")
    plt.ylabel("VRBLS [V]")
    plt.grid(True)

    # Custom colored legend
    legend_handles = [
        mpatches.Patch(color=colors["Raw"][0], label="Raw"),
        mpatches.Patch(color=colors["Raw Corrected"][0], label="Raw Corrected"),
        mpatches.Patch(color=colors["Mirrored"][0], label="Mirrored"),
        mpatches.Patch(color=colors["Corrected"][0], label="Corrected"),
    ]
    plt.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.show()

    # -------------------- Line Plots --------------------
    stat_sets = [
        (raw_means, raw_corr_means, mirrored_means, corrected_means, "Mean of VRBLs", "Mean"),
        (raw_stds, raw_corr_stds, mirrored_stds, corrected_stds, "Standard Deviation of VRBLs", "Standard Deviation"),
        (raw_mae, raw_corr_mae, mirrored_mae, corrected_mae, "Mean Absolute Error to Ideal VRBL", "Mean Absolute Error [V]"),
    ]

    for raw_y, raw_corr_y, mirrored_y, corrected_y, title, ylabel in stat_sets:
        plt.figure(figsize=(10, 5))
        plt.plot(labels, raw_y, label="Raw", marker='o')
        plt.plot(labels, raw_corr_y, label="Raw Corrected", marker='s')
        plt.plot(labels, mirrored_y, label="Mirrored", marker='x')
        plt.plot(labels, corrected_y, label="Corrected", marker='^')
        plt.title(title)
        plt.xlabel("Number of Ones per Column")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # -------------------- Improvement in STD (%) --------------------
    corrected_stds_np = np.array(corrected_stds)
    raw_stds_np = np.array(raw_stds)
    improvement_pct = (1 - corrected_stds_np / raw_stds_np) * 100

    # Display as DataFrame
    import pandas as pd

    df = pd.DataFrame({
        "Number of Ones": labels,
        "Raw STD": raw_stds_np,
        "Corrected STD": corrected_stds_np,
        "Improvement (%)": improvement_pct
    })

    print("\n=== Percentage Improvement in STD (Corrected vs Raw) ===")
    print(df.to_string(index=False))

    # --- Identify max improvement in MAE ---
    raw_mae_np = np.array(raw_mae)
    corrected_mae_np = np.array(corrected_mae)
    mae_improvement_pct = (1 - corrected_mae_np / raw_mae_np) * 100
    max_mae_idx = np.argmax(mae_improvement_pct)
    max_mae_label = labels[max_mae_idx]

    print(f"\nMax MAE improvement at {max_mae_label} ones:")
    print(f"Raw MAE: {raw_mae_np[max_mae_idx]:.6e}")
    print(f"Corrected MAE: {corrected_mae_np[max_mae_idx]:.6e}")
    print(f"Improvement: {mae_improvement_pct[max_mae_idx]:.2f}%")

# Learn the effective mismatch ratio X = (vdd/vrbl - 1) averaged over all cells
# using an alternating weight matrix (+1/-1 rows) and input=1
def learning_X(shape, spu_params, spd_params, vdd=1.0):
    input_matrix = np.ones(shape)
    weights = np.empty(shape, dtype=int)
    weights[::2, :] = 1  # Even-indexed rows: 0, 2, 4, ...
    weights[1::2, :] = -1  # Odd-indexed rows: 1, 3, 5, ...
    vrbls = simulate_sram_imc(weights, input_matrix, spu_params, spd_params)
    X = np.mean((vdd / vrbls) - 1)

    return X

# Apply correction to raw VRBLs using learned mismatch factor X
# Based on: VRBL_corrected = Vdd / (1 + (Vdd / VRBL_raw - 1) / X)
def correct_vrbls(vrbls_raw, X, vdd=1):
    correction_factor = (vdd / vrbls_raw - 1) / X
    vrbls_corrected = vdd / (1 + correction_factor)
    return vrbls_corrected

# Run simulation sweep across a range of gradient strengths
# Compare how VRBL error changes with and without balanced input redistribution
# Plots error vs gradient percentage for raw and balanced inputs
def run_gradient_sweep_summary(shape, base_spu, base_spd, no_var_spu, no_var_spd, gradient_range):
    vrbl_diff_raw_list = []
    vrbl_diff_balanced_list = []

    for grad in gradient_range:
        grad = grad / 100
        spu_params = {
            "mean": base_spu["mean"],
            "sigma_pct": base_spu["sigma_pct"],
            "gradient_x_pct": grad,
            "gradient_y_pct": grad,
        }
        spd_params = {
            "mean": base_spd["mean"],
            "sigma_pct": base_spd["sigma_pct"],
            "gradient_x_pct": grad,
            "gradient_y_pct": grad,
        }

        results = []
        for seed in range(30):  # Run 30 simulations per gradient level
            result = run_simulation(seed, shape, spu_params, spd_params, no_var_spu, no_var_spd)
            results.append(result)

        results = np.array(results)
        vrbl_diff_raw = np.mean(results[:, 0])       # Raw vs Ideal VRBL
        vrbl_diff_balanced = np.mean(results[:, 1])  # Balanced vs Ideal VRBL
        vrbl_diff_raw_list.append(vrbl_diff_raw)
        vrbl_diff_balanced_list.append(vrbl_diff_balanced)

    # Plotting the results
    plt.figure(figsize=(8, 5))
    plt.plot(gradient_range, vrbl_diff_raw_list, marker='o', label='Raw Input')
    plt.plot(gradient_range, vrbl_diff_balanced_list, marker='o', label='Balanced Input')
    plt.fill_between(gradient_range, vrbl_diff_raw_list, vrbl_diff_balanced_list,
                     color='lightgray', alpha=0.5, label='Improvement Gap')
    plt.xlabel("Gradient X and Y [%]")
    plt.ylabel("Average Absolute VRBL Diff [V]")
    plt.title("VRBL Error vs Gradient Strength (Raw vs Balanced)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Main ===
def main():
    shape = (256, 64)
    #shape = (30, 30)
    spu_params = {"mean": 10e3, "sigma_pct": 0.15, "gradient_x_pct": 0.15, "gradient_y_pct": 0.17}
    spd_params = {"mean": 10.5e3, "sigma_pct": 0.17, "gradient_x_pct": 0.20, "gradient_y_pct": 0.18}

    spu_params_ideal_true = {"mean": 10e3, "sigma_pct": 0.05, "gradient_x_pct": 0.05, "gradient_y_pct": 0.07}
    spd_params_ideal_true = {"mean": 10e3, "sigma_pct": 0.07, "gradient_x_pct": 0.10, "gradient_y_pct": 0.08}

    ideal_spu_params = {"mean": 10e3, "sigma_pct": 0.0, "gradient_x_pct": 0.0, "gradient_y_pct": 0.0}
    ideal_spd_params = {"mean": 10e3, "sigma_pct": 0.0, "gradient_x_pct": 0.0, "gradient_y_pct": 0.0}

    spu_params_no_gradients = {"mean": 10e3, "sigma_pct": 0.0, "gradient_x_pct": 0.0, "gradient_y_pct": 0.0}
    spd_params_no_gradients = {"mean": 10.5e3, "sigma_pct": 0.0, "gradient_x_pct": 0.0, "gradient_y_pct": 0.0}
    w_shape = (128, 32)

    X = learning_X(shape, spu_params, spd_params, vdd=1.0)
    #run_summary_simulation_comparison(shape, spu_params, spd_params, ideal_spu_params, ideal_spd_params)
    #plot_vrbls_distribution_raw_vs_balanced(shape, spu_params, spd_params)
    #plot_vrbls_distribution(shape, spu_params, spd_params)
    #plot_raw_vs_mirrored_distribution(w_shape, spu_params, spd_params, ideal_spu_params, ideal_spd_params, X)
    #plot_vrbls_distribution_ideal_true_vs_corrected(shape, spu_params_no_gradients, spd_params_no_gradients, ideal_spu_params, ideal_spd_params, vdd=1.0)
    #plot_raw_vs_mirrored_distribution1(w_shape, spu_params, spd_params, ideal_spu_params, ideal_spd_params)
    gradient_range = np.linspace(25, 50, 10)
    base_spu = {"mean": 10e3,"sigma_pct": 0.15}
    base_spd = {"mean": 10.5e3,"sigma_pct": 0.17}
    no_var_spu = {"mean": 10e3,"sigma_pct": 0.0,"gradient_x_pct": 0.0,"gradient_y_pct": 0.0}
    no_var_spd = {"mean": 10.5e3,"sigma_pct": 0.0,"gradient_x_pct": 0.0,"gradient_y_pct": 0.0}

    # --- Run Simulation Sweep ---
    run_gradient_sweep_summary(shape, base_spu, base_spd, no_var_spu, no_var_spd, gradient_range)


if __name__ == "__main__":
    main()
