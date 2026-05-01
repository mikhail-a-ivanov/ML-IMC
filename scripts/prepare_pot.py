import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

DTYPE = np.float32


def parabolic_potential(r, a, b, c):
    return a - b * r**2 + c * r**4


def smooth_repulsion_region(r_values, y_values, cutoff_idx, max_value=70):
    # Get the exact connection point values
    r_connect = r_values[cutoff_idx]
    y_connect = y_values[cutoff_idx]

    # Calculate the derivative at the connection point
    if cutoff_idx + 3 < len(r_values):
        slope, _, _, _, _ = linregress(
            r_values[cutoff_idx : cutoff_idx + 4],
            y_values[cutoff_idx : cutoff_idx + 4],
        )
        slope = DTYPE(slope)
    elif cutoff_idx + 1 < len(r_values):
        slope = DTYPE(
            (y_values[cutoff_idx + 1] - y_values[cutoff_idx])
            / (r_values[cutoff_idx + 1] - r_values[cutoff_idx]),
        )
    else:
        slope = DTYPE(0)

    # Set parameter a from condition U(0) = max_value
    a = DTYPE(max_value)

    # Calculate c and b to satisfy U(r_connect) = y_connect and U'(r_connect) = slope
    c = DTYPE((a + slope * r_connect / DTYPE(2) - y_connect) / (r_connect**4))
    b = DTYPE(-slope / (DTYPE(2) * r_connect) + DTYPE(2) * c * r_connect**2)

    print(f"Fitted model: U(r) = {a} - {b}*r^2 + {c}*r^4")

    # Create the smoothed function for the repulsion region
    y_smoothed = y_values.copy()
    for i in range(cutoff_idx):
        y_smoothed[i] = parabolic_potential(r_values[i], a, b, c)

    return y_smoothed


def process_potential_data(filename, max_potential_value=70, skip_points=5):
    try:
        data = np.loadtxt(filename, dtype=DTYPE)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        try:
            with open(filename) as f:
                lines = f.readlines()

            data = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append([DTYPE(parts[0]), DTYPE(parts[1])])

            data = np.array(data, dtype=DTYPE)
        except Exception as e2:
            print(f"Error parsing file manually: {e2}")
            return None, None, None

    r = data[:, 0]
    y = data[:, 1]

    # Find the cutoff point where values drop from high to normal
    diffs = np.abs(np.diff(y))
    significant_drops = np.where(diffs > 1000)[0]

    if len(significant_drops) > 0:
        initial_cutoff_idx = significant_drops[-1] + 1
        print(
            f"Found transition at index {initial_cutoff_idx}, r = {r[initial_cutoff_idx]}",
        )

        # Skip ahead a few points to where the repulsion behavior stabilizes
        cutoff_idx = min(initial_cutoff_idx + skip_points, len(r) - 1)
        print(f"Using stabilized point at index {cutoff_idx}, r = {r[cutoff_idx]}")
    else:
        # If no clear drop is found, use a threshold-based approach
        high_values = np.where(y > 1000)[0]
        if len(high_values) > 0:
            initial_cutoff_idx = high_values[-1] + 1
            cutoff_idx = min(initial_cutoff_idx + skip_points, len(r) - 1)
            print(
                f"Using point after high values: index {cutoff_idx}, r = {r[cutoff_idx]}",
            )
        else:
            print("No clear transition found, treating all values as normal.")
            cutoff_idx = 0

    # Apply the smoothing
    y_smoothed = smooth_repulsion_region(
        r,
        y,
        cutoff_idx,
        max_value=max_potential_value,
    )

    return r, y, y_smoothed, cutoff_idx


def plot_results(r, y_original, y_smoothed, cutoff_idx, filename="data.txt"):
    plt.figure(figsize=(10, 6))
    # plt.plot(r, y_original, 'r--', label='Original')
    plt.plot(r, y_smoothed, "b-", label="Smoothed")
    plt.axvline(x=r[cutoff_idx], color="g", linestyle=":", label="Connection Point")
    plt.legend()
    plt.xlabel("Distance (r)")
    plt.ylabel("Potential Energy")
    plt.title("Potential Energy Curve with Smoothed Repulsion Region")
    plt.grid(True)
    plt.show()


def main():
    filename = "methanol-IMC-potentials_form/02-100CH3OH.i005.pot"
    max_potential = 95
    skip_points = 30

    r, y_orig, y_smooth, cutoff_idx = process_potential_data(
        filename,
        max_potential_value=max_potential,
        skip_points=skip_points,
    )

    if r is not None:
        plot_results(r, y_orig, y_smooth, cutoff_idx)

        output_filename = "100_CH3OH_magic_pot.dat"
        try:
            np.savetxt(output_filename, np.column_stack((r, y_smooth)), fmt="%.8f")
            print(f"Smoothed data saved to '{output_filename}'")
        except Exception as e:
            print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
