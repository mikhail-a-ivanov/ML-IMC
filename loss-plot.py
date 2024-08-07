import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_data(file_name):
    """Load data from a file."""
    return np.genfromtxt(file_name).T


def set_plot_style():
    """Set custom style for minimalist look."""
    plt.style.use("default")  # Reset to default style
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def create_plot(data):
    """Create and return the figure and axis objects."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(data, color="steelblue", linewidth=0.8)
    return fig, ax


def set_labels(ax, file_name, use_log_scale):
    """Set title and axis labels."""
    ax.set_title(f"Loss function: {file_name}", fontsize=16, pad=20)
    ax.set_xlabel("Iteration", fontsize=12, labelpad=10)
    if use_log_scale:
        ax.set_ylabel("Loss value (log scale)", fontsize=12, labelpad=10)
    else:
        ax.set_ylabel("Loss value", fontsize=12, labelpad=10)


def set_axis_limits(ax, data, use_log_scale):
    """Set axis limits, adapting for log scale if necessary."""
    ax.set_xlim(0, len(data))

    if use_log_scale:
        min_val = max(data.min(), 1e-10)  # Avoid zero or negative values
        ax.set_ylim(min_val, data.max() * 1.1)
    else:
        ax.set_ylim(0, data.max() * 1.1)


def customize_spines(ax):
    """Customize plot spines."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def customize_ticks(ax):
    """Customize tick marks."""
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(axis="both", which="major", length=4)


def save_plot(fig, file_name, use_log_scale):
    """Save the plot to a file."""
    scale_type = "log" if use_log_scale else "linear"
    output_file = f"plot-{file_name}-{scale_type}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_file}")


def set_log_scale(ax):
    """Set logarithmic scale for y-axis."""
    ax.set_yscale("log")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot loss function from a file.")
    parser.add_argument("file_name", help="Name of the file containing loss values")
    parser.add_argument(
        "--log", action="store_true", help="Use logarithmic scale for y-axis"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    data = load_data(args.file_name)

    set_plot_style()
    fig, ax = create_plot(data)

    set_labels(ax, args.file_name, args.log)
    if args.log:
        set_log_scale(ax)
    set_axis_limits(ax, data, args.log)
    customize_spines(ax)
    customize_ticks(ax)

    plt.grid(False)
    plt.tight_layout()

    save_plot(fig, args.file_name, args.log)
    plt.show()


if __name__ == "__main__":
    main()
