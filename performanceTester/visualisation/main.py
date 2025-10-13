import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import argparse


def loadAndParse(file_path, dimensions):
    """
    Load and parse the input file into a NumPy array.

    Args:
        file_path: Path to the input file
        dimensions: Number of dimensions (1, 2, or 3)

    Returns:
        NumPy array reshaped according to dimensions
    """
    with open(file_path, "r") as f:
        lines = f.read().strip().split("\n")

    # Extract dimension from first line
    # Parse all float values from the second line
    values = np.array([float(x) for x in lines[0].split()])

    # Determine shape based on dimensions
    total_values = len(values)

    if dimensions == 1:
        # 1D: values remain as-is
        shaped_data = values
    elif dimensions == 2:
        # 2D: assume square matrix
        side_length = int(np.sqrt(total_values))
        if side_length * side_length != total_values:
            raise ValueError(
                f"Cannot reshape {total_values} values into a square 2D array"
            )
        shaped_data = values.reshape(side_length, side_length)
    elif dimensions == 3:
        # 3D: assume cubic array
        side_length = int(np.cbrt(total_values))
        if side_length**3 != total_values:
            raise ValueError(
                f"Cannot reshape {total_values} values into a cubic 3D array"
            )
        shaped_data = values.reshape(side_length, side_length, side_length)
    else:
        raise ValueError(
            f"Unsupported dimensions: {dimensions}. Only 1, 2, or 3 dimensions supported."
        )

    return shaped_data, dimensions


def visualizeData(stateSpace, predictions=None, file_name=""):
    """
    Visualize state space data based on dimensionality.

    Args:
        stateSpace: NumPy array of state space data
        predictions: Optional prediction data (not used in this implementation)
        file_name: Name of the file being visualized (for title)
    """
    dimensions = stateSpace.ndim

    # Create output directory if it doesn't exist
    output_dir = "./visualisationImages"
    os.makedirs(output_dir, exist_ok=True)

    # Generate base filename from input file (remove extension)
    base_name = os.path.splitext(file_name)[0]

    if dimensions == 1:
        # 1D Visualization: Line plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(stateSpace))
        plt.plot(x, stateSpace, "b-", linewidth=2)
        plt.xlabel("Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"1D State Space Visualization\n{file_name}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save with specific filename
        output_path = os.path.join(output_dir, f"{base_name}_1d_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
        plt.close()

    elif dimensions == 2:
        # 2D Visualization: 3D Surface plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Create meshgrid
        x = np.arange(stateSpace.shape[1])
        y = np.arange(stateSpace.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot surface
        surf = ax.plot_surface(
            X, Y, stateSpace, cmap="viridis", edgecolor="none", alpha=0.9
        )

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Labels and title
        ax.set_xlabel("X Axis", fontsize=11)
        ax.set_ylabel("Y Axis", fontsize=11)
        ax.set_zlabel("Value", fontsize=11)
        ax.set_title(f"2D State Space Surface Plot\n{file_name}", fontsize=14, pad=20)

        plt.tight_layout()

        # Save with specific filename
        output_path = os.path.join(output_dir, f"{base_name}_2d_surface.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
        plt.close()

    elif dimensions == 3:
        # 3D Visualization: 3D Surface plot with slider for Z dimension
        fig = plt.figure(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.15)
        ax = fig.add_subplot(111, projection="3d")

        # Initial slice through Z dimension
        initial_slice = stateSpace.shape[2] // 2

        # Create meshgrid for X and Y
        x = np.arange(stateSpace.shape[1])
        y = np.arange(stateSpace.shape[0])
        X, Y = np.meshgrid(x, y)

        # Get the slice data - this will be the Z height
        Z_data = stateSpace[:, :, initial_slice]

        # Plot surface - Z position is the slice data, color is also from slice data
        surf = ax.plot_surface(
            X, Y, Z_data, cmap="viridis", edgecolor="none", alpha=0.9
        )

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Value", fontsize=11)

        # Labels and title
        ax.set_xlabel("X Axis", fontsize=11)
        ax.set_ylabel("Y Axis", fontsize=11)
        ax.set_zlabel("Value", fontsize=11)
        ax.set_title(
            f"3D State Space Surface Plot - Z Dimension = {initial_slice}\n{file_name}",
            fontsize=14,
            pad=20,
        )

        # Create slider for Z dimension
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax_slider,
            "Z Dimension",
            0,
            stateSpace.shape[2] - 1,
            valinit=initial_slice,
            valstep=1,
        )

        # Update function for slider
        def update(val):
            slice_idx = int(slider.val)
            Z_data = stateSpace[:, :, slice_idx]

            # Clear and redraw the surface
            ax.clear()
            surf = ax.plot_surface(
                X, Y, Z_data, cmap="viridis", edgecolor="none", alpha=0.9
            )

            # Re-add labels
            ax.set_xlabel("X Axis", fontsize=11)
            ax.set_ylabel("Y Axis", fontsize=11)
            ax.set_zlabel("Value", fontsize=11)
            ax.set_title(
                f"3D State Space Surface Plot - Z Dimension = {slice_idx}\n{file_name}",
                fontsize=14,
                pad=20,
            )

            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.tight_layout()

        # Save with specific filename (saves the initial slice)
        output_path = os.path.join(output_dir, f"{base_name}_3d_surface.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

        # Show interactive plot
        plt.show()

    else:
        raise ValueError(
            f"Cannot visualize {dimensions}D data. Only 1D, 2D, and 3D supported."
        )


def run(functionPath: str, predictPath: str = None, dimensions: int = 2) -> None:
    """
    Main execution function to load and visualize state space data.

    Args:
        functionPath: Path to the input data file
        predictPath: Optional path to prediction data (not implemented)
        dimensions: Expected number of dimensions (will be overridden by file content)
    """
    # Check if file exists
    if not os.path.exists(functionPath):
        raise FileNotFoundError(f"File not found: {functionPath}")

    print(f"Loading data from: {functionPath}")

    # Load and parse the data
    stateSpace, actual_dimensions = loadAndParse(functionPath, dimensions)

    print(f"Data loaded successfully:")
    print(f"  Dimensions: {actual_dimensions}")
    print(f"  Shape: {stateSpace.shape}")
    print(f"  Value range: [{stateSpace.min():.4f}, {stateSpace.max():.4f}]")
    print(f"  Total elements: {stateSpace.size}")

    # Extract file name for display
    file_name = os.path.basename(functionPath)

    # Visualize the data
    visualizeData(stateSpace, predictions=None, file_name=file_name)


def main():
    parser = argparse.ArgumentParser(description="Visualize state space data.")
    parser.add_argument(
        "functionFilepath",
        nargs="?",
        type=str,
        default=None,
        help="Path to the main function data directory (e.g., PerformanceData2:50)",
    )

    args = parser.parse_args()

    # Determine base path
    if args.functionFilepath:
        base_path = os.path.join("../../", args.functionFilepath)
    else:
        base_path = "../../PerformanceData2:50"

    # --- Extract dimensions from base_path ---
    try:
        # Find the substring between the last non-digit and the colon
        # e.g. "../../PerformanceData2:50" -> extract "2"
        before_colon = base_path.split(":")[0]
        dims_str = "".join(ch for ch in before_colon if ch.isdigit())
        if not dims_str:
            raise ValueError("No dimension number found before ':' in base path.")
        dimensions = int(dims_str[-1])  # Use last digit in case of multi-digit folders
    except Exception as e:
        print(f"Could not extract dimensions from base path: {e}")
        dimensions = 2  # Default fallback

    print(f"Extracted dimensions: {dimensions}")

    # Process each function directory
    if os.path.exists(base_path):
        function_dirs = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]

        print(f"Found {len(function_dirs)} function directories: {function_dirs}")

        for func_name in sorted(function_dirs):
            func_path = os.path.join(base_path, func_name)
            txt_files = glob.glob(os.path.join(func_path, "*.txt"))

            print(f"\n{'='*60}")
            print(f"Processing {func_name}: {len(txt_files)} files found")
            print(f"{'='*60}")

            for file_path in sorted(txt_files):
                try:
                    print(f"\nVisualizing: {os.path.basename(file_path)}")
                    run(file_path, dimensions=dimensions)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    else:
        print(f"Base path not found: {base_path}")


if __name__ == "__main__":
    main()
