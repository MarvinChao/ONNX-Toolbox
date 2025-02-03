import argparse
import onnx

from onnx_analysis import ModelStats


def main():
    parser = argparse.ArgumentParser(description="Toolbox for analyzing the ONNX model")

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input ONNX model filename"
    )
    parser.add_argument(
        "--memory",
        "-m",
        type=int,
        default=0,
        required=False,
        help="Local memory size (in MBytes)",
    )
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        required=False,
        help="Generate ONNX analysis report",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        required=False,
        help="Saved processed onnx model",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        required=False,
        help="Verbose output for debugging purposes",
    )

    args = parser.parse_args()

    print_args(args)

    model_stats = ModelStats(args)
    if args.report == True:
        model_stats.generate_report()
    if args.save == True:
        model_stats.save_model()


def print_args(args):
    print("Input arguments:")
    print(f"========================================")
    for key, value in vars(args).items():
        print(f"{key:<10}:      {value}")
    print(f"========================================")


if __name__ == "__main__":
    main()
