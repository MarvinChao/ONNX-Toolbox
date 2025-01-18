import argparse
import onnx

from onnx_analysis import ModelStats


def main():
    parser = argparse.ArgumentParser(description="Toolbox for analyzing the ONNX model")

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input ONNX model filename"
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

    args = parser.parse_args()

    print_args(args)

    model_stats = ModelStats(args.input)
    if args.report == True:
        model_stats.generate_report()
    if args.save == True:
        model_stats.save_model()


def print_args(args):
    if args.input:
        print(f"Input ONNX model is {args.input}")
    if args.report:
        print(f"Generate report: {args.report}")
    if args.save:
        print(f"Save model: {args.save}")


if __name__ == "__main__":
    main()
