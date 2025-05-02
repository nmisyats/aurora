import argparse
from pathlib import Path
from aurora.plot import plot_camera_image, plot_camera_views, plot_model_matrix

def main():
    parser = argparse.ArgumentParser(prog="aurora")

    subparsers = parser.add_subparsers(dest="command")

    plot_parser = subparsers.add_parser("plot")
    plot_subparsers = plot_parser.add_subparsers(dest="plot_command")
    plot_image_parser = plot_subparsers.add_parser("image")
    plot_image_parser.add_argument("path", type=Path)
    plot_views_parser = plot_subparsers.add_parser("views")
    plot_views_parser.add_argument("path", type=Path)
    plot_model_parser = plot_subparsers.add_parser("model")
    plot_model_parser.add_argument("path", type=Path)

    recon_parser = subparsers.add_parser("reconstruct")
    recon_parser.add_argument("path", type=Path)

    args = parser.parse_args()

    if args.command == "plot":
        if args.plot_command == "image":
            plot_camera_image(args.path)
        elif args.plot_command == "views":
            plot_camera_views(args.path)
        elif args.plot_command == "model":
            plot_model_matrix(args.path)

if __name__ == "__main__":
    main()