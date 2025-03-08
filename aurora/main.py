import argparse
from pathlib import Path
from aurora.plot import plot_camera_image, plot_camera_views

def main():
    parser = argparse.ArgumentParser(prog="aurora")
    parser.add_argument("dataset", type=Path)

    subparsers = parser.add_subparsers(dest="command")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("type", choices=["image", "views"])
    show_parser.add_argument("name", type=str)

    args = parser.parse_args()

    if args.command == "show":
        if args.type == "image":
            plot_camera_image(args.dataset / args.name)
        elif args.type == "views":
            plot_camera_views(args.dataset / args.name)

if __name__ == "__main__":
    main()