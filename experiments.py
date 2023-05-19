import argparse
import subprocess

MODELS = ["nl1", "nl2", "watertank", "tank4d", "tank6d"]
TEMPLATES = ["pwc", "pwa", "nl"]


def cli():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="all",
        help="models to run",
        nargs="+",
        choices=["nl1", "nl2", "watertank", "tank4d", "tank6d", "all"],
    )
    parser.add_argument(
        "-t",
        "--templates",
        type=str,
        default="all",
        help="templates to run",
        nargs="+",
        choices=["pwc", "pwa", "nl", "all"],
    )
    parser.add_argument("--flowstar", type=bool, default=True, help="run flowstar")
    parser.add_argument("--spaceex", type=bool, default=True, help="run spaceex")
    args = parser.parse_args()
    return args


def main(args):
    models = args.models
    templates = args.templates
    flowstar = args.flowstar
    spaceex = args.spaceex

    if "all" in models:
        models = MODELS
    if "all" in templates:
        templates = TEMPLATES
    print(templates)

    for model in models:
        for template in templates:
            config = f"experiments/{model}/{template}-{model}-config.yaml"
            if model == "tank6d" and (template == "nl" or template == "pwc"):
                continue

            args = f"--config {config}"
            if template == "nl":
                args += f" --flowstar {flowstar}"
            elif template == "pwa" or template == "pwc":
                args += f" --spaceex {spaceex}"
            # args += f" --error-check {error_check}"
            subprocess.run("python3 main.py " + args, shell=True, check=True)


if __name__ == "__main__":
    args = cli()
    main(args)
