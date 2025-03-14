import argparse
from experiments.caltech101_experiment import run_caltech_experiment
from experiments.msrcv1_experiment import run_msrc_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepMVC Training")
    parser.add_argument("--dataset", type=str, choices=["caltech101", "msrcv1"], required=True)
    args = parser.parse_args()

    if args.dataset == "caltech101":
        run_caltech_experiment()
    elif args.dataset == "msrcv1":
        run_msrc_experiment()
