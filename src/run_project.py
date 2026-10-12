"""Command-line entry point for the COD return-risk study."""
import argparse
from ml_workflow import run_project

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the COD return-risk models")
    parser.add_argument("--dataset", help="Path to the approved privacy-reduced CSV; default: data/Modalii_Lenshub_ML_Dataset_Final.csv")
    args = parser.parse_args()
    run_project(dataset=args.dataset)

if __name__ == "__main__":
    main()
