from pathlib import Path
import importlib.util
import sys


def _load_parser_module():
    """Dynamically load the bundled `parser.py` module.

    This avoids import problems if the package dir contains characters
    (like a dash) that make a normal import fail.
    """
    root = Path(__file__).parent
    parser_path = root / "src" / "actions-metrics" / "parser.py"
    if not parser_path.exists():
        raise FileNotFoundError(f"parser not found at {parser_path}")

    spec = importlib.util.spec_from_file_location("am_parser", str(parser_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    parser = _load_parser_module()
    merged = parser.load_examples_and_merge(example_dir=Path(__file__).parent / "example")
    
    # Filter out unwanted jobs
    merged = parser.filter_jobs(merged)
    
    # Sort by workflow and job
    merged = parser.sort_by_workflow_and_job(merged)
    
    # print a compact preview
    print(merged.head(25))

    # save to file
    output_file = "merged_metrics.csv"
    merged.write_csv(output_file)
    print(f"\nMerged data saved to {output_file}")


if __name__ == "__main__":
    main()
