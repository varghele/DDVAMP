# args/pipeline_args.py

import argparse
import os


def buildParser():
    """Create and return an argument parser for pipeline control parameters."""
    parser = argparse.ArgumentParser(description='Pipeline control parameters')

    # Pipeline Flow Control
    flow_group = parser.add_argument_group('Pipeline Flow Control')
    flow_group.add_argument('--steps', nargs='+',
                            choices=['preparation', 'training'],
                            default=None,
                            help='Specific steps to run (default: all steps)')

    # Pipeline Output Configuration
    output_group = parser.add_argument_group('Pipeline Output Configuration')
    output_group.add_argument('--protein-name', type=str, required=True,
                              help='Name of the protein being analyzed (e.g., "ala")')
    output_group.add_argument('--base-dir', type=str,
                              default="data",
                              help='Base directory for all data (default: data)')
    output_group.add_argument('--log-level', type=str,
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              default='INFO',
                              help='Logging level for pipeline execution')

    # Pipeline Execution Configuration
    exec_group = parser.add_argument_group('Pipeline Execution Configuration')
    exec_group.add_argument('--parallel', action='store_true',
                            help='Enable parallel processing where possible')
    exec_group.add_argument('--debug-mode', action='store_true',
                            help='Run pipeline in debug mode with additional checks')

    return parser


def validate_args(args):
    """Validate the parsed pipeline arguments."""
    # Create base directory structure
    protein_dir = os.path.join(args.base_dir, args.protein_name)
    args.interim_dir = os.path.join(protein_dir, 'interim')
    args.proc_dir = os.path.join(protein_dir, 'proc')
    args.model_dir = os.path.join(protein_dir, 'models', 'revgraphvamp')

    # Create all necessary directories
    for directory in [args.interim_dir, args.proc_dir, args.model_dir]:
        os.makedirs(directory, exist_ok=True)

    # Set output_dir to protein directory
    args.output_dir = protein_dir

    return args


if __name__ == "__main__":
    parser = buildParser()
    args = parser.parse_args()
    args = validate_args(args)
