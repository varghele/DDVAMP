# pipeline.py
import os
from typing import List, Optional, Dict
from utils import preparation
from train.train_main import RevVAMPTrainer
from args.pipeline_args import buildParser as buildPipelineParser
from args.preparation_args import buildParser as buildPrepParser, validate_args as validate_prep_args
from args.training_args import buildParser as buildTrainParser


class Pipeline:
    def __init__(self, pipeline_args, preparation_args, training_args):
        self.pipeline_args = pipeline_args
        self.preparation_args = preparation_args
        self.training_args = training_args
        self.results = {}

        # Create directory structure
        self.base_dir = os.path.join('data', self.pipeline_args.protein_name)
        self.interim_dir = os.path.join(self.base_dir, 'interim')
        self.proc_dir = os.path.join(self.base_dir, 'proc')
        self.model_dir = os.path.join(self.base_dir, 'models', 'revgraphvamp')

        # Create all necessary directories
        for directory in [self.interim_dir, self.proc_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)

        # Update output directories in args
        self.preparation_args.output_dir = self.interim_dir
        self.training_args.save_folder = self.model_dir
        self.training_args.data_path = os.path.join(self.interim_dir,
                                                    os.path.basename(self.training_args.data_path))

    def run_preparation(self) -> Dict:
        """Step 1: Process molecular dynamics trajectories"""
        print("Starting trajectory processing...")
        print(f"Saving preparation results to: {self.interim_dir}")
        results = preparation.run_pipeline(self.preparation_args)
        self.results['preparation'] = results
        return results

    def run_training(self) -> Dict:
        """Step 2: Run model training using RevVAMPTrainer"""
        print("Starting model training...")
        print(f"Saving model to: {self.model_dir}")
        print(f"Using data from: {self.training_args.data_path}")

        # Initialize trainer with arguments
        trainer = RevVAMPTrainer(self.training_args)

        # Train model and get results
        model, epochs_trained = trainer.train()

        # Store results
        self.results['training'] = {
            'model': model,
            'epochs_trained': epochs_trained,
            'training_scores': model._train_scores if hasattr(model, '_train_scores') else None,
            'validation_scores': model._validation_scores if hasattr(model, '_validation_scores') else None
        }

        return self.results['training']

    def run(self, steps: Optional[List[str]] = None) -> Dict:
        """Run specified pipeline steps"""
        available_steps = {
            'preparation': self.run_preparation,
            'training': self.run_training
        }

        if steps is None:
            steps = list(available_steps.keys())

        for step in steps:
            if step not in available_steps:
                raise ValueError(f"Unknown step: {step}")
            print(f"\nExecuting step: {step}")
            available_steps[step]()

        return self.results


def main():
    # Create parsers for each component
    pipeline_parser = buildPipelineParser()
    prep_parser = buildPrepParser()
    train_parser = buildTrainParser()

    # Combine all arguments into a single parser
    import argparse
    all_args = argparse.ArgumentParser(parents=[pipeline_parser, prep_parser, train_parser],
                                       conflict_handler='resolve')
    args = all_args.parse_args()

    # Create and run pipeline
    pipeline = Pipeline(
        pipeline_args=args,
        preparation_args=args,
        training_args=args
    )

    results = pipeline.run(args.steps)
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
