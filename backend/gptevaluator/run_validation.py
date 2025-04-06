"""
Script to run the GPT-4o validation experiments
"""

import argparse
import os
from gpt4o_validation import validate_gpt4o_evaluation, validate_ranking_capability

def main():
    parser = argparse.ArgumentParser(description="Validate GPT-4o as an evaluator for document retrieval")
    
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed MS MARCO data")
    
    parser.add_argument("--single_eval", action="store_true",
                        help="Run single passage evaluation validation")
    
    parser.add_argument("--ranking_eval", action="store_true",
                        help="Run passage ranking evaluation validation")
    
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples/queries to evaluate")
    
    parser.add_argument("--passages_per_query", type=int, default=5,
                        help="Number of passages to rank per query (for ranking eval)")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it by running: export OPENAI_API_KEY=your_key_here")
        return
    
    # Run the selected evaluations
    if args.single_eval or (not args.single_eval and not args.ranking_eval):
        print("\n=== Running Single Passage Evaluation Validation ===")
        single_results = validate_gpt4o_evaluation(
            args.data_dir, 
            num_samples=args.num_samples
        )
        print(f"Results saved to gpt4o_validation_results.csv")
    
    if args.ranking_eval or (not args.single_eval and not args.ranking_eval):
        print("\n=== Running Passage Ranking Evaluation Validation ===")
        ranking_results = validate_ranking_capability(
            args.data_dir,
            num_queries=args.num_samples,
            passages_per_query=args.passages_per_query
        )
        print(f"Results saved to gpt4o_ranking_validation.csv")
        print(f"Visualization saved to gpt4o_ranking_results.png")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()