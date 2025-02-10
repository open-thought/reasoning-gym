import argparse
from datetime import datetime
import json
import os
from openai import OpenAI
from typing import Any, Dict, List

from reasoning_gym.factory import DATASETS, create_dataset

class OpenRouterEvaluator:
    def __init__(self, model: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        self.model = model
        self.extra_headers = {}

    def get_model_response(self, prompt: str) -> str:
        """Get response from the model via OpenRouter API."""
        try:
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
            raise

    def evaluate_datasets(self, dataset_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate model on multiple datasets with their respective configurations."""
        all_results = []
        
        for dataset_config in dataset_configs:
            dataset_name = dataset_config.pop('name')
            print(f"\nEvaluating dataset: {dataset_name}")
            
            try:
                # Create dataset with its specific configuration
                data = create_dataset(dataset_name, **dataset_config)
                results = []
                
                for entry in data:
                    try:
                        response = self.get_model_response(entry['question'])
                        score = data.score_answer(answer=response, entry=entry)

                        result = {
                            'question': entry['question'],
                            'expected_answer': entry['answer'],
                            'model_answer': response,
                            'score': score,
                            'metadata': entry['metadata']
                        }
                        results.append(result)
                        print(f"Processed question {len(results)}/{len(data)}. Score: {score}")

                    except Exception as e:
                        print(f"Error processing question: {entry['question']}")
                        print(f"Error: {str(e)}")

                # Calculate aggregate metrics
                total_score = sum(r['score'] for r in results)
                metrics = {
                    'dataset_name': dataset_name,
                    'model': self.model,
                    'size': len(data),
                    'average_score': total_score / len(results) if results else 0,
                    'total_examples': len(results),
                    'timestamp': datetime.now().isoformat(),
                    'config': dataset_config
                }

                all_results.append({
                    'metrics': metrics,
                    'results': results
                })

            except Exception as e:
                print(f"Error evaluating dataset {dataset_name}: {str(e)}")
                continue

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models on reasoning datasets')
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument('--config', required=True, 
                       help='Path to JSON configuration file')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset configurations
    with open(args.config, 'r') as f:
        dataset_configs = json.load(f)

    evaluator = OpenRouterEvaluator(model=args.model)
    all_results = evaluator.evaluate_datasets(dataset_configs)

    # Save results
    output_file = os.path.join(
        args.output_dir,
        f"evaluation_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Save detailed results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create summary
    summary = []
    for result in all_results:
        metrics = result['metrics']
        summary_entry = {
            'dataset_name': metrics['dataset_name'],
            'model': metrics['model'],
            'average_score': metrics['average_score'],
            'total_examples': metrics['total_examples'],
            'timestamp': metrics['timestamp'],
            'config': metrics['config']
        }
        summary.append(summary_entry)

    # Save summary to a separate file
    summary_file = os.path.join(
        args.output_dir,
        f"summary_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\nEvaluation Summary:")
    for entry in summary:
        print(f"\nDataset: {entry['dataset_name']}")
        print(f"Average Score: {entry['average_score']:.2%}")
        print(f"Total Examples: {entry['total_examples']}")
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()