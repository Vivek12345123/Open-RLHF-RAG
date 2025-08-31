#!/usr/bin/env python3
import os
import json
import argparse
import logging
import pandas as pd
from datetime import datetime
from subprocess import run, PIPE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_evaluation(model_name, datasets, max_samples=200, output_dir="./results", use_reward=True):
    """Run evaluation on specified datasets"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset configurations
    dataset_configs = {
        "microsoft/ms_marco": {
            "split": "dev",
            "description": "Passage ranking dataset with real-world queries from Bing search"
        },
        "mwong/fever-evidence-related": {
            "split": "validation",
            "description": "Fact verification dataset with claims to be verified against evidence"
        },
        "wandb/RAGTruth-processed": {
            "split": "train", # Using train split as test may not exist
            "description": "RAGTruth benchmark for evaluating RAG system capabilities"
        },
        "hotpot_qa": {
            "config": "distractor",
            "split": "validation",
            "description": "Multi-hop reasoning QA requiring information from multiple documents"
        },
        "squad_v2": {
            "split": "validation",
            "description": "Reading comprehension with unanswerable questions"
        },
        "trivia_qa": {
            "config": "rc.nocontext",
            "split": "validation",
            "description": "Reading comprehension with trivia questions"
        }
    }
    
    # Prepare commands
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"Starting evaluation for {dataset}")
        config = dataset_configs.get(dataset, {})
        
        # Build command
        cmd = [
            "python", "rag_evaluator.py",
            "--model_name", model_name,
            "--datasets", dataset,
            "--max_samples", str(max_samples),
            "--output_dir", f"{output_dir}/{dataset.replace('/', '_')}",
            "--retriever_type", "mock" # Using mock retriever for consistent evaluation
        ]
        
        # Add reward model if requested
        if use_reward:
            cmd.extend(["--reward_model_path", "OpenRLHF/Llama-3-8b-rm-mixture"])
            
        # Add specific split if configured
        if "split" in config:
            cmd.extend(["--splits", config["split"]])
            
        # Run evaluation
        logger.info(f"Running: {' '.join(cmd)}")
        start_time = datetime.now()
        
        result = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log output
        logger.info(f"Finished {dataset} in {duration:.2f} seconds")
        
        if result.returncode != 0:
            logger.error(f"Error evaluating {dataset}: {result.stderr}")
        else:
            logger.info(f"Successfully evaluated {dataset}")
            
        # Load results for this dataset
        try:
            dataset_dir = f"{output_dir}/{dataset.replace('/', '_')}"
            result_file = f"{dataset_dir}/{dataset.replace('/', '_')}_results.json"
            
            with open(result_file, 'r') as f:
                dataset_results = json.load(f)
                
            all_results[dataset] = dataset_results
            
        except Exception as e:
            logger.error(f"Error loading results for {dataset}: {e}")
    
    # Generate combined report
    generate_combined_report(all_results, output_dir, dataset_configs)
    
def generate_combined_report(all_results, output_dir, dataset_configs):
    """Generate a comprehensive report combining all dataset results"""
    # Create results directory
    report_dir = f"{output_dir}/combined_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # Create summary table
    summary_data = []
    
    for dataset, results in all_results.items():
        if "aggregate_metrics" not in results:
            continue
            
        metrics = results["aggregate_metrics"]
        dataset_info = results["dataset_info"]
        
        row = {
            "Dataset": dataset,
            "Description": dataset_configs.get(dataset, {}).get("description", ""),
            "Samples": dataset_info["num_processed"],
            "F1 Score": f"{metrics.get('avg_f1', 0.0):.3f} ± {metrics.get('std_f1', 0.0):.3f}",
            "F1 (Raw)": metrics.get('avg_f1', 0.0),
            "ROUGE-1": f"{metrics.get('avg_rouge1', 0.0):.3f}",
            "ROUGE-1 (Raw)": metrics.get('avg_rouge1', 0.0),
            "ROUGE-2": f"{metrics.get('avg_rouge2', 0.0):.3f}",
            "ROUGE-2 (Raw)": metrics.get('avg_rouge2', 0.0),
            "ROUGE-L": f"{metrics.get('avg_rougeL', 0.0):.3f}",
            "ROUGE-L (Raw)": metrics.get('avg_rougeL', 0.0),
            "Exact Match": f"{metrics.get('avg_exact_match', 0.0):.3f}",
            "Exact Match (Raw)": metrics.get('avg_exact_match', 0.0),
            "Reward": f"{metrics.get('avg_reward', 0.0):.3f}" if 'avg_reward' in metrics else "N/A",
            "Reward (Raw)": metrics.get('avg_reward', 0.0) if 'avg_reward' in metrics else None
        }
        summary_data.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV and JSON
    summary_df.to_csv(f"{report_dir}/summary_table.csv", index=False)
    
    # Create pretty formatted version for research paper inclusion
    display_df = summary_df[["Dataset", "Samples", "F1 Score", "ROUGE-1", "ROUGE-L", "Exact Match", "Reward"]]
    display_df.to_csv(f"{report_dir}/paper_ready_table.csv", index=False)
    
    # Save raw metrics for further analysis
    raw_metrics = {
        dataset: {
            "F1": row["F1 (Raw)"],
            "ROUGE-1": row["ROUGE-1 (Raw)"],
            "ROUGE-2": row["ROUGE-2 (Raw)"],
            "ROUGE-L": row["ROUGE-L (Raw)"],
            "Exact Match": row["Exact Match (Raw)"],
            "Reward": row["Reward (Raw)"],
            "Samples": row["Samples"],
            "Description": row["Description"]
        }
        for dataset, row in zip(summary_df["Dataset"], summary_data)
    }
    
    with open(f"{report_dir}/raw_metrics.json", "w") as f:
        json.dump(raw_metrics, f, indent=2)
    
    # Create a LaTeX table for direct inclusion in research papers
    latex_table = generate_latex_table(display_df)
    with open(f"{report_dir}/latex_table.tex", "w") as f:
        f.write(latex_table)
    
    # Generate detailed report with analysis
    generate_detailed_report(all_results, raw_metrics, report_dir)
    
    logger.info(f"Combined report generated in {report_dir}")

def generate_latex_table(df):
    """Generate a LaTeX table for research papers"""
    latex = "\\begin{table}[h]\n\\centering\n\\caption{OpenRLHF-RAG Evaluation Results}\n\\label{tab:rag-results}\n"
    latex += "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    latex += "Dataset & Samples & F1 Score & ROUGE-1 & ROUGE-L & Exact Match & Reward \\\\\n\\midrule\n"
    
    for _, row in df.iterrows():
        latex += f"{row['Dataset'].replace('/', '\\_')} & {row['Samples']} & {row['F1 Score']} & {row['ROUGE-1']} & {row['ROUGE-L']} & {row['Exact Match']} & {row['Reward']} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"
    return latex

def generate_detailed_report(all_results, raw_metrics, report_dir):
    """Generate a detailed analysis report in markdown format"""
    report = "# OpenRLHF-RAG Evaluation Report\n\n"
    report += f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    # Overall statistics
    report += "## Overall Performance Summary\n\n"
    
    avg_f1 = sum(metrics["F1"] for metrics in raw_metrics.values() if metrics["F1"] is not None) / len(raw_metrics)
    avg_rouge1 = sum(metrics["ROUGE-1"] for metrics in raw_metrics.values() if metrics["ROUGE-1"] is not None) / len(raw_metrics)
    avg_rougeL = sum(metrics["ROUGE-L"] for metrics in raw_metrics.values() if metrics["ROUGE-L"] is not None) / len(raw_metrics)
    avg_em = sum(metrics["Exact Match"] for metrics in raw_metrics.values() if metrics["Exact Match"] is not None) / len(raw_metrics)
    
    report += f"- **Average F1 Score across all datasets**: {avg_f1:.3f}\n"
    report += f"- **Average ROUGE-1 Score**: {avg_rouge1:.3f}\n"
    report += f"- **Average ROUGE-L Score**: {avg_rougeL:.3f}\n"
    report += f"- **Average Exact Match Rate**: {avg_em:.3f}\n"
    report += f"- **Total samples evaluated**: {sum(metrics['Samples'] for metrics in raw_metrics.values())}\n\n"
    
    # Per-dataset analysis
    report += "## Dataset-Specific Analysis\n\n"
    
    for dataset, metrics in raw_metrics.items():
        report += f"### {dataset}\n\n"
        report += f"*{metrics['Description']}*\n\n"
        report += f"- **Samples evaluated**: {metrics['Samples']}\n"
        report += f"- **F1 Score**: {metrics['F1']:.3f}\n"
        report += f"- **ROUGE-1 Score**: {metrics['ROUGE-1']:.3f}\n"
        report += f"- **ROUGE-2 Score**: {metrics['ROUGE-2']:.3f}\n"
        report += f"- **ROUGE-L Score**: {metrics['ROUGE-L']:.3f}\n"
        report += f"- **Exact Match Rate**: {metrics['Exact Match']:.3f}\n"
        
        if metrics['Reward'] is not None:
            report += f"- **Reward Score**: {metrics['Reward']:.3f}\n"
        
        report += "\n"
    
    # Methodology
    report += "## Evaluation Methodology\n\n"
    report += "This evaluation was conducted using the OpenRLHF-RAG framework with the following methodology:\n\n"
    report += "1. **Retrieval**: Mock retriever was used to ensure consistent evaluation across datasets\n"
    report += "2. **Generation**: LLM was used to generate responses based on retrieved context and questions\n"
    report += "3. **Metrics**: F1, ROUGE, Exact Match, and Reward scores were calculated\n"
    report += "4. **Datasets**: Six diverse datasets were used covering different aspects of RAG performance\n\n"
    
    # Write report
    with open(f"{report_dir}/detailed_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive OpenRLHF-RAG evaluation")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                       help="Model to use for evaluation")
    parser.add_argument("--max_samples", type=int, default=200,
                       help="Number of samples per dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--no_reward", action="store_true",
                       help="Disable reward model evaluation")
    
    args = parser.parse_args()
    
    # All datasets to evaluate
    datasets = [
        "microsoft/ms_marco",
        "mwong/fever-evidence-related",
        "wandb/RAGTruth-processed",
        "hotpot_qa",
        "squad_v2", 
        "trivia_qa"
    ]
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run evaluation
    run_evaluation(
        model_name=args.model_name,
        datasets=datasets,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        use_reward=not args.no_reward
    )
