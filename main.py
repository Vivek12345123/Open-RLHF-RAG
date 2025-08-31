import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Optional
from datasets import load_dataset
from rouge_score import rouge_scorer
from collections import Counter
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Try importing different retriever libraries but don't fail if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Install with: pip install faiss-gpu or faiss-cpu")

try:
    from elasticsearch import Elasticsearch
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    warnings.warn("Elasticsearch not available. Install with: pip install elasticsearch")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    warnings.warn("ChromaDB not available. Install with: pip install chromadb")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.generation import GenerationConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluator for RAG systems using OpenRLHF components"""
    
    def __init__(self, 
                 model_name: str,
                 retriever_type: str = "faiss",
                 retriever_params: Dict = None,
                 reward_model_path: str = None,
                 max_length: int = 2048,
                 batch_size: int = 8,
                 device: str = None,
                 generation_config: Dict = None):
        """
        Initialize the RAG evaluator.
        
        Args:
            model_name: Name or path of the LLM to use
            retriever_type: Type of retriever ("faiss", "es", "chroma")
            retriever_params: Parameters for the retriever
            reward_model_path: Path to an optional reward model
            max_length: Maximum sequence length
            batch_size: Batch size for generation
            device: Device to use ("cuda", "cpu")
            generation_config: Additional generation configuration parameters
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.generation_config = generation_config or {}
        
        # Initialize tokenizer and model
        logger.info(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        self.model.eval()
        
        # Initialize reward model if provided
        self.reward_model = None
        if reward_model_path:
            logger.info(f"Loading reward model from {reward_model_path}...")
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_path,
                num_labels=1,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )
            self.reward_model.eval()
        
        # Setup ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize retriever
        self.retriever = self._init_retriever(retriever_type, retriever_params or {})
    
    def _init_retriever(self, retriever_type, params):
        """Initialize the appropriate retriever based on type"""
        if retriever_type == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS is required for FAISS retriever. Install with: pip install faiss-gpu or faiss-cpu")
            
            # Setup FAISS retriever - parameters should include index_path and documents
            index = None
            documents = params.get("documents", [])
            
            if "index_path" in params and os.path.exists(params["index_path"]):
                try:
                    index = faiss.read_index(params["index_path"])
                    logger.info(f"Loaded FAISS index from {params['index_path']}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
            
            return {"type": "faiss", "index": index, "params": params}
            
        elif retriever_type == "es":
            if not ES_AVAILABLE:
                raise ImportError("Elasticsearch is required for ES retriever. Install with: pip install elasticsearch")
            
            # Setup Elasticsearch retriever
            es_config = params.get("connection", {})
            try:
                es = Elasticsearch(**es_config)
                if not es.ping():
                    logger.warning("Could not connect to Elasticsearch")
            except Exception as e:
                logger.error(f"Error connecting to Elasticsearch: {e}")
                es = None
                
            return {"type": "es", "client": es, "params": params}
            
        elif retriever_type == "chroma":
            if not CHROMADB_AVAILABLE:
                raise ImportError("ChromaDB is required for Chroma retriever. Install with: pip install chromadb")
            
            # Setup ChromaDB retriever
            chroma_config = params.get("connection", {})
            try:
                client = chromadb.Client(chroma_config)
            except Exception as e:
                logger.error(f"Error connecting to ChromaDB: {e}")
                client = None
                
            return {"type": "chroma", "client": client, "params": params}
            
        elif retriever_type == "mock":
            # Mock retriever for testing
            return {"type": "mock", "params": params}
            
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents based on the query"""
        retriever = self.retriever
        
        if retriever["type"] == "faiss":
            # FAISS retrieval logic
            if retriever["index"] is None:
                logger.error("FAISS index not built or loaded")
                return []
            
            # Convert query to vector using model embedding
            query_vector = self._get_embedding(query)
            
            # Search in FAISS index
            distances, indices = retriever["index"].search(np.array([query_vector]), top_k)
            
            # Convert results to documents
            documents = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(retriever["params"].get("documents", [])):  # Valid index
                    doc = retriever["params"]["documents"][idx]
                    doc["score"] = float(distances[0][i])
                    documents.append(doc)
            
            return documents
            
        elif retriever["type"] == "es":
            # Elasticsearch retrieval logic
            if retriever["client"] is None:
                logger.error("Elasticsearch client not connected")
                return []
                
            es_query = {
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": top_k
            }
            
            # Add any custom query parameters
            if "query_params" in retriever["params"]:
                es_query.update(retriever["params"]["query_params"])
            
            # Execute search
            try:
                index_name = retriever["params"].get("index_name", "documents")
                results = retriever["client"].search(index=index_name, body=es_query)
                
                # Convert results to documents
                documents = []
                for hit in results["hits"]["hits"]:
                    doc = {
                        "id": hit["_id"],
                        "content": hit["_source"].get("content", ""),
                        "title": hit["_source"].get("title", ""),
                        "score": hit["_score"]
                    }
                    documents.append(doc)
                
                return documents
            except Exception as e:
                logger.error(f"Error during Elasticsearch search: {e}")
                return []
            
        elif retriever["type"] == "chroma":
            # ChromaDB retrieval logic
            if retriever["client"] is None:
                logger.error("ChromaDB client not connected")
                return []
                
            try:
                collection_name = retriever["params"].get("collection_name", "documents")
                collection = retriever["client"].get_collection(collection_name)
                
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                
                # Convert results to documents
                documents = []
                for i, (doc_id, content, metadata, distance) in enumerate(zip(
                    results.get("ids", [[]])[0],
                    results.get("documents", [[]])[0],
                    results.get("metadatas", [[]])[0],
                    results.get("distances", [[]])[0]
                )):
                    doc = {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": distance
                    }
                    documents.append(doc)
                
                return documents
            except Exception as e:
                logger.error(f"Error during ChromaDB search: {e}")
                return []
                
        elif retriever["type"] == "mock":
            # Mock retriever for testing
            mock_docs = retriever["params"].get("mock_documents", [])
            if not mock_docs:
                # Generate some mock documents related to the query
                mock_docs = [
                    {
                        "id": f"doc_{i}",
                        "title": f"Mock Document {i}",
                        "content": f"This is mock content related to: {query}. Information point {i}.",
                        "score": 1.0 - (i * 0.1)
                    }
                    for i in range(min(top_k, 5))
                ]
            return mock_docs[:top_k]
        
        return []
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=self.max_length).to(self.device)
        
        with torch.no_grad():
            # Use model's embedding layer to get embeddings
            embeddings = self.model.get_input_embeddings()(inputs["input_ids"])
            
            # Average the embeddings across sequence length
            embedding = torch.mean(embeddings, dim=1).cpu().numpy()
        
        return embedding
    
    def generate_answer(self, query: str, context: str = None) -> str:
        """Generate an answer using the model"""
        # Prepare prompt with context if available
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                              max_length=self.max_length).to(self.device)
        
        # Configure generation parameters
        default_gen_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # Override with any user-provided generation config
        gen_params = {**default_gen_config, **self.generation_config}
        gen_config = GenerationConfig(**gen_params)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode output, extracting only the generated part (not input)
        generated_text = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return generated_text.strip()
    
    def compute_reward(self, query: str, response: str, context: str = None) -> float:
        """Compute reward score for a response"""
        if not self.reward_model:
            logger.warning("No reward model available")
            return 0.0
        
        # Prepare input for reward model
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: {response}"
        else:
            prompt = f"Question: {query}\n\nAnswer: {response}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.max_length, padding="max_length").to(self.device)
        
        with torch.no_grad():
            # Get reward score
            outputs = self.reward_model(**inputs)
            reward = outputs.logits.item()
        
        return reward
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for evaluation"""
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def compute_f1_score(self, prediction: str, reference: str) -> float:
        """Compute F1 score between prediction and reference"""
        pred_tokens = self.normalize_text(prediction).split()
        ref_tokens = self.normalize_text(reference).split()
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def compute_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute various metrics between prediction and reference"""
        metrics = {}
        
        # F1 score
        metrics["f1"] = self.compute_f1_score(prediction, reference)
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge_scorer.score(reference, prediction)
            metrics["rouge1"] = rouge_scores["rouge1"].fmeasure
            metrics["rouge2"] = rouge_scores["rouge2"].fmeasure
            metrics["rougeL"] = rouge_scores["rougeL"].fmeasure
        except Exception as e:
            logger.warning(f"Error computing ROUGE scores: {e}")
            metrics["rouge1"] = metrics["rouge2"] = metrics["rougeL"] = 0.0
        
        # Exact match (after normalization)
        metrics["exact_match"] = float(self.normalize_text(prediction) == self.normalize_text(reference))
        
        return metrics
    
    def evaluate_dataset(self, dataset_name_or_path: str, split: str = "test", 
                         max_samples: int = None, retrieval_only: bool = False,
                         query_key: str = None, answer_key: str = None) -> Dict:
        """
        Evaluate RAG system on a dataset
        
        Args:
            dataset_name_or_path: HuggingFace dataset name or path
            split: Dataset split to use
            max_samples: Maximum number of samples to evaluate
            retrieval_only: Whether to only evaluate retrieval without generation
            query_key: Key for the query field in the dataset (auto-detected if None)
            answer_key: Key for the answer field in the dataset (auto-detected if None)
        """
        logger.info(f"Loading dataset {dataset_name_or_path}, split {split}...")
        
        # Load dataset
        try:
            if "@" in dataset_name_or_path:
                data_type, data_path = dataset_name_or_path.split("@", 1)
                if data_type == "json":
                    dataset = load_dataset(data_type, data_files={split: f"{data_path}/{split}.jsonl"}, split=split)
                else:
                    dataset = load_dataset(data_type, data_path, split=split)
            else:
                dataset = load_dataset(dataset_name_or_path, split=split)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {"error": str(e)}
        
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Auto-detect query and answer keys if not provided
        if not query_key or not answer_key:
            first_item = dataset[0] if len(dataset) > 0 else {}
            
            # Find the most likely query key
            query_candidates = ["query", "question", "input", "prompt", "context_messages"]
            if not query_key:
                for candidate in query_candidates:
                    if candidate in first_item:
                        query_key = candidate
                        break
                if not query_key and first_item:
                    logger.warning(f"Could not auto-detect query key. Available keys: {list(first_item.keys())}")
                    query_key = list(first_item.keys())[0]  # Just use the first key
            
            # Find the most likely answer key
            answer_candidates = ["answer", "output", "response", "ground_truth", "chosen"]
            if not answer_key:
                for candidate in answer_candidates:
                    if candidate in first_item:
                        answer_key = candidate
                        break
                if not answer_key and first_item:
                    logger.warning(f"Could not auto-detect answer key. Available keys: {list(first_item.keys())}")
                    answer_key = list(first_item.keys())[-1]  # Just use the last key
        
        logger.info(f"Using query_key='{query_key}', answer_key='{answer_key}'")
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")
        
        results = []
        
        # Process dataset
        for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
            try:
                # Extract query and reference answer
                query = self._extract_field(item, query_key)
                reference = self._extract_field(item, answer_key)
                
                # Skip items without query or reference
                if not query or not reference:
                    logger.warning(f"Skipping item {i}: missing query or reference")
                    continue
                
                # Retrieve documents
                retrieved_docs = self.retrieve_documents(query, top_k=5)
                
                # Format context from retrieved documents
                context = self._format_context(retrieved_docs)
                
                if not retrieval_only:
                    # Generate answer
                    prediction = self.generate_answer(query, context)
                    
                    # Compute reward if available
                    reward = self.compute_reward(query, prediction, context)
                    
                    # Compute metrics
                    metrics = self.compute_metrics(prediction, reference)
                    
                    result = {
                        "id": i,
                        "query": query,
                        "reference": reference,
                        "prediction": prediction,
                        "num_retrieved_docs": len(retrieved_docs),
                        "reward": reward,
                        **metrics
                    }
                else:
                    # Only evaluate retrieval
                    result = {
                        "id": i,
                        "query": query,
                        "reference": reference,
                        "num_retrieved_docs": len(retrieved_docs),
                        "retrieved_content": context[:200] + "..." if len(context) > 200 else context
                    }
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                continue
        
        # Compute aggregate metrics
        agg_metrics = {}
        
        if not retrieval_only and results:
            for metric in ["f1", "rouge1", "rouge2", "rougeL", "exact_match", "reward"]:
                values = [r[metric] for r in results if metric in r]
                if values:
                    agg_metrics[f"avg_{metric}"] = sum(values) / len(values)
                    agg_metrics[f"min_{metric}"] = min(values)
                    agg_metrics[f"max_{metric}"] = max(values)
                    agg_metrics[f"std_{metric}"] = np.std(values)
        
        return {
            "results": results,
            "aggregate_metrics": agg_metrics,
            "dataset_info": {
                "name": dataset_name_or_path,
                "split": split,
                "num_samples": len(dataset),
                "num_processed": len(results),
                "query_key": query_key,
                "answer_key": answer_key
            }
        }
    
    def _extract_field(self, item: Dict, key: str) -> str:
        """Extract a field from a dataset item, handling complex structures"""
        if not key or key not in item:
            return ""
        
        value = item[key]
        
        # Handle list of messages (like in chat datasets)
        if isinstance(value, list) and all(isinstance(msg, dict) for msg in value):
            # Try to extract from chat format
            text_parts = []
            for msg in value:
                if "content" in msg:
                    text_parts.append(msg["content"])
            return " ".join(text_parts)
        
        # Handle nested dictionaries
        if isinstance(value, dict):
            # Try common fields
            for field in ["text", "content", "value"]:
                if field in value:
                    return value[field]
            # Just concatenate all string values
            return " ".join(str(v) for v in value.values() if isinstance(v, (str, int, float)))
        
        # Handle lists of strings
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return " ".join(value)
        
        # Default case
        return str(value)
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents into a context string"""
        context_parts = []
        
        for i, doc in enumerate(docs):
            # Extract title and content
            title = doc.get("title", doc.get("name", f"Document {i+1}"))
            content = doc.get("content", doc.get("text", ""))
            
            if content:
                context_parts.append(f"[{title}] {content}")
        
        return "\n\n".join(context_parts)
    
    def evaluate_datasets(self, datasets_config: List[Dict], output_dir: str = "./results") -> Dict:
        """Evaluate RAG system on multiple datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        for config in datasets_config:
            dataset_name = config.get("name")
            split = config.get("split", "test")
            max_samples = config.get("max_samples")
            retrieval_only = config.get("retrieval_only", False)
            query_key = config.get("query_key")
            answer_key = config.get("answer_key")
            
            logger.info(f"Evaluating dataset {dataset_name}, split {split}")
            
            result = self.evaluate_dataset(
                dataset_name_or_path=dataset_name,
                split=split,
                max_samples=max_samples,
                retrieval_only=retrieval_only,
                query_key=query_key,
                answer_key=answer_key
            )
            
            all_results[dataset_name] = result
            
            # Save individual dataset results
            dataset_filename = dataset_name.replace("/", "_").replace("@", "_")
            with open(f"{output_dir}/{dataset_filename}_results.json", "w") as f:
                json.dump(result, f, indent=2)
        
        # Generate comparison table
        comparison_table = self._generate_comparison_table(all_results)
        comparison_table.to_csv(f"{output_dir}/comparison_table.csv", index=False)
        
        # Generate plots
        self._generate_plots(all_results, output_dir)
        
        logger.info(f"Evaluation complete. Results saved to {output_dir}")
        
        return all_results
    
    def _generate_comparison_table(self, all_results: Dict) -> pd.DataFrame:
        """Generate comparison table for all datasets"""
        table_data = []
        
        for dataset_name, result in all_results.items():
            if "aggregate_metrics" not in result or not result["aggregate_metrics"]:
                continue
                
            metrics = result["aggregate_metrics"]
            dataset_info = result["dataset_info"]
            
            row = {
                "Dataset": dataset_name,
                "Samples": dataset_info["num_processed"],
                "F1 Score": f"{metrics.get('avg_f1', 0.0):.3f} ± {metrics.get('std_f1', 0.0):.3f}",
                "ROUGE-1": f"{metrics.get('avg_rouge1', 0.0):.3f}",
                "ROUGE-L": f"{metrics.get('avg_rougeL', 0.0):.3f}",
                "Exact Match": f"{metrics.get('avg_exact_match', 0.0):.3f}",
                "Reward": f"{metrics.get('avg_reward', 0.0):.3f}" if 'avg_reward' in metrics else "N/A"
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def _generate_plots(self, all_results: Dict, output_dir: str):
        """Generate performance visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. F1 Scores by Dataset
        datasets = []
        f1_scores = []
        
        for dataset_name, result in all_results.items():
            if "results" not in result or not result["results"]:
                continue
                
            f1_vals = [r.get("f1", 0.0) for r in result["results"] if "f1" in r]
            if f1_vals:
                datasets.append(dataset_name)
                f1_scores.append(f1_vals)
        
        if datasets and f1_scores:
            axes[0, 0].boxplot(f1_scores, labels=datasets)
            axes[0, 0].set_title('F1 Score Distribution by Dataset')
            axes[0, 0].set_ylabel('F1 Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. ROUGE Scores Comparison
        if datasets:
            rouge1_means = []
            rouge2_means = []
            rougeL_means = []
            
            for dataset_name, result in all_results.items():
                if "aggregate_metrics" not in result or not result["aggregate_metrics"]:
                    continue
                    
                metrics = result["aggregate_metrics"]
                rouge1_means.append(metrics.get("avg_rouge1", 0.0))
                rouge2_means.append(metrics.get("avg_rouge2", 0.0))
                rougeL_means.append(metrics.get("avg_rougeL", 0.0))
            
            x = np.arange(len(datasets))
            width = 0.25
            
            if x.size > 0:  # Check if there are datasets to plot
                axes[0, 1].bar(x - width, rouge1_means, width, label='ROUGE-1')
                axes[0, 1].bar(x, rouge2_means, width, label='ROUGE-2')
                axes[0, 1].bar(x + width, rougeL_means, width, label='ROUGE-L')
                axes[0, 1].set_title('ROUGE Scores by Dataset')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(datasets, rotation=45)
                axes[0, 1].legend()
        
        # 3. Exact Match Rates
        if datasets:
            exact_match_rates = []
            
            for dataset_name, result in all_results.items():
                if "aggregate_metrics" not in result or not result["aggregate_metrics"]:
                    continue
                    
                metrics = result["aggregate_metrics"]
                exact_match_rates.append(metrics.get("avg_exact_match", 0.0))
            
            axes[1, 0].bar(datasets, exact_match_rates, alpha=0.8)
            axes[1, 0].set_title('Exact Match Rate by Dataset')
            axes[1, 0].set_ylabel('Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Reward Distribution
        all_rewards = []
        dataset_labels = []
        
        for dataset_name, result in all_results.items():
            if "results" not in result or not result["results"]:
                continue
                
            rewards = [r.get("reward", 0.0) for r in result["results"] if "reward" in r]
            if rewards:
                all_rewards.extend(rewards)
                dataset_labels.extend([dataset_name] * len(rewards))
        
        if all_rewards:
            axes[1, 1].hist(all_rewards, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rag_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {output_dir}/rag_performance_analysis.png")

def get_dataset_config(dataset_name: str) -> Dict:
    """Get dataset-specific configuration based on known datasets"""
    # Dictionary of known dataset configurations
    dataset_configs = {
        "microsoft/ms_marco": {
            "config": "v2.1",
            "split": "dev",
            "query_key": "query",
            "answer_key": "answers"
        },
        "mwong/fever-evidence-related": {
            "split": "validation",
            "query_key": "claim", 
            "answer_key": "label"
        },
        "wandb/RAGTruth-processed": {
            "split": "train",
            "query_key": "question",
            "answer_key": "answer"
        },
        "hotpot_qa": {
            "config": "distractor",
            "split": "validation",
            "query_key": "question",
            "answer_key": "answer"
        },
        "squad_v2": {
            "split": "validation",
            "query_key": "question",
            "answer_key": "answers"
        },
        "trivia_qa": {
            "config": "rc.nocontext",
            "split": "validation",
            "query_key": "question", 
            "answer_key": "answer"
        },
        "natural_questions": {
            "split": "validation",
            "query_key": "question",
            "answer_key": "annotations"
        }
    }
    
    # Extract base dataset name (without config or version)
    base_name = dataset_name.split("/")[-1].split("@")[0]
    
    # Find matching configuration
    for key, config in dataset_configs.items():
        if base_name in key or key in dataset_name:
            return config
    
    # Default configuration
    return {
        "split": "test",
        "query_key": None,  # Auto-detect
        "answer_key": None  # Auto-detect
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG systems with OpenRLHF")
    
    # Model and reward model parameters
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name or path of the LLM to use")
    parser.add_argument("--reward_model_path", type=str,
                        help="Path to reward model (optional)")
                        
    # Retriever parameters
    parser.add_argument("--retriever_type", type=str, default="mock",
                        choices=["faiss", "es", "chroma", "mock"],
                        help="Type of retriever to use (default: mock)")
    parser.add_argument("--faiss_index_path", type=str,
                        help="Path to FAISS index (for faiss retriever)")
    parser.add_argument("--document_path", type=str,
                        help="Path to documents (JSON/JSONL) for retrieval")
                        
    # Dataset parameters
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated list of datasets to evaluate (e.g., 'microsoft/ms_marco,hotpot_qa')")
    parser.add_argument("--splits", type=str,
                        help="Comma-separated list of dataset splits (default: auto-detect per dataset)")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples per dataset (default: 100)")
    parser.add_argument("--query_keys", type=str,
                        help="Comma-separated list of query keys (default: auto-detect per dataset)")
    parser.add_argument("--answer_keys", type=str,
                        help="Comma-separated list of answer keys (default: auto-detect per dataset)")
                        
    # Evaluation parameters
    parser.add_argument("--output_dir", type=str, default="./rag_results",
                        help="Output directory for results (default: ./rag_results)")
    parser.add_argument("--retrieval_only", action="store_true",
                        help="Evaluate only retrieval without generation")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation (default: 8)")
                        
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens to generate (default: 1024)")
    
    args = parser.parse_args()
    
    # Parse datasets
    datasets = [ds.strip() for ds in args.datasets.split(",")]
    
    # Create retriever parameters
    retriever_params = {}
    if args.retriever_type == "faiss" and args.faiss_index_path:
        retriever_params["index_path"] = args.faiss_index_path
        
        # Load documents if available
        if args.document_path:
            try:
                if args.document_path.endswith(".json"):
                    with open(args.document_path, "r") as f:
                        documents = json.load(f)
                elif args.document_path.endswith(".jsonl"):
                    documents = []
                    with open(args.document_path, "r") as f:
                        for line in f:
                            documents.append(json.loads(line))
                else:
                    logger.warning(f"Unsupported document format: {args.document_path}")
                    documents = []
                
                retriever_params["documents"] = documents
                logger.info(f"Loaded {len(documents)} documents from {args.document_path}")
            except Exception as e:
                logger.error(f"Error loading documents: {e}")
    
    # Prepare generation config
    generation_config = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens
    }
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        model_name=args.model_name,
        retriever_type=args.retriever_type,
        retriever_params=retriever_params,
        reward_model_path=args.reward_model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        generation_config=generation_config
    )
    
    # Create dataset configurations
    datasets_config = []
    
    # Parse optional parameters
    splits = args.splits.split(",") if args.splits else []
    query_keys = args.query_keys.split(",") if args.query_keys else []
    answer_keys = args.answer_keys.split(",") if args.answer_keys else []
    
    # Ensure lists have the right length
    if splits and len(splits) == 1 and len(datasets) > 1:
        splits = splits * len(datasets)
    if query_keys and len(query_keys) == 1 and len(datasets) > 1:
        query_keys = query_keys * len(datasets)
    if answer_keys and len(answer_keys) == 1 and len(datasets) > 1:
        answer_keys = answer_keys * len(datasets)
    
    # Build dataset configurations
    for i, dataset in enumerate(datasets):
        # Get default configuration for this dataset
        config = get_dataset_config(dataset)
        
        # Override with command-line arguments if provided
        if splits and i < len(splits):
            config["split"] = splits[i]
        if query_keys and i < len(query_keys):
            config["query_key"] = query_keys[i]
        if answer_keys and i < len(answer_keys):
            config["answer_key"] = answer_keys[i]
        
        # Create the final configuration
        dataset_config = {
            "name": dataset,
            "split": config["split"],
            "query_key": config["query_key"],
            "answer_key": config["answer_key"],
            "max_samples": args.max_samples,
            "retrieval_only": args.retrieval_only
        }
        
        datasets_config.append(dataset_config)
    
    # Run evaluation
    evaluator.evaluate_datasets(datasets_config, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
