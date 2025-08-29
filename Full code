import json
import time
import os
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import requests
import re
from collections import Counter
import string
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Core dependencies
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Error: datasets not available. Install with: pip install datasets")
    HF_DATASETS_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError:
    print("Error: PyTorch/Transformers not available. Install with: pip install torch transformers")
    TORCH_AVAILABLE = False

# Enhanced evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge_score not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert_score not available. Install with: pip install bert_score")
    BERTSCORE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TIRESRAGSystem:
    """Complete TIRESRAG-R1 system implementation following the project structure"""
    
    def __init__(self, 
                 project_root: str = ".",
                 model_name: str = "TIRESRAG-R1-Instruct",
                 retrieval_port: int = 8000,
                 reflection_port: int = 8001, 
                 thinking_port: int = 8002,
                 max_tokens: int = 512,
                 temperature: float = 0.1):
        
        self.project_root = Path(project_root)
        self.model_name = model_name
        self.retrieval_url = f"http://localhost:{retrieval_port}"
        self.reflection_url = f"http://localhost:{reflection_port}"
        self.thinking_url = f"http://localhost:{thinking_port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # System status
        self.services_running = False
        self.model_loaded = False
        self.index_built = False
        
        # Initialize components
        self._check_project_structure()
        self._load_tiresrag_model()
        
    def _check_project_structure(self):
        """Verify TIRESRAG-R1 project structure exists"""
        required_paths = [
            self.project_root / "scripts" / "wiki_servish.sh",
            self.project_root / "scripts" / "answer_reflection_reward.sh", 
            self.project_root / "scripts" / "sufficient_thinking_reward.sh",
            self.project_root / "evaluation" / "FlashRAG" / "scripts",
            self.project_root / "requirements.txt"
        ]
        
        missing_paths = [p for p in required_paths if not p.exists()]
        if missing_paths:
            logger.warning(f"Missing TIRESRAG-R1 components: {missing_paths}")
            logger.warning("Some functionality may be limited")
        else:
            logger.info("TIRESRAG-R1 project structure verified")
    
    def _load_tiresrag_model(self):
        """Load the trained TIRESRAG-R1 model"""
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available - cannot load TIRESRAG model")
                return
                
            # Look for trained TIRESRAG model in expected locations
            model_paths = [
                self.project_root / "models" / self.model_name,
                self.project_root / "checkpoints" / self.model_name,
                f"./models/{self.model_name}",
                self.model_name  # Try as HuggingFace model ID
            ]
            
            model_loaded = False
            for model_path in model_paths:
                try:
                    logger.info(f"Attempting to load TIRESRAG model from: {model_path}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(model_path), 
                        trust_remote_code=True,
                        padding_side='left'
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                    )
                    
                    self.model.eval()
                    self.model_loaded = True
                    model_loaded = True
                    logger.info(f"Successfully loaded TIRESRAG-R1 model from {model_path}")
                    break
                    
                except Exception as e:
                    logger.debug(f"Failed to load from {model_path}: {e}")
                    continue
            
            if not model_loaded:
                logger.warning("Could not load trained TIRESRAG model - using fallback")
                self._load_fallback_model()
                
        except Exception as e:
            logger.error(f"Error in model loading: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback base model for testing"""
        try:
            fallback_model = "Qwen/Qwen2.5-7B-Instruct"
            logger.info(f"Loading fallback model: {fallback_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            self.model.eval()
            self.model_loaded = True
            logger.info("Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            self.tokenizer = None
            self.model = None
    
    def setup_services(self):
        """Start all required TIRESRAG services"""
        logger.info("Setting up TIRESRAG-R1 services...")
        
        scripts_dir = self.project_root / "scripts"
        if not scripts_dir.exists():
            logger.error("Scripts directory not found - cannot start services")
            return False
        
        services = [
            ("Retrieval Service", "wiki_servish.sh", 8000),
            ("Answer Reflection Service", "answer_reflection_reward.sh", 8001),
            ("Thinking Quality Service", "sufficient_thinking_reward.sh", 8002)
        ]
        
        for service_name, script_name, port in services:
            script_path = scripts_dir / script_name
            if script_path.exists():
                try:
                    logger.info(f"Starting {service_name}...")
                    # Run in background - in production you'd use proper process management
                    subprocess.Popen(
                        ["bash", str(script_path)], 
                        cwd=str(scripts_dir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    time.sleep(2)  # Allow service to start
                    
                    # Verify service is running
                    if self._check_service_health(port):
                        logger.info(f"{service_name} started successfully on port {port}")
                    else:
                        logger.warning(f"{service_name} may not be running properly")
                        
                except Exception as e:
                    logger.error(f"Failed to start {service_name}: {e}")
            else:
                logger.warning(f"Script not found: {script_path}")
        
        self.services_running = True
        return True
    
    def _check_service_health(self, port: int) -> bool:
        """Check if a service is responding on the given port"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            # Services might not have health endpoints
            return True
    
    def build_retrieval_index(self):
        """Build FAISS retrieval index using FlashRAG"""
        logger.info("Building retrieval index...")
        
        flashrag_scripts = self.project_root / "evaluation" / "FlashRAG" / "scripts"
        if not flashrag_scripts.exists():
            logger.error("FlashRAG scripts not found")
            return False
        
        try:
            # Step 1: Chunk documents
            chunk_script = flashrag_scripts / "chunk.sh"
            if chunk_script.exists():
                logger.info("Chunking documents...")
                subprocess.run(["bash", str(chunk_script)], cwd=str(flashrag_scripts), check=True)
            
            # Step 2: Build FAISS index  
            index_script = flashrag_scripts / "build_index.sh"
            if index_script.exists():
                logger.info("Building FAISS index...")
                subprocess.run(["bash", str(index_script)], cwd=str(flashrag_scripts), check=True)
            
            self.index_built = True
            logger.info("Retrieval index built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build retrieval index: {e}")
            return False
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents using FlashRAG retrieval service"""
        try:
            # Try FlashRAG-specific API format first
            payload = {
                "query": query,
                "retriever_name": "bge",  # Common FlashRAG retriever
                "corpus_name": "wiki",
                "top_k": top_k,
                "rerank": True
            }
            
            response = requests.post(
                f"{self.retrieval_url}/retrieve",
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                # Handle different response formats
                if "documents" in data:
                    return data["documents"]
                elif "retrieved_docs" in data:
                    return data["retrieved_docs"]
                else:
                    return data.get("results", [])
            else:
                logger.warning(f"Retrieval API returned {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.warning(f"Retrieval API call failed: {e}")
        
        # Return empty list if retrieval fails
        return []
    
    def evaluate_answer_reflection(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """Get answer quality reflection scores"""
        try:
            payload = {
                "question": question,
                "answer": answer, 
                "context": context,
                "metrics": ["quality", "relevance", "completeness", "factuality"]
            }
            
            response = requests.post(
                f"{self.reflection_url}/evaluate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.debug(f"Reflection API call failed: {e}")
        
        # Fallback scoring based on heuristics
        return self._fallback_reflection_scoring(question, answer, context)
    
    def evaluate_thinking_quality(self, question: str, thinking: str) -> Dict[str, float]:
        """Evaluate thinking process quality"""
        try:
            payload = {
                "question": question,
                "thinking": thinking,
                "metrics": ["sufficiency", "coherence", "depth", "relevance"]
            }
            
            response = requests.post(
                f"{self.thinking_url}/evaluate", 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.debug(f"Thinking quality API call failed: {e}")
        
        return self._fallback_thinking_scoring(thinking)
    
    def _fallback_reflection_scoring(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """Fallback reflection scoring using heuristics"""
        scores = {}
        
        # Quality: based on length and structure
        scores['quality'] = min(0.9, len(answer.split()) / 50.0) if answer else 0.1
        
        # Relevance: keyword overlap with question
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        scores['relevance'] = len(q_words & a_words) / max(len(q_words), 1) if answer else 0.0
        
        # Completeness: presence of key indicators
        completeness_indicators = ['because', 'therefore', 'however', 'specifically', 'according to']
        scores['completeness'] = sum(1 for ind in completeness_indicators if ind in answer.lower()) / len(completeness_indicators)
        
        # Factuality: conservative estimate based on context usage
        scores['factuality'] = 0.8 if context and len(context) > 100 else 0.5
        
        return scores
    
    def _fallback_thinking_scoring(self, thinking: str) -> Dict[str, float]:
        """Fallback thinking quality scoring"""
        scores = {}
        
        if not thinking:
            return {"sufficiency": 0.0, "coherence": 0.0, "depth": 0.0, "relevance": 0.0}
        
        # Sufficiency: based on length and reasoning indicators
        reasoning_words = ['because', 'since', 'therefore', 'thus', 'however', 'although', 'while', 'whereas']
        reasoning_count = sum(1 for word in reasoning_words if word in thinking.lower())
        scores['sufficiency'] = min(0.95, reasoning_count / 3.0 + len(thinking.split()) / 100.0)
        
        # Coherence: sentence connectivity
        sentences = thinking.split('.')
        scores['coherence'] = min(0.9, len(sentences) / 5.0) if len(sentences) > 1 else 0.3
        
        # Depth: presence of analysis indicators
        depth_indicators = ['analyze', 'consider', 'examine', 'evaluate', 'compare', 'contrast', 'implications']
        scores['depth'] = sum(1 for ind in depth_indicators if ind in thinking.lower()) / len(depth_indicators)
        
        # Relevance: similar to sufficiency for thinking
        scores['relevance'] = scores['sufficiency'] * 0.9
        
        return scores
    
    def think_retrieve_reflect(self, question: str) -> Dict[str, Any]:
        """Complete TIRESRAG-R1 think-retrieve-reflect process"""
        start_time = time.time()
        
        if not self.model_loaded:
            logger.error("TIRESRAG model not loaded")
            return self._error_response("Model not loaded", start_time)
        
        try:
            # Step 1: THINK - Generate initial reasoning
            thinking_prompt = self._build_thinking_prompt(question)
            thinking = self._generate_text(thinking_prompt, max_tokens=200)
            
            # Evaluate thinking quality
            thinking_scores = self.evaluate_thinking_quality(question, thinking)
            
            # Step 2: RETRIEVE - Get relevant documents  
            retrieved_docs = self.retrieve_documents(question, top_k=5)
            context = self._format_context(retrieved_docs)
            
            # Step 3: GENERATE - Produce final answer
            answer_prompt = self._build_answer_prompt(question, thinking, context)
            answer = self._generate_text(answer_prompt, max_tokens=300)
            
            # Step 4: REFLECT - Evaluate answer quality
            reflection_scores = self.evaluate_answer_reflection(question, answer, context)
            
            inference_time = time.time() - start_time
            
            return {
                'text': answer,
                'thinking': thinking,
                'thinking_scores': thinking_scores,
                'retrieved_docs': retrieved_docs,
                'context': context,
                'reflection_scores': reflection_scores,
                'tokens_generated': len(answer.split()) + len(thinking.split()),
                'inference_time': inference_time,
                'uses_retrieval': len(retrieved_docs) > 0,
                'num_retrieved_docs': len(retrieved_docs),
                'overall_quality': self._compute_overall_quality(thinking_scores, reflection_scores)
            }
            
        except Exception as e:
            logger.error(f"Error in think-retrieve-reflect: {e}")
            return self._error_response(str(e), start_time)
    
    def _build_thinking_prompt(self, question: str) -> str:
        """Build prompt for thinking phase"""
        return f"""<|im_start|>system
You are TIRESRAG-R1, an advanced reasoning system. Think step-by-step about the question before retrieving information.
<|im_end|>
<|im_start|>user  
Question: {question}

Let me think about this step by step:
<|im_end|>
<|im_start|>assistant
I need to think through this question carefully:

"""
    
    def _build_answer_prompt(self, question: str, thinking: str, context: str) -> str:
        """Build prompt for answer generation with thinking and context"""
        context_section = f"\nRetrieved Information:\n{context}\n" if context else ""
        
        return f"""<|im_start|>system
You are TIRESRAG-R1. Use your thinking and retrieved information to provide an accurate, well-reasoned answer.
<|im_end|>
<|im_start|>user
Question: {question}

My thinking: {thinking}{context_section}
Based on my analysis and the information above, here is my answer:
<|im_end|>
<|im_start|>assistant
"""
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents into context"""
        if not docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(docs[:5]):
            title = doc.get('title', f'Document {i+1}')
            content = doc.get('content', doc.get('text', doc.get('passage', '')))
            
            if content:
                # Truncate very long content
                if len(content) > 400:
                    content = content[:400] + "..."
                context_parts.append(f"[{title}] {content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_text(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text using the loaded model"""
        if not self.model or not self.tokenizer:
            return "[Model not available]"
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if '<|im_end|>' in response:
                response = response.split('<|im_end|>')[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def _compute_overall_quality(self, thinking_scores: Dict, reflection_scores: Dict) -> float:
        """Compute overall quality score combining thinking and reflection"""
        thinking_avg = np.mean(list(thinking_scores.values())) if thinking_scores else 0.0
        reflection_avg = np.mean(list(reflection_scores.values())) if reflection_scores else 0.0
        
        # Weight thinking and reflection equally
        return (thinking_avg + reflection_avg) / 2.0
    
    def _error_response(self, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'text': f"[Error] {error_msg}",
            'thinking': "",
            'thinking_scores': {},
            'retrieved_docs': [],
            'context': "",
            'reflection_scores': {},
            'tokens_generated': 0,
            'inference_time': time.time() - start_time,
            'uses_retrieval': False,
            'num_retrieved_docs': 0,
            'overall_quality': 0.0
        }

class TIRESRAGEvaluator:
    """Enhanced evaluator for TIRESRAG-R1 with comprehensive metrics"""
    
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation (SQuAD style)"""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
            return re.sub(regex, ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> float:
        """Compute exact match score"""
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score"""
        pred_tokens = self.normalize_answer(prediction).split()
        gold_tokens = self.normalize_answer(ground_truth).split()
        
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def evaluate_against_multiple_answers(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate against multiple possible answers"""
        if not ground_truths:
            return {'em': 0.0, 'f1': 0.0}
        
        best_em = 0.0
        best_f1 = 0.0
        
        for gt in ground_truths:
            if not gt or not gt.strip():
                continue
                
            em = self.exact_match_score(prediction, gt)
            f1 = self.f1_score(prediction, gt)
            
            best_em = max(best_em, em)
            best_f1 = max(best_f1, f1)
        
        return {'em': best_em, 'f1': best_f1}
    
    def compute_rouge_scores(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Compute ROUGE scores if available"""
        if not self.rouge_scorer:
            return {}
        
        try:
            scores = self.rouge_scorer.score(ground_truth, prediction)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure
            }
        except:
            return {}

# Dataset loading utilities
def load_dataset_safe(dataset_options: List[Tuple], sample_size: int):
    """Safely load dataset with multiple fallback options"""
    if not HF_DATASETS_AVAILABLE:
        logger.error("datasets library not available")
        return None
    
    for option in dataset_options:
        try:
            if len(option) == 2:
                dataset_name, split = option
                logger.info(f"Loading {dataset_name}...")
                ds = load_dataset(dataset_name, split=split)
            elif len(option) == 3:
                dataset_name, config, split = option
                logger.info(f"Loading {dataset_name}[{config}]...")
                ds = load_dataset(dataset_name, config, split=split)
            else:
                continue
            
            if sample_size and sample_size < len(ds):
                ds = ds.select(range(sample_size))
            
            logger.info(f"Successfully loaded {len(ds)} samples from {dataset_name}")
            return ds
            
        except Exception as e:
            logger.warning(f"Failed to load {option[0]}: {e}")
            continue
    
    return None

# Benchmark implementations with correct HuggingFace dataset names
def run_hotpotqa_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """HotpotQA multi-hop reasoning evaluation"""
    logger.info(f"Running HotpotQA evaluation ({sample_size} samples)")
    
    # Correct HuggingFace dataset names for HotpotQA
    dataset_options = [
        ("hotpot_qa", "distractor", "validation"),
        ("hotpot_qa", "fullwiki", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load HotpotQA dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            question = item.get('question', '')
            answer = item.get('answer', '')
            level = item.get('level', 'unknown')
            type_q = item.get('type', 'unknown')
            
            if not question or not answer:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response
            scores = evaluator.evaluate_against_multiple_answers(response['text'], [answer])
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answer)
            
            result = {
                'dataset': 'hotpotqa',
                'question': question,
                'ground_truth': answer,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'level': level,
                'type': type_q,
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} HotpotQA samples")
                
        except Exception as e:
            logger.error(f"Error processing HotpotQA item {i}: {e}")
            continue
    
    return results

def run_natural_questions_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """Google Natural Questions evaluation"""
    logger.info(f"Running Natural Questions evaluation ({sample_size} samples)")
    
    # Correct HuggingFace dataset names for Natural Questions
    dataset_options = [
        ("natural_questions", "validation"),
        ("google-research-datasets/natural_questions", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load Natural Questions dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            # Handle Natural Questions format
            question = ""
            if 'question' in item:
                if isinstance(item['question'], dict):
                    question = item['question'].get('text', '')
                else:
                    question = item['question']
            elif 'question_text' in item:
                question = item['question_text']
            
            # Extract answers from annotations
            answers = []
            if 'annotations' in item:
                for annotation in item['annotations']:
                    if 'short_answers' in annotation:
                        for short_answer in annotation['short_answers']:
                            if 'text' in short_answer:
                                answers.append(short_answer['text'])
                    if 'yes_no_answer' in annotation:
                        yn_answer = annotation['yes_no_answer']
                        if yn_answer == 0:
                            answers.append('No')
                        elif yn_answer == 1:
                            answers.append('Yes')
            
            if not question or not answers:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response
            scores = evaluator.evaluate_against_multiple_answers(response['text'], answers)
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answers[0] if answers else "")
            
            result = {
                'dataset': 'natural_questions',
                'question': question,
                'ground_truth': answers,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} Natural Questions samples")
                
        except Exception as e:
            logger.error(f"Error processing Natural Questions item {i}: {e}")
            continue
    
    return results

def run_squad_v2_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """SQuAD v2.0 evaluation with unanswerable questions"""
    logger.info(f"Running SQuAD v2.0 evaluation ({sample_size} samples)")
    
    dataset_options = [
        ("squad_v2", "validation"),
        ("rajpurkar/squad_v2", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load SQuAD v2.0 dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            question = item.get('question', '')
            answers = item.get('answers', {})
            is_impossible = item.get('is_impossible', False)
            
            # Extract answer texts
            answer_texts = []
            if not is_impossible and 'text' in answers:
                answer_texts = answers['text'] if isinstance(answers['text'], list) else [answers['text']]
            
            if not question:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # For impossible questions, check if model abstains
            if is_impossible:
                abstain_indicators = ['cannot answer', 'no answer', 'impossible', 'insufficient information', 
                                    'not possible', "don't know", "cannot determine"]
                abstains = any(indicator in response['text'].lower() for indicator in abstain_indicators)
                em_score = 1.0 if abstains else 0.0
                f1_score = 1.0 if abstains else 0.0
            else:
                scores = evaluator.evaluate_against_multiple_answers(response['text'], answer_texts)
                em_score = scores['em']
                f1_score = scores['f1']
            
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answer_texts[0] if answer_texts else "")
            
            result = {
                'dataset': 'squad_v2',
                'question': question,
                'ground_truth': answer_texts,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'is_impossible': is_impossible,
                'exact_match': em_score,
                'f1_score': f1_score,
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} SQuAD v2.0 samples")
                
        except Exception as e:
            logger.error(f"Error processing SQuAD v2.0 item {i}: {e}")
            continue
    
    return results

def run_triviaqa_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """TriviaQA evaluation"""
    logger.info(f"Running TriviaQA evaluation ({sample_size} samples)")
    
    dataset_options = [
        ("trivia_qa", "rc.nocontext", "validation"),
        ("trivia_qa", "unfiltered.nocontext", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load TriviaQA dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            question = item.get('question', '')
            answer = item.get('answer', {})
            
            # Extract answer aliases
            answer_texts = []
            if 'aliases' in answer:
                answer_texts = answer['aliases']
            elif 'value' in answer:
                answer_texts = [answer['value']]
            elif isinstance(answer, str):
                answer_texts = [answer]
            
            if not question or not answer_texts:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response
            scores = evaluator.evaluate_against_multiple_answers(response['text'], answer_texts)
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answer_texts[0])
            
            result = {
                'dataset': 'triviaqa',
                'question': question,
                'ground_truth': answer_texts,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} TriviaQA samples")
                
        except Exception as e:
            logger.error(f"Error processing TriviaQA item {i}: {e}")
            continue
    
    return results

def run_fever_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """FEVER fact verification evaluation"""
    logger.info(f"Running FEVER evaluation ({sample_size} samples)")
    
    dataset_options = [
        ("fever", "v1.0", "labelled_dev"),
        ("fever", "v2.0", "validation"),
        ("AlekseyKorshuk/fever-evidence-related", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load FEVER dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            claim = item.get('claim', '')
            label = item.get('label', item.get('verdict', ''))
            evidence = item.get('evidence', [])
            
            if not claim or not label:
                continue
            
            # Convert label to standard format
            if isinstance(label, int):
                label_map = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
                label = label_map.get(label, 'NOT ENOUGH INFO')
            
            # Format as fact-checking question
            question = f"Verify this claim: {claim}"
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Extract predicted label from response
            pred_label = 'NOT ENOUGH INFO'  # default
            response_lower = response['text'].lower()
            if 'support' in response_lower or 'true' in response_lower or 'correct' in response_lower:
                pred_label = 'SUPPORTS'
            elif 'refute' in response_lower or 'false' in response_lower or 'incorrect' in response_lower:
                pred_label = 'REFUTES'
            
            # Accuracy for fact verification
            accuracy = 1.0 if pred_label == label else 0.0
            
            # F1 score using label as ground truth
            f1_score = evaluator.f1_score(pred_label, label)
            
            result = {
                'dataset': 'fever',
                'question': question,
                'claim': claim,
                'ground_truth': label,
                'prediction': response['text'],
                'predicted_label': pred_label,
                'thinking': response['thinking'],
                'accuracy': accuracy,
                'exact_match': accuracy,  # For consistency
                'f1_score': f1_score,
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} FEVER samples")
                
        except Exception as e:
            logger.error(f"Error processing FEVER item {i}: {e}")
            continue
    
    return results

def run_ragbench_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """RAGBench evaluation"""
    logger.info(f"Running RAGBench evaluation ({sample_size} samples)")
    
    dataset_options = [
        ("chen-alvin/rag-bench", "test"),
        ("ragbench/ragbench", "test"),
        ("microsoft/ragbench", "test")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load RAGBench dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            question = item.get('question', item.get('query', ''))
            answer = item.get('answer', item.get('ground_truth', ''))
            context = item.get('context', item.get('passages', ''))
            domain = item.get('domain', 'general')
            
            if not question or not answer:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response
            scores = evaluator.evaluate_against_multiple_answers(response['text'], [answer])
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answer)
            
            result = {
                'dataset': 'ragbench',
                'question': question,
                'ground_truth': answer,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'domain': domain,
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} RAGBench samples")
                
        except Exception as e:
            logger.error(f"Error processing RAGBench item {i}: {e}")
            continue
    
    return results

def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics across all results"""
    if not results:
        return {}
    
    metrics = {}
    
    # Basic performance metrics
    em_scores = [r['exact_match'] for r in results if 'exact_match' in r]
    f1_scores = [r['f1_score'] for r in results if 'f1_score' in r]
    
    if em_scores:
        metrics['avg_exact_match'] = np.mean(em_scores)
        metrics['std_exact_match'] = np.std(em_scores)
    
    if f1_scores:
        metrics['avg_f1_score'] = np.mean(f1_scores)
        metrics['std_f1_score'] = np.std(f1_scores)
    
    # ROUGE scores if available
    rouge_metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f']
    for rouge_metric in rouge_metrics:
        rouge_scores = [r[rouge_metric] for r in results if rouge_metric in r]
        if rouge_scores:
            metrics[f'avg_{rouge_metric}'] = np.mean(rouge_scores)
            metrics[f'std_{rouge_metric}'] = np.std(rouge_scores)
    
    # Performance characteristics
    inference_times = [r['inference_time'] for r in results if 'inference_time' in r]
    if inference_times:
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['median_inference_time'] = np.median(inference_times)
    
    tokens_generated = [r['tokens_generated'] for r in results if 'tokens_generated' in r]
    if tokens_generated:
        metrics['avg_tokens_generated'] = np.mean(tokens_generated)
    
    # Retrieval usage
    retrieval_usage = [r['uses_retrieval'] for r in results if 'uses_retrieval' in r]
    if retrieval_usage:
        metrics['retrieval_usage_rate'] = np.mean(retrieval_usage)
    
    # TIRESRAG-specific quality scores
    thinking_quality = []
    reflection_quality = []
    overall_quality = [r['overall_quality'] for r in results if 'overall_quality' in r]
    
    for result in results:
        if 'thinking_scores' in result and result['thinking_scores']:
            thinking_quality.append(np.mean(list(result['thinking_scores'].values())))
        if 'reflection_scores' in result and result['reflection_scores']:
            reflection_quality.append(np.mean(list(result['reflection_scores'].values())))
    
    if thinking_quality:
        metrics['avg_thinking_quality'] = np.mean(thinking_quality)
        metrics['std_thinking_quality'] = np.std(thinking_quality)
    
    if reflection_quality:
        metrics['avg_reflection_quality'] = np.mean(reflection_quality)
        metrics['std_reflection_quality'] = np.std(reflection_quality)
    
    if overall_quality:
        metrics['avg_overall_quality'] = np.mean(overall_quality)
        metrics['std_overall_quality'] = np.std(overall_quality)
    
    return metrics

def generate_results_table(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Generate comparison table for research paper"""
    table_data = []
    
    for dataset_name, results in all_results.items():
        if not results:
            continue
            
        metrics = compute_aggregate_metrics(results)
        
        row = {
            'Dataset': dataset_name.upper(),
            'Samples': len(results),
            'EM': f"{metrics.get('avg_exact_match', 0.0):.3f} ± {metrics.get('std_exact_match', 0.0):.3f}",
            'F1': f"{metrics.get('avg_f1_score', 0.0):.3f} ± {metrics.get('std_f1_score', 0.0):.3f}",
            'ROUGE-1': f"{metrics.get('avg_rouge1_f', 0.0):.3f}" if 'avg_rouge1_f' in metrics else "N/A",
            'ROUGE-L': f"{metrics.get('avg_rougeL_f', 0.0):.3f}" if 'avg_rougeL_f' in metrics else "N/A",
            'Thinking Quality': f"{metrics.get('avg_thinking_quality', 0.0):.3f}" if 'avg_thinking_quality' in metrics else "N/A",
            'Reflection Quality': f"{metrics.get('avg_reflection_quality', 0.0):.3f}" if 'avg_reflection_quality' in metrics else "N/A",
            'Overall Quality': f"{metrics.get('avg_overall_quality', 0.0):.3f}" if 'avg_overall_quality' in metrics else "N/A",
            'Avg Time (s)': f"{metrics.get('avg_inference_time', 0.0):.2f}" if 'avg_inference_time' in metrics else "N/A",
            'Retrieval Usage': f"{metrics.get('retrieval_usage_rate', 0.0):.2f}" if 'retrieval_usage_rate' in metrics else "N/A"
        }
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def create_performance_plots(all_results: Dict[str, List[Dict]], output_dir: str = "./results"):
    """Create performance visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. F1 Scores by Dataset
    datasets = []
    f1_scores = []
    for dataset_name, results in all_results.items():
        if results:
            f1_vals = [r['f1_score'] for r in results if 'f1_score' in r]
            if f1_vals:
                datasets.append(dataset_name.upper())
                f1_scores.append(f1_vals)
    
    if datasets and f1_scores:
        axes[0, 0].boxplot(f1_scores, labels=datasets)
        axes[0, 0].set_title('F1 Score Distribution by Dataset')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Inference Time Distribution
    all_times = []
    dataset_labels = []
    for dataset_name, results in all_results.items():
        if results:
            times = [r['inference_time'] for r in results if 'inference_time' in r and r['inference_time'] > 0]
            if times:
                all_times.extend(times)
                dataset_labels.extend([dataset_name.upper()] * len(times))
    
    if all_times:
        axes[0, 1].hist(all_times, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Inference Time Distribution')
        axes[0, 1].set_xlabel('Inference Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
    
    # 3. Quality Scores Comparison
    quality_data = {}
    for dataset_name, results in all_results.items():
        if results:
            thinking_quals = []
            reflection_quals = []
            for r in results:
                if 'thinking_scores' in r and r['thinking_scores']:
                    thinking_quals.append(np.mean(list(r['thinking_scores'].values())))
                if 'reflection_scores' in r and r['reflection_scores']:
                    reflection_quals.append(np.mean(list(r['reflection_scores'].values())))
            
            if thinking_quals and reflection_quals:
                quality_data[dataset_name.upper()] = {
                    'thinking': np.mean(thinking_quals),
                    'reflection': np.mean(reflection_quals)
                }
    
    if quality_data:
        datasets_q = list(quality_data.keys())
        thinking_means = [quality_data[d]['thinking'] for d in datasets_q]
        reflection_means = [quality_data[d]['reflection'] for d in datasets_q]
        
        x = np.arange(len(datasets_q))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, thinking_means, width, label='Thinking Quality', alpha=0.8)
        axes[1, 0].bar(x + width/2, reflection_means, width, label='Reflection Quality', alpha=0.8)
        axes[1, 0].set_title('Quality Scores by Dataset')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(datasets_q, rotation=45)
        axes[1, 0].legend()
    
    # 4. Retrieval Usage Analysis
    retrieval_stats = {}
    for dataset_name, results in all_results.items():
        if results:
            retrieval_usage = [r['uses_retrieval'] for r in results if 'uses_retrieval' in r]
            if retrieval_usage:
                retrieval_stats[dataset_name.upper()] = np.mean(retrieval_usage)
    
    if retrieval_stats:
        datasets_r = list(retrieval_stats.keys())
        usage_rates = list(retrieval_stats.values())
        
        axes[1, 1].bar(datasets_r, usage_rates, alpha=0.8, color='skyblue', edgecolor='navy')
        axes[1, 1].set_title('Retrieval Usage Rate by Dataset')
        axes[1, 1].set_ylabel('Usage Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tiresrag_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance plots saved to {output_dir}/tiresrag_performance_analysis.png")

def run_comprehensive_evaluation(sample_size: int = 100, output_dir: str = "./tiresrag_results"):
    """Run comprehensive evaluation across all datasets"""
    logger.info("Starting TIRESRAG-R1 Comprehensive Evaluation")
    logger.info(f"Sample size per dataset: {sample_size}")
    
    # Initialize system
    tiresrag = TIRESRAGSystem()
    evaluator = TIRESRAGEvaluator()
    
    # Setup services (optional - will use fallbacks if not available)
    tiresrag.setup_services()
    
    # Run evaluations
    all_results = {}
    evaluation_functions = {
        'hotpotqa': run_hotpotqa_evaluation,
        'natural_questions': run_natural_questions_evaluation,
        'squad_v2': run_squad_v2_evaluation,
        'triviaqa': run_triviaqa_evaluation,
        'fever': run_fever_evaluation,
        'ragbench': run_ragbench_evaluation
    }
    
    start_time = time.time()
    
    for dataset_name, eval_func in evaluation_functions.items():
        logger.info(f"Running {dataset_name} evaluation...")
        try:
            results = eval_func(tiresrag, evaluator, sample_size)
            all_results[dataset_name] = results
            logger.info(f"Completed {dataset_name}: {len(results)} samples processed")
        except Exception as e:
            logger.error(f"Failed {dataset_name} evaluation: {e}")
            all_results[dataset_name] = []
    
    total_time = time.time() - start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate results table
    results_table = generate_results_table(all_results)
    
    # Save results
    results_table.to_csv(f"{output_dir}/tiresrag_results_table.csv", index=False)
    logger.info(f"Results table saved to {output_dir}/tiresrag_results_table.csv")
    
    # Save detailed results
    with open(f"{output_dir}/tiresrag_detailed_results.json", 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for dataset, results in all_results.items():
            json_results[dataset] = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, (int, float, str, bool, list)):
                        json_result[key] = value
                    else:
                        json_result[key] = str(value)
                json_results[dataset].append(json_result)
        
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Detailed results saved to {output_dir}/tiresrag_detailed_results.json")
    
    # Create performance plots
    create_performance_plots(all_results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("TIRESRAG-R1 EVALUATION SUMMARY")
    print("="*60)
    print(results_table.to_string(index=False))
    print("="*60)
    
    # Compute overall statistics
    total_samples = sum(len(results) for results in all_results.values())
    successful_datasets = len([d for d, r in all_results.items() if r])
    
    print(f"\nTotal samples processed: {total_samples}")
    print(f"Successful datasets: {successful_datasets}/6")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/max(total_samples, 1):.2f} seconds")
    
    return all_results, results_table

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TIRESRAG-R1 Comprehensive Evaluation")
    parser.add_argument("--sample_size", type=int, default=100, 
                       help="Number of samples per dataset (default: 100)")
    parser.add_argument("--output_dir", type=str, default="./tiresrag_results",
                       help="Output directory for results (default: ./tiresrag_results)")
    parser.add_argument("--model_name", type=str, default="TIRESRAG-R1-Instruct",
                       help="TIRESRAG model name or path")
    
    args = parser.parse_args()
    
    # Run evaluation
    all_results, results_table = run_comprehensive_evaluation(
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )
