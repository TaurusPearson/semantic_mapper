"""
llm_integration.py

Integration module for LLM prompt generation with main.py pipeline.
Handles LLM initialization, GPU tracking, and metrics collection.
"""

import os
import json
import torch
from typing import Dict, List, Optional, Tuple

from llm_prompt_generator import LLMPromptGenerator, PromptMetrics, FilteredConfusionMatrix, DEFAULT_FOCUS_CLASSES


def initialize_llm_prompts(
    classes: List[str],
    llm_config: Dict,
    script_dir: str,
    current_prompt_mode: str
) -> Tuple[Optional[Dict[str, List[str]]], Dict, str]:
    """
    Initialize LLM prompt generation if enabled.
    
    Args:
        classes: List of class names from eval_info.yaml
        llm_config: LLM configuration dictionary
        script_dir: Path to script directory for saving outputs
        current_prompt_mode: Current prompt mode setting
        
    Returns:
        Tuple of (llm_prompts dict or None, llm_metrics dict, updated prompt_mode)
    """
    llm_prompts = None
    llm_metrics = {}
    
    # Check if LLM mode should be activated
    # Only run LLM if prompt_mode is 'llm' - ignore 'enabled' flag for other modes
    if current_prompt_mode != 'llm':
        return None, {}, current_prompt_mode
    
    print("\n[LLM] Initializing LLM prompt generation...")
    
    # Track GPU memory before LLM load
    if llm_config.get("track_gpu_usage", True) and torch.cuda.is_available():
        torch.cuda.synchronize()
        llm_metrics["gpu_mem_before_llm_gb"] = torch.cuda.memory_allocated() / 1024**3
    
    force_regenerate = llm_config.get("force_regenerate", False)
    
    # Auto-detect prompts file based on model name if not explicitly set
    prompts_file = llm_config.get("prompts_file")
    if not prompts_file:
        model_short_name = llm_config.get("model_name", "Qwen/Qwen2.5-7B-Instruct").split('/')[-1]
        auto_prompts_file = os.path.join(script_dir, f"llm_prompts_{model_short_name}.json")
        if os.path.exists(auto_prompts_file):
            prompts_file = auto_prompts_file
            print(f"[LLM] Auto-detected prompts file: {prompts_file}")
    
    # Check if pre-generated prompts file exists (skip if force_regenerate)
    if not force_regenerate and prompts_file and os.path.exists(prompts_file):
        print(f"[LLM] Loading pre-generated prompts from: {prompts_file}")
        with open(prompts_file, 'r') as f:
            llm_prompts = json.load(f)
        llm_metrics["source"] = "cached_file"
        llm_metrics["prompts_file"] = prompts_file
        
        # Compute metrics even for cached prompts
        diversity_metrics = PromptMetrics.compute_diversity(llm_prompts)
        length_metrics = PromptMetrics.compute_avg_prompt_length(llm_prompts)
        grounding_metrics = PromptMetrics.compute_grounding_score(llm_prompts)
        
        llm_metrics["prompt_diversity"] = diversity_metrics["global_diversity"]
        llm_metrics["avg_prompt_length"] = length_metrics["global_avg_length"]
        llm_metrics["unique_ngrams"] = diversity_metrics["unique_ngrams"]
        llm_metrics["total_ngrams"] = diversity_metrics["total_ngrams"]
        llm_metrics["grounding_score"] = grounding_metrics["global_grounding"]
        llm_metrics["grounded_prompts"] = grounding_metrics["grounded_count"]
        
        print(f"[LLM] Prompt diversity: {diversity_metrics['global_diversity']:.2%}")
        print(f"[LLM] Avg prompt length: {length_metrics['global_avg_length']:.1f} words")
        print(f"[LLM] Grounding score: {grounding_metrics['global_grounding']:.1%} ({grounding_metrics['grounded_count']}/{grounding_metrics['total_prompts']} prompts)")
    else:
        if force_regenerate:
            print("[LLM] Force regenerate enabled - ignoring cache")
            # Clear per-class cache directory
            cache_dir = os.path.join(os.path.dirname(__file__), "llm_prompt_cache")
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                print(f"[LLM] Cleared cache directory: {cache_dir}")
        
        # Generate prompts using LLM with quality gate
        generator = LLMPromptGenerator(
            model_name=llm_config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            num_prompts=llm_config.get("num_prompts", 5),
            use_cache=not force_regenerate,  # Disable cache if force_regenerate
            similarity_threshold=llm_config.get("similarity_threshold", 0.7),
            max_retries=llm_config.get("max_retries", 3),
            prompt_style=llm_config.get("prompt_style", "cupl")
        )
        
        print(f"[LLM] Generating prompts for {len(classes)} classes...")
        llm_prompts = generator.generate_all_prompts(classes)
        llm_metrics["source"] = "generated"
        
        # Track GPU memory after LLM generation
        if llm_config.get("track_gpu_usage", True) and torch.cuda.is_available():
            torch.cuda.synchronize()
            llm_metrics["gpu_mem_after_llm_gb"] = torch.cuda.memory_allocated() / 1024**3
            llm_metrics["gpu_mem_llm_delta_gb"] = (
                llm_metrics["gpu_mem_after_llm_gb"] - llm_metrics["gpu_mem_before_llm_gb"]
            )
            print(f"[LLM] GPU memory used by LLM: {llm_metrics['gpu_mem_llm_delta_gb']:.2f} GB")
        
        # Compute prompt quality metrics
        diversity_metrics = PromptMetrics.compute_diversity(llm_prompts)
        length_metrics = PromptMetrics.compute_avg_prompt_length(llm_prompts)
        grounding_metrics = PromptMetrics.compute_grounding_score(llm_prompts)
        
        llm_metrics["prompt_diversity"] = diversity_metrics["global_diversity"]
        llm_metrics["avg_prompt_length"] = length_metrics["global_avg_length"]
        llm_metrics["unique_ngrams"] = diversity_metrics["unique_ngrams"]
        llm_metrics["total_ngrams"] = diversity_metrics["total_ngrams"]
        llm_metrics["per_class_diversity"] = diversity_metrics["per_class_diversity"]
        llm_metrics["grounding_score"] = grounding_metrics["global_grounding"]
        llm_metrics["grounded_prompts"] = grounding_metrics["grounded_count"]
        llm_metrics["ungrounded_examples"] = grounding_metrics["ungrounded_examples"]
        
        print(f"[LLM] Prompt diversity: {diversity_metrics['global_diversity']:.2%}")
        print(f"[LLM] Avg prompt length: {length_metrics['global_avg_length']:.1f} words")
        print(f"[LLM] Grounding score: {grounding_metrics['global_grounding']:.1%} ({grounding_metrics['grounded_count']}/{grounding_metrics['total_prompts']} prompts)")
        
        if grounding_metrics["ungrounded_examples"]:
            print(f"[LLM] Warning: {len(grounding_metrics['ungrounded_examples'])} ungrounded prompts found")
        
        # Save generated prompts
        model_short_name = llm_config['model_name'].split('/')[-1]
        prompts_output = os.path.join(script_dir, f"llm_prompts_{model_short_name}.json")
        generator.export_prompts(llm_prompts, prompts_output)
        llm_metrics["prompts_file"] = prompts_output
        
        # Cleanup LLM to free GPU memory
        del generator
        torch.cuda.empty_cache()
        
        if llm_config.get("track_gpu_usage", True) and torch.cuda.is_available():
            torch.cuda.synchronize()
            llm_metrics["gpu_mem_after_cleanup_gb"] = torch.cuda.memory_allocated() / 1024**3
            print(f"[LLM] GPU memory after cleanup: {llm_metrics['gpu_mem_after_cleanup_gb']:.2f} GB")
    
    # Update prompt mode
    updated_mode = 'llm'
    print(f"[LLM] Loaded {len(llm_prompts)} class prompts. Prompt mode set to 'llm'.")
    
    return llm_prompts, llm_metrics, updated_mode


def run_filtered_confusion_analysis(
    confusion_matrix,
    all_class_names: List[str],
    output_dir: str,
    focus_classes: List[str] = None
) -> Dict:
    """
    Run filtered confusion matrix analysis on a subset of classes.
    
    Args:
        confusion_matrix: Full confusion matrix (numpy array)
        all_class_names: List of all class names
        output_dir: Directory to save analysis results
        focus_classes: List of class names to focus on (default: DEFAULT_FOCUS_CLASSES)
        
    Returns:
        Dictionary with filtered confusion matrix results
    """
    focus = focus_classes or DEFAULT_FOCUS_CLASSES
    analyzer = FilteredConfusionMatrix(focus_classes=focus)
    
    results = analyzer.filter_confusion_matrix(
        confusion_matrix, all_class_names, focus
    )
    
    if "error" in results:
        print(f"[FilteredConfusion] {results['error']}")
        return results
    
    # Save results
    if output_dir:
        analysis_path = os.path.join(output_dir, "filtered_confusion_matrix.json")
        analyzer.analyze_and_save(confusion_matrix, all_class_names, analysis_path)
    
    # Print summary
    analyzer.print_summary(results)
    
    return results


def save_llm_metrics(llm_metrics: Dict, output_dir: str):
    """Save LLM metrics to a JSON file."""
    if not llm_metrics:
        return
    
    metrics_path = os.path.join(output_dir, "llm_metrics.json")
    
    # Filter out non-serializable items
    serializable = {}
    for k, v in llm_metrics.items():
        if isinstance(v, (int, float, str, bool, list)):
            serializable[k] = v
        elif isinstance(v, dict):
            # Handle nested dicts (like per_class_diversity)
            serializable[k] = {str(kk): vv for kk, vv in v.items()}
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"[LLM] Metrics saved to: {metrics_path}")


# Default LLM configuration template
DEFAULT_LLM_CONFIG = {
    "enabled": False,                              # Set True to use LLM-generated prompts
    "model_name": "Qwen/Qwen2.5-7B-Instruct",     # HuggingFace model ID
    "num_prompts": 5,                              # Prompts per class
    "cache_prompts": True,                         # Cache generated prompts to disk
    "prompts_file": None,                          # Path to pre-generated prompts JSON (optional)
    "track_gpu_usage": True,                       # Track LLM GPU memory usage
}
