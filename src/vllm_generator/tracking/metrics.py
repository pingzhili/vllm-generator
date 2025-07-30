import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np


class MetricsCollector:
    """Collects and aggregates generation metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
        
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and record the duration"""
        if name not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.record(f"{name}_time", duration)
        del self.timers[name]
        return duration
    
    def record(self, metric_name: str, value: Any):
        """Record a metric value"""
        self.metrics[metric_name].append(value)
    
    def record_batch(self, metrics_dict: Dict[str, Any]):
        """Record multiple metrics at once"""
        for name, value in metrics_dict.items():
            self.record(name, value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for name, values in self.metrics.items():
            if not values:
                continue
            
            # Handle numeric metrics
            if all(isinstance(v, (int, float)) for v in values):
                summary[name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "mean": np.mean(values),
                    "std": np.std(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "median": np.median(values)
                }
            else:
                # Non-numeric metrics
                summary[name] = {
                    "count": len(values),
                    "unique": len(set(str(v) for v in values)),
                    "values": values[:10]  # First 10 values
                }
        
        return summary
    
    def get_throughput_metrics(self) -> Dict[str, float]:
        """Calculate throughput metrics"""
        metrics = {}
        
        # Tokens per second
        if "tokens" in self.metrics and "generation_time" in self.metrics:
            total_tokens = sum(self.metrics["tokens"])
            total_time = sum(self.metrics["generation_time"])
            if total_time > 0:
                metrics["tokens_per_second"] = total_tokens / total_time
        
        # Items per second
        if "generation_time" in self.metrics:
            total_items = len(self.metrics["generation_time"])
            total_time = sum(self.metrics["generation_time"])
            if total_time > 0:
                metrics["items_per_second"] = total_items / total_time
        
        return metrics
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error-related metrics"""
        error_metrics = {
            "total_errors": len(self.metrics.get("errors", [])),
            "error_rate": 0.0,
            "error_types": {}
        }
        
        # Calculate error rate
        total_attempts = len(self.metrics.get("generation_time", []))
        if total_attempts > 0:
            error_metrics["error_rate"] = error_metrics["total_errors"] / total_attempts
        
        # Count error types
        for error in self.metrics.get("errors", []):
            error_type = type(error).__name__ if isinstance(error, Exception) else str(error)
            error_metrics["error_types"][error_type] = error_metrics["error_types"].get(error_type, 0) + 1
        
        return error_metrics
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.timers.clear()
    
    def merge(self, other: "MetricsCollector"):
        """Merge metrics from another collector"""
        for name, values in other.metrics.items():
            self.metrics[name].extend(values)


class TokenMetrics:
    """Specialized metrics for token usage tracking"""
    
    def __init__(self):
        self.input_tokens = []
        self.output_tokens = []
        self.total_tokens = []
        
    def record_generation(
        self,
        input_tokens: int,
        output_tokens: int
    ):
        """Record token usage for a generation"""
        self.input_tokens.append(input_tokens)
        self.output_tokens.append(output_tokens)
        self.total_tokens.append(input_tokens + output_tokens)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get token usage summary"""
        if not self.total_tokens:
            return {}
        
        return {
            "total_input_tokens": sum(self.input_tokens),
            "total_output_tokens": sum(self.output_tokens),
            "total_tokens": sum(self.total_tokens),
            "avg_input_tokens": np.mean(self.input_tokens),
            "avg_output_tokens": np.mean(self.output_tokens),
            "avg_total_tokens": np.mean(self.total_tokens),
            "max_input_tokens": max(self.input_tokens),
            "max_output_tokens": max(self.output_tokens),
            "max_total_tokens": max(self.total_tokens)
        }
    
    def estimate_cost(
        self,
        input_price_per_1k: float = 0.01,
        output_price_per_1k: float = 0.03
    ) -> Dict[str, float]:
        """Estimate generation cost based on token usage"""
        total_input = sum(self.input_tokens)
        total_output = sum(self.output_tokens)
        
        input_cost = (total_input / 1000) * input_price_per_1k
        output_cost = (total_output / 1000) * output_price_per_1k
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "cost_per_generation": (input_cost + output_cost) / len(self.total_tokens) if self.total_tokens else 0
        }