"""Metrics collection for tracking performance."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    
    prompt_length: int
    response_length: int
    latency: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    model_endpoint: Optional[str] = None


@dataclass
class BatchMetrics:
    """Metrics for a batch of requests."""
    
    batch_id: int
    batch_size: int
    successful_requests: int
    failed_requests: int
    total_latency: float
    start_time: datetime
    end_time: datetime
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_requests + self.failed_requests
        return self.successful_requests / total if total > 0 else 0.0
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency per request."""
        return self.total_latency / self.batch_size if self.batch_size > 0 else 0.0
    
    @property
    def throughput(self) -> float:
        """Calculate throughput in requests per second."""
        duration = (self.end_time - self.start_time).total_seconds()
        return self.batch_size / duration if duration > 0 else 0.0


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.request_metrics: List[RequestMetrics] = []
        self.batch_metrics: List[BatchMetrics] = []
        self.start_time = datetime.now()
    
    def record_request(self, metrics: RequestMetrics) -> None:
        """Record metrics for a single request."""
        self.request_metrics.append(metrics)
    
    def record_batch(self, metrics: BatchMetrics) -> None:
        """Record metrics for a batch."""
        self.batch_metrics.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.request_metrics:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "average_latency": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "total_duration": 0.0,
                "overall_throughput": 0.0,
            }
        
        successful = [m for m in self.request_metrics if m.success]
        failed = [m for m in self.request_metrics if not m.success]
        latencies = [m.latency for m in self.request_metrics]
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.5)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_requests": len(self.request_metrics),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(self.request_metrics),
            "average_latency": statistics.mean(latencies),
            "p50_latency": sorted_latencies[p50_idx] if sorted_latencies else 0.0,
            "p95_latency": sorted_latencies[p95_idx] if sorted_latencies else 0.0,
            "p99_latency": sorted_latencies[p99_idx] if sorted_latencies else 0.0,
            "total_duration": total_duration,
            "overall_throughput": len(self.request_metrics) / total_duration if total_duration > 0 else 0.0,
        }
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary of batch metrics."""
        if not self.batch_metrics:
            return {
                "total_batches": 0,
                "average_batch_size": 0.0,
                "average_success_rate": 0.0,
                "average_throughput": 0.0,
            }
        
        return {
            "total_batches": len(self.batch_metrics),
            "average_batch_size": statistics.mean(b.batch_size for b in self.batch_metrics),
            "average_success_rate": statistics.mean(b.success_rate for b in self.batch_metrics),
            "average_throughput": statistics.mean(b.throughput for b in self.batch_metrics),
        }
    
    def get_endpoint_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary by model endpoint."""
        endpoint_metrics = {}
        
        for metric in self.request_metrics:
            if metric.model_endpoint:
                if metric.model_endpoint not in endpoint_metrics:
                    endpoint_metrics[metric.model_endpoint] = {
                        "requests": [],
                        "latencies": [],
                        "successes": 0,
                        "failures": 0,
                    }
                
                endpoint_metrics[metric.model_endpoint]["requests"].append(metric)
                endpoint_metrics[metric.model_endpoint]["latencies"].append(metric.latency)
                
                if metric.success:
                    endpoint_metrics[metric.model_endpoint]["successes"] += 1
                else:
                    endpoint_metrics[metric.model_endpoint]["failures"] += 1
        
        # Calculate summary for each endpoint
        summary = {}
        for endpoint, data in endpoint_metrics.items():
            total = data["successes"] + data["failures"]
            summary[endpoint] = {
                "total_requests": total,
                "success_rate": data["successes"] / total if total > 0 else 0.0,
                "average_latency": statistics.mean(data["latencies"]) if data["latencies"] else 0.0,
                "p95_latency": sorted(data["latencies"])[int(len(data["latencies"]) * 0.95)] if data["latencies"] else 0.0,
            }
        
        return summary