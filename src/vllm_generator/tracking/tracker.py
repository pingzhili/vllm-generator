"""Tracking implementation for monitoring pipeline execution."""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime

from .metrics import MetricsCollector, RequestMetrics, BatchMetrics
from ..utils import get_logger


class ExecutionTracker:
    """Track execution metrics and performance."""
    
    def __init__(self):
        """Initialize execution tracker."""
        self.logger = get_logger("ExecutionTracker")
        self.metrics_collector = MetricsCollector()
        self._batch_start_times: Dict[int, float] = {}
        self._batch_metrics: Dict[int, Dict[str, Any]] = {}
    
    @contextmanager
    def track_request(
        self,
        prompt: str,
        model_endpoint: Optional[str] = None
    ):
        """Track a single request execution."""
        start_time = time.time()
        metrics = RequestMetrics(
            prompt_length=len(prompt),
            response_length=0,
            latency=0.0,
            success=False,
            model_endpoint=model_endpoint
        )
        
        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
        finally:
            metrics.latency = time.time() - start_time
            self.metrics_collector.record_request(metrics)
    
    def start_batch(self, batch_id: int, batch_size: int) -> None:
        """Start tracking a batch."""
        self._batch_start_times[batch_id] = time.time()
        self._batch_metrics[batch_id] = {
            "size": batch_size,
            "successful": 0,
            "failed": 0,
            "start_time": datetime.now()
        }
        self.logger.debug(f"Started tracking batch {batch_id} with {batch_size} items")
    
    def end_batch(self, batch_id: int) -> None:
        """End tracking a batch."""
        if batch_id not in self._batch_start_times:
            self.logger.warning(f"Batch {batch_id} was not started")
            return
        
        end_time = datetime.now()
        start_time = self._batch_metrics[batch_id]["start_time"]
        total_latency = time.time() - self._batch_start_times[batch_id]
        
        metrics = BatchMetrics(
            batch_id=batch_id,
            batch_size=self._batch_metrics[batch_id]["size"],
            successful_requests=self._batch_metrics[batch_id]["successful"],
            failed_requests=self._batch_metrics[batch_id]["failed"],
            total_latency=total_latency,
            start_time=start_time,
            end_time=end_time
        )
        
        self.metrics_collector.record_batch(metrics)
        
        # Clean up
        del self._batch_start_times[batch_id]
        del self._batch_metrics[batch_id]
        
        self.logger.debug(
            f"Completed batch {batch_id}: {metrics.successful_requests}/{metrics.batch_size} "
            f"successful, {metrics.average_latency:.2f}s avg latency"
        )
    
    def record_batch_success(self, batch_id: int, count: int = 1) -> None:
        """Record successful requests in a batch."""
        if batch_id in self._batch_metrics:
            self._batch_metrics[batch_id]["successful"] += count
    
    def record_batch_failure(self, batch_id: int, count: int = 1) -> None:
        """Record failed requests in a batch."""
        if batch_id in self._batch_metrics:
            self._batch_metrics[batch_id]["failed"] += count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of tracked metrics."""
        return {
            "request_summary": self.metrics_collector.get_summary(),
            "batch_summary": self.metrics_collector.get_batch_summary(),
            "endpoint_summary": self.metrics_collector.get_endpoint_summary(),
        }
    
    def log_summary(self) -> None:
        """Log summary metrics."""
        summary = self.get_summary()
        
        # Log request summary
        req_summary = summary["request_summary"]
        self.logger.info(
            f"Request Summary: {req_summary['total_requests']} total, "
            f"{req_summary['success_rate']:.2%} success rate, "
            f"{req_summary['average_latency']:.2f}s avg latency, "
            f"{req_summary['overall_throughput']:.2f} req/s"
        )
        
        # Log batch summary
        batch_summary = summary["batch_summary"]
        if batch_summary["total_batches"] > 0:
            self.logger.info(
                f"Batch Summary: {batch_summary['total_batches']} batches, "
                f"{batch_summary['average_batch_size']:.1f} avg size, "
                f"{batch_summary['average_throughput']:.2f} req/s avg throughput"
            )
        
        # Log endpoint summary
        endpoint_summary = summary["endpoint_summary"]
        for endpoint, stats in endpoint_summary.items():
            self.logger.info(
                f"Endpoint {endpoint}: {stats['total_requests']} requests, "
                f"{stats['success_rate']:.2%} success rate, "
                f"{stats['average_latency']:.2f}s avg latency"
            )