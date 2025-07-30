"""Tests for tracking and metrics."""

import pytest
import time
from datetime import datetime

from vllm_generator.tracking import (
    MetricsCollector,
    RequestMetrics,
    BatchMetrics,
    ExecutionTracker
)


class TestRequestMetrics:
    """Test RequestMetrics functionality."""
    
    def test_request_metrics_creation(self):
        """Test creating request metrics."""
        metrics = RequestMetrics(
            prompt_length=100,
            response_length=50,
            latency=1.5,
            success=True,
            model_endpoint="test_endpoint"
        )
        
        assert metrics.prompt_length == 100
        assert metrics.response_length == 50
        assert metrics.latency == 1.5
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.model_endpoint == "test_endpoint"
        assert isinstance(metrics.timestamp, datetime)


class TestBatchMetrics:
    """Test BatchMetrics functionality."""
    
    def test_batch_metrics_properties(self):
        """Test batch metrics calculated properties."""
        start = datetime.now()
        end = datetime.fromtimestamp(start.timestamp() + 10)  # 10 seconds later
        
        metrics = BatchMetrics(
            batch_id=1,
            batch_size=100,
            successful_requests=95,
            failed_requests=5,
            total_latency=150.0,
            start_time=start,
            end_time=end
        )
        
        assert metrics.success_rate == 0.95
        assert metrics.average_latency == 1.5
        assert abs(metrics.throughput - 10.0) < 0.1  # ~10 requests/second
    
    def test_batch_metrics_edge_cases(self):
        """Test batch metrics edge cases."""
        start = datetime.now()
        
        # All failed
        metrics = BatchMetrics(
            batch_id=1,
            batch_size=10,
            successful_requests=0,
            failed_requests=10,
            total_latency=0,
            start_time=start,
            end_time=start  # Same time
        )
        
        assert metrics.success_rate == 0.0
        assert metrics.average_latency == 0.0
        assert metrics.throughput == 0.0  # Division by zero handled


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_record_and_summarize_requests(self):
        """Test recording and summarizing request metrics."""
        collector = MetricsCollector()
        
        # Record some requests
        for i in range(10):
            metrics = RequestMetrics(
                prompt_length=100 + i,
                response_length=50 + i,
                latency=1.0 + i * 0.1,
                success=i < 8,  # 8 successful, 2 failed
                error="Error" if i >= 8 else None
            )
            collector.record_request(metrics)
        
        summary = collector.get_summary()
        
        assert summary["total_requests"] == 10
        assert summary["successful_requests"] == 8
        assert summary["failed_requests"] == 2
        assert summary["success_rate"] == 0.8
        assert 1.4 < summary["average_latency"] < 1.5
        assert summary["p50_latency"] > 0
        assert summary["p95_latency"] > summary["p50_latency"]
    
    def test_empty_collector_summary(self):
        """Test summary with no data."""
        collector = MetricsCollector()
        summary = collector.get_summary()
        
        assert summary["total_requests"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["average_latency"] == 0.0
    
    def test_batch_summary(self):
        """Test batch metrics summary."""
        collector = MetricsCollector()
        
        # Record some batches
        start = datetime.now()
        for i in range(3):
            end = datetime.fromtimestamp(start.timestamp() + 10)
            metrics = BatchMetrics(
                batch_id=i,
                batch_size=100,
                successful_requests=90 + i,
                failed_requests=10 - i,
                total_latency=100.0,
                start_time=start,
                end_time=end
            )
            collector.record_batch(metrics)
        
        summary = collector.get_batch_summary()
        
        assert summary["total_batches"] == 3
        assert summary["average_batch_size"] == 100
        assert 0.9 < summary["average_success_rate"] < 0.95
    
    def test_endpoint_summary(self):
        """Test endpoint-specific summary."""
        collector = MetricsCollector()
        
        # Record requests for different endpoints
        for endpoint in ["endpoint1", "endpoint2"]:
            for i in range(5):
                metrics = RequestMetrics(
                    prompt_length=100,
                    response_length=50,
                    latency=1.0 + i * 0.1,
                    success=True,
                    model_endpoint=endpoint
                )
                collector.record_request(metrics)
        
        summary = collector.get_endpoint_summary()
        
        assert len(summary) == 2
        assert "endpoint1" in summary
        assert "endpoint2" in summary
        
        for endpoint_stats in summary.values():
            assert endpoint_stats["total_requests"] == 5
            assert endpoint_stats["success_rate"] == 1.0
            assert endpoint_stats["average_latency"] > 0


class TestExecutionTracker:
    """Test ExecutionTracker functionality."""
    
    def test_track_request_context_manager(self):
        """Test tracking request with context manager."""
        tracker = ExecutionTracker()
        
        with tracker.track_request("Test prompt", model_endpoint="test") as metrics:
            metrics.response_length = 100
            time.sleep(0.1)  # Simulate processing
        
        summary = tracker.get_summary()
        req_summary = summary["request_summary"]
        
        assert req_summary["total_requests"] == 1
        assert req_summary["successful_requests"] == 1
        assert req_summary["average_latency"] > 0.1
    
    def test_track_request_with_error(self):
        """Test tracking request that fails."""
        tracker = ExecutionTracker()
        
        with pytest.raises(ValueError):
            with tracker.track_request("Test prompt"):
                raise ValueError("Test error")
        
        summary = tracker.get_summary()
        req_summary = summary["request_summary"]
        
        assert req_summary["total_requests"] == 1
        assert req_summary["successful_requests"] == 0
        assert req_summary["failed_requests"] == 1
    
    def test_batch_tracking(self):
        """Test batch tracking methods."""
        tracker = ExecutionTracker()
        
        # Track a batch
        tracker.start_batch(batch_id=1, batch_size=10)
        
        # Record some successes and failures
        tracker.record_batch_success(1, count=8)
        tracker.record_batch_failure(1, count=2)
        
        # End batch
        tracker.end_batch(1)
        
        summary = tracker.get_summary()
        batch_summary = summary["batch_summary"]
        
        assert batch_summary["total_batches"] == 1
        assert batch_summary["average_batch_size"] == 10
    
    def test_batch_tracking_invalid(self):
        """Test ending batch that wasn't started."""
        tracker = ExecutionTracker()
        
        # Should not raise error, just log warning
        tracker.end_batch(999)
        
        summary = tracker.get_summary()
        assert summary["batch_summary"]["total_batches"] == 0