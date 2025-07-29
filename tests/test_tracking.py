import pytest
import time
import json
from pathlib import Path

from src.tracking.tracker import GenerationTracker, DistributedTracker
from src.tracking.metrics import MetricsCollector, TokenMetrics


class TestGenerationTracker:
    
    def test_tracker_initialization(self, temp_dir):
        """Test tracker initialization"""
        tracker = GenerationTracker(output_dir=temp_dir)
        
        assert tracker.output_dir == temp_dir
        assert tracker.completed_items == 0
        assert tracker.total_items == 0
        assert tracker.metrics == {}
    
    def test_tracker_start_stop(self, temp_dir):
        """Test starting and stopping tracker"""
        tracker = GenerationTracker(output_dir=temp_dir, save_interval=1)
        
        tracker.start(total_items=100)
        assert tracker.total_items == 100
        assert tracker.start_time is not None
        
        # Update progress
        tracker.update_progress(50)
        assert tracker.completed_items == 50
        
        # Stop tracker
        tracker.stop()
        assert tracker.end_time is not None
        assert tracker.end_time > tracker.start_time
    
    def test_tracker_save_load_state(self, temp_dir):
        """Test saving and loading tracker state"""
        tracker = GenerationTracker(output_dir=temp_dir)
        
        tracker.start(total_items=100)
        tracker.update_progress(50)
        tracker.update_metrics({"tokens_per_second": 150})
        
        # Save state
        tracker.save_state()
        
        # Load state
        state = tracker.load_state()
        assert state["completed_items"] == 50
        assert state["total_items"] == 100
        assert state["metrics"]["tokens_per_second"] == 150
        assert "timestamp" in state
    
    def test_tracker_paths(self, temp_dir):
        """Test tracker path generation"""
        tracker = GenerationTracker(output_dir=temp_dir)
        
        output_path = tracker.get_output_path()
        assert output_path.startswith(str(temp_dir))
        assert output_path.endswith(".parquet")
        
        metadata_path = tracker.get_metadata_path()
        assert metadata_path == str(temp_dir / "generation_metadata.json")
        
        checkpoint_path = tracker.get_checkpoint_path()
        assert checkpoint_path == str(temp_dir / "checkpoint.json")
    
    def test_save_final_metrics(self, temp_dir):
        """Test saving final metrics"""
        tracker = GenerationTracker(output_dir=temp_dir)
        
        metrics = {
            "total_tokens": 10000,
            "total_time": 100,
            "tokens_per_second": 100
        }
        tracker.update_metrics(metrics)
        tracker.save_final_metrics()
        
        metrics_file = temp_dir / "final_metrics.json"
        assert metrics_file.exists()
        
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
        assert saved_metrics == metrics


class TestDistributedTracker:
    
    def test_distributed_tracker_init(self, temp_dir):
        """Test distributed tracker initialization"""
        tracker = DistributedTracker(num_workers=4, output_dir=temp_dir)
        
        assert tracker.num_workers == 4
        assert len(tracker.worker_progress) == 4
        assert all(p == 0 for p in tracker.worker_progress)
        assert len(tracker.worker_metrics) == 4
    
    def test_update_worker_progress(self, temp_dir):
        """Test updating worker progress"""
        tracker = DistributedTracker(num_workers=4, output_dir=temp_dir)
        tracker.start(total_items=100)
        
        # Update individual workers
        tracker.update_worker_progress(0, 10)
        tracker.update_worker_progress(1, 15)
        tracker.update_worker_progress(2, 20)
        tracker.update_worker_progress(3, 5)
        
        assert tracker.worker_progress == [10, 15, 20, 5]
        assert tracker.completed_items == 50
    
    def test_update_worker_metrics(self, temp_dir):
        """Test updating worker metrics"""
        tracker = DistributedTracker(num_workers=2, output_dir=temp_dir)
        
        tracker.update_worker_metrics(0, {"tokens": 100, "time": 10})
        tracker.update_worker_metrics(1, {"tokens": 150, "time": 15})
        
        # Check aggregated metrics
        assert tracker.metrics["tokens"] == 250
        assert tracker.metrics["time"] == 25
    
    def test_worker_summary(self, temp_dir):
        """Test getting worker summary"""
        tracker = DistributedTracker(num_workers=3, output_dir=temp_dir)
        
        tracker.update_worker_progress(0, 10)
        tracker.update_worker_progress(1, 20)
        tracker.update_worker_progress(2, 15)
        
        tracker.update_worker_metrics(0, {"tokens": 100})
        tracker.update_worker_metrics(1, {"tokens": 200})
        tracker.update_worker_metrics(2, {"tokens": 150})
        
        summary = tracker.get_worker_summary()
        
        assert summary["num_workers"] == 3
        assert summary["worker_progress"] == [10, 20, 15]
        assert summary["total_progress"] == 45
        assert len(summary["worker_metrics"]) == 3


class TestMetricsCollector:
    
    def test_metrics_collector_basic(self):
        """Test basic metrics collection"""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record("tokens", 100)
        collector.record("tokens", 150)
        collector.record("tokens", 200)
        
        collector.record("latency", 0.1)
        collector.record("latency", 0.15)
        collector.record("latency", 0.2)
        
        summary = collector.get_summary()
        
        assert summary["tokens"]["count"] == 3
        assert summary["tokens"]["sum"] == 450
        assert summary["tokens"]["mean"] == 150
        assert summary["tokens"]["min"] == 100
        assert summary["tokens"]["max"] == 200
        
        assert summary["latency"]["count"] == 3
        assert summary["latency"]["mean"] == pytest.approx(0.15, rel=1e-3)
    
    def test_metrics_collector_timers(self):
        """Test timer functionality"""
        collector = MetricsCollector()
        
        collector.start_timer("generation")
        time.sleep(0.1)
        duration = collector.stop_timer("generation")
        
        assert duration >= 0.1
        assert "generation_time" in collector.metrics
        assert collector.metrics["generation_time"][0] >= 0.1
    
    def test_record_batch(self):
        """Test batch recording"""
        collector = MetricsCollector()
        
        metrics_dict = {
            "tokens": 100,
            "latency": 0.1,
            "temperature": 0.7
        }
        
        collector.record_batch(metrics_dict)
        
        assert len(collector.metrics["tokens"]) == 1
        assert len(collector.metrics["latency"]) == 1
        assert len(collector.metrics["temperature"]) == 1
    
    def test_throughput_metrics(self):
        """Test throughput calculation"""
        collector = MetricsCollector()
        
        # Record generation data
        for i in range(10):
            collector.record("tokens", 100)
            collector.record("generation_time", 0.1)
        
        throughput = collector.get_throughput_metrics()
        
        assert throughput["tokens_per_second"] == pytest.approx(1000, rel=1e-3)
        assert throughput["items_per_second"] == pytest.approx(10, rel=1e-3)
    
    def test_error_metrics(self):
        """Test error metrics"""
        collector = MetricsCollector()
        
        # Record some successful generations
        for i in range(8):
            collector.record("generation_time", 0.1)
        
        # Record some errors
        collector.record("errors", ValueError("Invalid input"))
        collector.record("errors", TimeoutError("Timeout"))
        
        error_metrics = collector.get_error_metrics()
        
        assert error_metrics["total_errors"] == 2
        assert error_metrics["error_rate"] == pytest.approx(0.2, rel=1e-3)
        assert "ValueError" in error_metrics["error_types"]
        assert "TimeoutError" in error_metrics["error_types"]
    
    def test_merge_collectors(self):
        """Test merging metrics from multiple collectors"""
        collector1 = MetricsCollector()
        collector1.record("tokens", 100)
        collector1.record("tokens", 150)
        
        collector2 = MetricsCollector()
        collector2.record("tokens", 200)
        collector2.record("tokens", 250)
        
        collector1.merge(collector2)
        
        assert len(collector1.metrics["tokens"]) == 4
        assert sum(collector1.metrics["tokens"]) == 700


class TestTokenMetrics:
    
    def test_token_metrics_recording(self):
        """Test recording token usage"""
        metrics = TokenMetrics()
        
        metrics.record_generation(input_tokens=50, output_tokens=100)
        metrics.record_generation(input_tokens=60, output_tokens=120)
        metrics.record_generation(input_tokens=70, output_tokens=140)
        
        summary = metrics.get_summary()
        
        assert summary["total_input_tokens"] == 180
        assert summary["total_output_tokens"] == 360
        assert summary["total_tokens"] == 540
        assert summary["avg_input_tokens"] == 60
        assert summary["avg_output_tokens"] == 120
        assert summary["max_total_tokens"] == 210
    
    def test_cost_estimation(self):
        """Test cost estimation based on token usage"""
        metrics = TokenMetrics()
        
        # Record some usage
        for _ in range(10):
            metrics.record_generation(input_tokens=1000, output_tokens=2000)
        
        costs = metrics.estimate_cost(
            input_price_per_1k=0.01,
            output_price_per_1k=0.03
        )
        
        assert costs["input_cost"] == pytest.approx(0.1, rel=1e-3)
        assert costs["output_cost"] == pytest.approx(0.6, rel=1e-3)
        assert costs["total_cost"] == pytest.approx(0.7, rel=1e-3)
        assert costs["cost_per_generation"] == pytest.approx(0.07, rel=1e-3)