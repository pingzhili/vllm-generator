import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import threading

logger = logging.getLogger(__name__)


class GenerationTracker:
    """Tracks generation progress and saves metadata"""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        save_interval: int = 60,
        enable_progress_bar: bool = True
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.enable_progress_bar = enable_progress_bar
        
        # Tracking data
        self.start_time = None
        self.end_time = None
        self.completed_items = 0
        self.total_items = 0
        self.metrics = {}
        
        # Progress bar
        self.pbar = None
        if enable_progress_bar:
            try:
                from tqdm import tqdm
                self.tqdm = tqdm
            except ImportError:
                logger.warning("tqdm not available, progress bar disabled")
                self.enable_progress_bar = False
        
        # Auto-save thread
        self.save_thread = None
        self.stop_save_thread = threading.Event()
        
    def start(self, total_items: int):
        """Start tracking"""
        self.start_time = time.time()
        self.total_items = total_items
        self.completed_items = 0
        
        # Create progress bar
        if self.enable_progress_bar:
            self.pbar = self.tqdm(
                total=total_items,
                desc="Generating",
                unit="items"
            )
        
        # Start auto-save thread
        self.save_thread = threading.Thread(target=self._auto_save)
        self.save_thread.start()
        
        logger.info(f"Started tracking {total_items} items")
    
    def update_progress(self, completed: int, total: Optional[int] = None):
        """Update progress"""
        self.completed_items = completed
        if total:
            self.total_items = total
        
        if self.pbar:
            self.pbar.n = completed
            self.pbar.refresh()
        
        # Calculate ETA
        if self.start_time and completed > 0:
            elapsed = time.time() - self.start_time
            rate = completed / elapsed
            remaining = self.total_items - completed
            eta = remaining / rate if rate > 0 else 0
            
            if self.pbar:
                self.pbar.set_postfix({
                    "rate": f"{rate:.1f} items/s",
                    "eta": f"{eta:.0f}s"
                })
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics"""
        self.metrics.update(metrics)
    
    def stop(self):
        """Stop tracking"""
        self.end_time = time.time()
        
        # Stop auto-save thread
        self.stop_save_thread.set()
        if self.save_thread:
            self.save_thread.join()
        
        # Close progress bar
        if self.pbar:
            self.pbar.close()
        
        # Save final state
        self.save_state()
        
        total_time = self.end_time - self.start_time
        logger.info(
            f"Completed {self.completed_items}/{self.total_items} items "
            f"in {total_time:.1f}s ({self.completed_items/total_time:.1f} items/s)"
        )
    
    def save_state(self):
        """Save current state to file"""
        state = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "completed_items": self.completed_items,
            "total_items": self.total_items,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        state_path = self.output_dir / "tracker_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load saved state"""
        state_path = self.output_dir / "tracker_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
        return None
    
    def _auto_save(self):
        """Auto-save thread function"""
        while not self.stop_save_thread.is_set():
            self.stop_save_thread.wait(self.save_interval)
            if not self.stop_save_thread.is_set():
                self.save_state()
    
    def get_output_path(self) -> str:
        """Get output file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(self.output_dir / f"output_{timestamp}.parquet")
    
    def get_metadata_path(self) -> str:
        """Get metadata file path"""
        return str(self.output_dir / "generation_metadata.json")
    
    def get_checkpoint_path(self) -> str:
        """Get checkpoint file path"""
        return str(self.output_dir / "checkpoint.json")
    
    def save_final_metrics(self):
        """Save final metrics"""
        if not self.metrics:
            return
        
        metrics_path = self.output_dir / "final_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


class DistributedTracker(GenerationTracker):
    """Tracker for distributed generation across multiple workers"""
    
    def __init__(self, num_workers: int, **kwargs):
        super().__init__(**kwargs)
        self.num_workers = num_workers
        self.worker_progress = [0] * num_workers
        self.worker_metrics = [{} for _ in range(num_workers)]
        
    def update_worker_progress(self, worker_id: int, completed: int):
        """Update progress for a specific worker"""
        self.worker_progress[worker_id] = completed
        total_completed = sum(self.worker_progress)
        self.update_progress(total_completed)
    
    def update_worker_metrics(self, worker_id: int, metrics: Dict[str, Any]):
        """Update metrics for a specific worker"""
        self.worker_metrics[worker_id] = metrics
        
        # Aggregate metrics
        aggregated = {}
        for key in metrics.keys():
            if isinstance(metrics[key], (int, float)):
                # Sum numeric metrics
                aggregated[key] = sum(
                    m.get(key, 0) for m in self.worker_metrics
                )
        
        self.update_metrics(aggregated)
    
    def get_worker_summary(self) -> Dict[str, Any]:
        """Get summary of worker performance"""
        summary = {
            "num_workers": self.num_workers,
            "worker_progress": self.worker_progress,
            "total_progress": sum(self.worker_progress),
            "worker_metrics": self.worker_metrics
        }
        return summary