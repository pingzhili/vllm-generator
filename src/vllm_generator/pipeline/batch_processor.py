import logging
from typing import List, Dict, Any, Optional, Iterator
from queue import Queue, Empty
from threading import Thread, Event
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles efficient batch processing with prefetching"""
    
    def __init__(
        self,
        batch_size: int = 32,
        prefetch_batches: int = 2,
        max_queue_size: int = 10
    ):
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.max_queue_size = max_queue_size
        
        # Queues for prefetching
        self.input_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        
        # Control
        self.stop_event = Event()
        self.prefetch_thread = None
        
    def process_items(
        self,
        items: List[Dict[str, Any]],
        process_fn: callable,
        num_workers: int = 1
    ) -> Iterator[Dict[str, Any]]:
        """Process items in batches with prefetching"""
        # Start prefetch thread
        self.prefetch_thread = Thread(
            target=self._prefetch_worker,
            args=(items,)
        )
        self.prefetch_thread.start()
        
        # Start processing workers
        workers = []
        for i in range(num_workers):
            worker = Thread(
                target=self._process_worker,
                args=(process_fn,)
            )
            worker.start()
            workers.append(worker)
        
        # Yield results as they become available
        total_items = len(items)
        processed = 0
        
        while processed < total_items:
            try:
                result_batch = self.output_queue.get(timeout=1)
                for result in result_batch:
                    yield result
                    processed += 1
            except Empty:
                # Check if workers are still alive
                if not any(w.is_alive() for w in workers):
                    break
        
        # Cleanup
        self.stop_event.set()
        self.prefetch_thread.join()
        for worker in workers:
            worker.join()
    
    def _prefetch_worker(self, items: List[Dict[str, Any]]):
        """Worker thread for prefetching batches"""
        num_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            if self.stop_event.is_set():
                break
            
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch = items[start_idx:end_idx]
            
            # Put batch in queue
            self.input_queue.put((batch_idx, batch))
            
            logger.debug(f"Prefetched batch {batch_idx} with {len(batch)} items")
        
        # Signal end of input
        for _ in range(self.max_queue_size):
            self.input_queue.put(None)
    
    def _process_worker(self, process_fn: callable):
        """Worker thread for processing batches"""
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=1)
                if item is None:
                    break
                
                batch_idx, batch = item
                
                # Process batch
                start_time = time.time()
                results = process_fn(batch)
                process_time = time.time() - start_time
                
                logger.debug(
                    f"Processed batch {batch_idx} with {len(batch)} items "
                    f"in {process_time:.2f}s"
                )
                
                # Put results in output queue
                self.output_queue.put(results)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Put empty results to maintain order
                self.output_queue.put([])


class SmartBatchProcessor(BatchProcessor):
    """Smart batch processor with dynamic batch sizing"""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        target_latency: float = 1.0,
        **kwargs
    ):
        super().__init__(batch_size=initial_batch_size, **kwargs)
        self.target_latency = target_latency
        self.latency_history = []
        self.batch_size_history = []
        
    def adjust_batch_size(self, latency: float):
        """Dynamically adjust batch size based on latency"""
        self.latency_history.append(latency)
        self.batch_size_history.append(self.batch_size)
        
        # Keep only recent history
        if len(self.latency_history) > 10:
            self.latency_history = self.latency_history[-10:]
            self.batch_size_history = self.batch_size_history[-10:]
        
        # Calculate average latency
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        
        # Adjust batch size
        if avg_latency > self.target_latency * 1.2:
            # Decrease batch size
            self.batch_size = max(1, int(self.batch_size * 0.8))
            logger.info(f"Decreased batch size to {self.batch_size} (latency: {avg_latency:.2f}s)")
        elif avg_latency < self.target_latency * 0.8:
            # Increase batch size
            self.batch_size = int(self.batch_size * 1.2)
            logger.info(f"Increased batch size to {self.batch_size} (latency: {avg_latency:.2f}s)")


class DistributedBatchProcessor:
    """Batch processor for distributed generation across multiple workers"""
    
    def __init__(
        self,
        num_workers: int,
        batch_size: int = 32,
        strategy: str = "round_robin"
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.strategy = strategy
        
        # Work queues for each worker
        self.work_queues = [Queue() for _ in range(num_workers)]
        self.result_queues = [Queue() for _ in range(num_workers)]
        
    def distribute_work(
        self,
        items: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Distribute items across workers"""
        worker_items = [[] for _ in range(self.num_workers)]
        
        if self.strategy == "round_robin":
            for i, item in enumerate(items):
                worker_idx = i % self.num_workers
                worker_items[worker_idx].append(item)
        
        elif self.strategy == "balanced":
            # Balance by estimated cost
            costs = [0] * self.num_workers
            
            for item in items:
                # Estimate cost (e.g., by prompt length)
                cost = len(item.get("prompt", ""))
                
                # Assign to worker with lowest cost
                min_worker = min(range(self.num_workers), key=lambda i: costs[i])
                worker_items[min_worker].append(item)
                costs[min_worker] += cost
        
        else:
            raise ValueError(f"Unknown distribution strategy: {self.strategy}")
        
        return worker_items
    
    def create_batches(
        self,
        items: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Create batches from items"""
        batches = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batches.append(batch)
        
        return batches