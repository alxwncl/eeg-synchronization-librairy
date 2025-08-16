import time
import psutil
from threading import Lock



class RAMMonitor:
    """Monitor RAM usage and control task submission to prevent memory saturation."""
    
    def __init__(self, max_memory_percent=85, check_interval=0.5, verbose=False):
        """
        Initialize RAM monitor.
        
        Args:
            max_memory_percent (float): Maximum RAM usage percentage before blocking new tasks
            check_interval (float): How often to check RAM usage (seconds)
            verbose (bool): Whether to print RAM status messages
        """
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.verbose = verbose
        self.lock = Lock()
        self._last_check = 0
        self._last_memory_percent = 0
    
    def get_memory_usage(self):
        """Get current memory usage percentage."""
        current_time = time.time()
        with self.lock:
            # Cache memory check for performance (avoid checking too frequently)
            if current_time - self._last_check < self.check_interval:
                return self._last_memory_percent
            
            memory = psutil.virtual_memory()
            self._last_memory_percent = memory.percent
            self._last_check = current_time
            
            # if self.verbose and memory.percent > self.max_memory_percent * 0.9:
            #     print(f"RAM usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
            
            return self._last_memory_percent
    
    def can_submit_task(self):
        """Check if it's safe to submit a new task based on RAM usage."""
        return self.get_memory_usage() < self.max_memory_percent
    
    def wait_for_memory(self, timeout=300):
        """
        Wait for memory usage to drop below threshold.
        
        Args:
            timeout (float): Maximum time to wait in seconds
            
        Returns:
            bool: True if memory is available, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.can_submit_task():
                return True
            if self.verbose:
                memory_percent = self.get_memory_usage()
                print(f"Waiting for RAM to free up... Current usage: {memory_percent:.1f}%")
            time.sleep(self.check_interval)
        return False