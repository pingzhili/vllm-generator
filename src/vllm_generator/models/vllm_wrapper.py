import logging
import time
from typing import Optional, List, Dict, Any, Iterator
from abc import ABC, abstractmethod
from vllm_generator.models import ModelConfig

logger = logging.getLogger(__name__)


class BaseVLLMModel(ABC):
    """Base class for vLLM model implementations"""
    
    @abstractmethod
    def generate(self, prompts: List[str], sampling_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate responses for a list of prompts"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompts: List[str], sampling_params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """Generate responses in streaming mode"""
        pass
    
    @abstractmethod
    def get_tokenizer(self):
        """Get the tokenizer from the model"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown the model"""
        pass


class VLLMModel(BaseVLLMModel):
    """Wrapper for vLLM model (real implementation)"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm = None
        self.sampling_params_class = None
        
        try:
            # Try to import vLLM
            from vllm import LLM, SamplingParams
            self.sampling_params_class = SamplingParams
            
            # Initialize vLLM
            logger.info(f"Initializing vLLM with model: {config.model}")
            self.llm = LLM(**config.to_vllm_args())
            logger.info("vLLM initialized successfully")
            
        except ImportError:
            logger.warning("vLLM not available, using mock model for testing")
            # Fall back to mock model
            self.__class__ = MockVLLMModel
            MockVLLMModel.__init__(self, config)
    
    def generate(self, prompts: List[str], sampling_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate responses for a list of prompts"""
        if sampling_params is None:
            sampling_params = self.config.to_sampling_params()
        
        # Create SamplingParams object
        params = self.sampling_params_class(**sampling_params)

        logger.info(f"Sampling params: {params}")
        
        # Generate
        start_time = time.time()
        outputs = self.llm.generate(prompts, params)
        latency = time.time() - start_time
        
        # Process outputs
        results = []
        for i, output in enumerate(outputs):
            result = {
                "prompt": prompts[i],
                "response": output.outputs[0].text,
                "tokens": len(output.outputs[0].token_ids),
                "latency": latency / len(outputs),  # Average latency
                "finish_reason": output.outputs[0].finish_reason
            }
            results.append(result)
        
        return results
    
    def generate_stream(self, prompts: List[str], sampling_params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """Generate responses in streaming mode"""
        if sampling_params is None:
            sampling_params = self.config.to_sampling_params()
        
        # Create SamplingParams object
        params = self.sampling_params_class(**sampling_params)

        logger.info(f"Sampling params: {params}")
        
        # Stream generation
        for output in self.llm.generate(prompts, params, use_tqdm=False):
            yield {
                "prompt_idx": output.prompt_token_ids,
                "text": output.outputs[0].text,
                "finished": output.finished
            }
    
    def get_tokenizer(self):
        """Get the tokenizer from the model"""
        if self.llm and hasattr(self.llm, 'get_tokenizer'):
            return self.llm.get_tokenizer()
        elif self.llm and hasattr(self.llm, 'tokenizer'):
            return self.llm.tokenizer
        else:
            logger.warning("Tokenizer not available from vLLM model")
            return None
    
    def shutdown(self):
        """Shutdown the model"""
        if self.llm:
            logger.info("Shutting down vLLM")
            # vLLM doesn't have explicit shutdown, but we can help with cleanup
            del self.llm
            self.llm = None


class MockVLLMModel(BaseVLLMModel):
    """Mock vLLM model for testing on systems without GPU"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        logger.info(f"Initializing mock vLLM model for: {config.model}")
        
        # Simulate model loading time
        time.sleep(0.5)
        
        self.response_templates = [
            "This is a mock response generated for testing purposes.",
            "Mock model output: The answer to your question would typically appear here.",
            "Test response: In a real scenario, the LLM would generate meaningful content.",
            "Sample output: This demonstrates the system functionality without actual generation.",
            "Mock generation: Real responses would be more contextually relevant."
        ]
    
    def generate(self, prompts: List[str], sampling_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate mock responses"""
        if sampling_params is None:
            sampling_params = self.config.to_sampling_params()
        
        results = []
        for i, prompt in enumerate(prompts):
            # Simulate generation time
            time.sleep(0.1)
            
            # Generate mock response
            response_idx = i % len(self.response_templates)
            response = self.response_templates[response_idx]
            
            # Add some variation based on temperature
            temp = sampling_params.get("temperature", 1.0)
            if temp > 1.0:
                response += f" (Temperature: {temp}, more creative output would appear here)"
            
            # Simulate token count
            tokens = len(response.split()) * 2  # Rough approximation
            
            result = {
                "prompt": prompt,
                "response": response,
                "tokens": tokens,
                "latency": 0.1,
                "finish_reason": "stop"
            }
            results.append(result)
        
        return results
    
    def generate_stream(self, prompts: List[str], sampling_params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """Generate mock responses in streaming mode"""
        for i, prompt in enumerate(prompts):
            response = self.response_templates[i % len(self.response_templates)]
            words = response.split()
            
            # Stream word by word
            for j, word in enumerate(words):
                time.sleep(0.01)  # Simulate streaming delay
                yield {
                    "prompt_idx": i,
                    "text": " ".join(words[:j+1]),
                    "finished": j == len(words) - 1
                }
    
    def get_tokenizer(self):
        """Get a mock tokenizer for testing"""
        # Return a mock tokenizer that has apply_chat_template method
        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                # Simple mock implementation
                formatted = []
                for msg in messages:
                    formatted.append(f"{msg['role']}: {msg['content']}")
                result = "\n".join(formatted)
                if add_generation_prompt:
                    result += "\nAssistant:"
                return result
        
        return MockTokenizer()
    
    def shutdown(self):
        """Shutdown mock model"""
        logger.info("Shutting down mock vLLM model")


class VLLMServer:
    """Wrapper for vLLM server mode (for multi-server parallelism)"""
    
    def __init__(self, config: ModelConfig, port: int = 8000):
        self.config = config
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start(self):
        """Start vLLM server"""
        import subprocess
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
        ]
        
        if self.config.dtype != "auto":
            cmd.extend(["--dtype", self.config.dtype])
        
        logger.info(f"Starting vLLM server on port {self.port}")
        self.process = subprocess.Popen(cmd)
        
        # Wait for server to be ready
        self._wait_for_server()
    
    def _wait_for_server(self, timeout: int = 60):
        """Wait for server to be ready"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info(f"vLLM server ready on port {self.port}")
                    return
            except:
                pass
            time.sleep(1)
        
        raise TimeoutError(f"vLLM server failed to start on port {self.port}")
    
    def generate(self, prompts: List[str], sampling_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate via API"""
        import requests
        
        if sampling_params is None:
            sampling_params = self.config.to_sampling_params()

        results = []
        for prompt in prompts:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "prompt": prompt,
                    **sampling_params
                }
            )
            response.raise_for_status()
            
            data = response.json()
            result = {
                "prompt": prompt,
                "response": data["choices"][0]["text"],
                "tokens": data["usage"]["completion_tokens"],
                "latency": data.get("latency", 0),
                "finish_reason": data["choices"][0]["finish_reason"]
            }
            results.append(result)
        
        return results
    
    def shutdown(self):
        """Shutdown server"""
        if self.process:
            logger.info(f"Shutting down vLLM server on port {self.port}")
            self.process.terminate()
            self.process.wait()
            self.process = None