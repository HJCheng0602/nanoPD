import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from engine.engine import Engine

class CollocatedWorker:
    def __init__(self, model_path:str, gpu_id:int=0, block_size:int=16, max_blocks:int=512):
        self.engine = Engine(model_path=model_path, block_size=block_size, max_blocks=max_blocks, device=f"cuda:{gpu_id}")
        self.finished = self.engine.scheduler.finished
    
    def add_request(self, prompt:str):
        return self.engine.add_request(prompt)
    def step(self):
        return self.engine.step()
    def run_until_done(self, max_tokens_per_seq:int=500) -> dict:
        return self.engine.run_until_done(max_tokens_per_seq)
    
    def run_until_done_single(self, prompt: str, max_new_tokens: int = 500) -> str:
        return self.engine.generate(prompt, max_new_tokens=max_new_tokens)
