from collections import deque
from typing import List, Optional
import bisect

class OutputLengthPredictor:
    def __init__(self, buckets:List[int] = None, window:int=50, default:int=256, min_samples:int=5):
        if buckets is None:
            buckets = [64, 128, 256, 512, 1024, 2048]
        self.buckets = sorted(buckets)
        self.window = window
        self.default = default
        self.min_samples = min_samples

        self._bucket_data: List[deque] = [
            deque(maxlen=window) for _ in range(len(buckets))
        ]

        self._global_data:deque = deque(maxlen=window * len(buckets))


    def _bucket_idx(self, prompt_len:int) -> int:
        idx = bisect.bisect_left(self.buckets, prompt_len)
        return min(idx, len(self.buckets) - 1)
    
    @staticmethod
    def _avg(dq:deque) -> Optional[float]:
        if not dq:
            return None
        return sum(dq) / len(dq)
    

    def predict(self, prompt_len:int) -> int:
        idx = self._bucket_idx(prompt_len)
        bucket = self._bucket_data[idx]

        if len(bucket) >= self.min_samples:
            return max(1, round(self._avg(bucket)))
        
        global_avg = self._avg(self._global_data)
        if global_avg is not None:
            return max(1, round(global_avg))
        
        return self.default
    
    def update(self, prompt_len:int, actual_output_len:int):
        idx = self._bucket_idx(prompt_len)
        self._bucket_data[idx].append(actual_output_len)
        self._global_data.append(actual_output_len)

    def stats(self) -> dict:
        result = {}
        for i, upper in enumerate(self.buckets):
            dq = self._bucket_data[i]
            label = f"<={upper}"
            result[label] = {
                "n":len(dq),
                "avg":round(self._avg(dq), 1) if dq else None
            }
        return result


if __name__ == "__main__":
    pred = OutputLengthPredictor(default=256, min_samples=3)
 
    # cold start
    print(f"cold start predict(80)  = {pred.predict(80)}")   # -> 256
    print(f"cold start predict(500) = {pred.predict(500)}")  # -> 256
 
    # feed some data
    for actual in [120, 130, 110, 140, 125]:
        pred.update(prompt_len=80, actual_output_len=actual)
    for actual in [400, 420, 380]:
        pred.update(prompt_len=500, actual_output_len=actual)
 
    print(f"\nafter updates:")
    print(f"  predict(80)  = {pred.predict(80)}")    # -> ~125
    print(f"  predict(100) = {pred.predict(100)}")   # same bucket <=128, -> ~125
    print(f"  predict(500) = {pred.predict(500)}")   # bucket <=512, too few samples, fallback to global
    print(f"  predict(2000)= {pred.predict(2000)}")  # largest bucket, fallback to global
 
    print(f"\nstats: {pred.stats()}")