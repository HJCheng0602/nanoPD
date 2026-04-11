import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from cost_model.analytical import AnalyticalCostModel
from router.output_lenth_predictor import OutputLengthPredictor

class Router:
    def __init__(self, cost_model: AnalyticalCostModel, predictor: OutputLengthPredictor = None):
        self.cost_model = cost_model
        self.predictor = predictor or OutputLengthPredictor()
        self._history: list = []   # stores (prompt_len, predicted_output_len, decision) tuples

    @classmethod
    def from_params(cls, params_path: str, **predictor_kwargs) -> "Router":
        cm = AnalyticalCostModel.load_params(params_path=params_path)
        pred = OutputLengthPredictor(**predictor_kwargs)
        return cls(cost_model=cm, predictor=pred)

    def route(self, prompt_len: int, system_load: int,
              decode_batch_size: int = 1) -> str:
        predicted_output_len = self.predictor.predict(prompt_len)
        decision, t_c, t_d = self.cost_model.route(
            prompt_len, predicted_output_len, system_load,
            decode_batch_size=decode_batch_size
        )
        self._history.append((prompt_len, predicted_output_len, decision))
        return decision

    def update(self, prompt_len: int, actual_output_len: int):
        self.predictor.update(prompt_len, actual_output_len)

    def decision_stats(self) -> dict:
        if not self._history:
            return {"total": 0, "collocated": 0, "disaggregated": 0}
        total = len(self._history)
        n_disagg = sum(1 for _, _, d in self._history if d == "disaggregated")
        return {
            "total":        total,
            "collocated":   total - n_disagg,
            "disaggregated": n_disagg,
            "disagg_ratio": n_disagg / total,
        }
    
if __name__ == "__main__":
    router = Router.from_params("cost_model/params.json")
 
    print("load=0:")
    for L in [64, 128, 256, 512, 1024, 2048]:
        d = router.route(L, system_load=0)
        print(f"  prompt_len={L:5d} -> {d}")
 
    print("\nload=4:")
    for L in [64, 128, 256, 512, 1024, 2048]:
        d = router.route(L, system_load=4)
        print(f"  prompt_len={L:5d} -> {d}")
 
    print(f"\nstats: {router.decision_stats()}")