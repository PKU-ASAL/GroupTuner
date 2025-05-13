from .lib.core import SRTunerModule
from .common import Tuner

# SRTuner as a standalone tuner
class SRTuner(Tuner):
    def __init__(self, search_space, evaluator,name):
        super().__init__(search_space, evaluator, name)

        # User can customize reward func as Python function and pass to module.
        # In this demo, we use the default reward func. 
        self.mod = SRTunerModule(search_space, default_perf = self.default_perf)

    def generate_candidates(self, batch_size=1):
        return self.mod.generate_candidates(batch_size)

    def evaluate_candidates(self, candidates):
        return self.evaluator.evaluate(candidates)
    def reflect_feedback(self, perfs):
        self.mod.reflect_feedback(perfs)