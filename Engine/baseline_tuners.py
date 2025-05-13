from .common import Tuner
import random
# from deap import base,creator,tools

class RandomTuner(Tuner):
    def __init__(self, search_space, evaluator,name):
        super().__init__(search_space, evaluator,name)
    def generate_candidates(self, batch_size=1):
        candidates=[]

        for _ in range(batch_size):
            opt_setting = dict()

            for op in self.search_space.space:
                rv = random.randint(0, len(self.search_space.space[op])-1)
                opt_setting[op] = self.search_space.space[op][rv]
            self.search_space.setting=opt_setting

            candidates.append(self.search_space.convert_to_str())
        return candidates[0]
    
    def evaluate_candidates(self, candidates):
        return self.evaluator.evaluate(candidates)

    def reflect_feedback(self,perfs):
        # Random search. Do nothing
        pass






