import random
import copy

class Setup:
    def __init__(self, gene_size, gene_type="binary", sl_method="roulette",\
                     gene_length=50, pop_size=5, elite_rate=0.01, t_size=5):
        self.gene_type = gene_type
        self.sl_method=sl_method
        self.gene_length = gene_length
        self.pop_size = pop_size
        if pop_size < 100:
            self.elite_rate=1.0 / pop_size
        else:
            self.elite_rate=elite_rate
        self.t_size = t_size
        self.gene_size = gene_size
        self.current_max = -100000
        self.unchanged = 0
        self.optimum = gene_length

    def get_cardinality(self, index):
        return [0,1]

    def get_elites(self,population):
        res = []
        for i in range(int (self.pop_size*self.elite_rate)):
            res.append(copy.deepcopy(population[i]))
        return res
    
class SetupEda(Setup):
    def __init__(self, gene_type="binary", sl_method="roulette", \
                     gene_length=20, pop_size=100, elite_rate=0.01, \
                     t_size=5, alpha=1, sample_rate=0.1):
        Setup.__init__(self,int(sample_rate*pop_size), gene_type, sl_method, gene_length, pop_size, elite_rate, t_size)
        self.alpha = alpha
        self.sample_rate = sample_rate