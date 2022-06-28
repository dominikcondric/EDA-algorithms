# coding: UTF-8
import math
import itertools

class BICMetric():
    def __init__(self):
        pass
        
    def set_samples(self, samples):
        self.samples = samples
        self.prev_likeli_table = {}
        self.prev_cpt_size_table = {}

    def score(self, bn):
        self.bn = bn
        return self.get_likelihood() - \
            0.5 * self.get_CPT_size()  * math.log(len(self.samples), 2)

    def get_likelihood(self):
        res = 0.0
        for i in range(self.bn.num_of_var):
            if (i,tuple(self.bn.get_parents(i))) in self.prev_likeli_table:
                res += self.prev_likeli_table[(i,tuple(self.bn.get_parents(i)))]
                continue
            pi_size = len(list(self.bn.pi_table[i]))
            res_j = 0.0
            for j in range(pi_size):
                n_ij = self.count_samples(i, j, None)
                for k in range(len([0, 1])):
                    n_ijk = self.count_samples(i, j, k)
                    if n_ijk < 1:
                        continue
                    res_j += n_ijk * math.log(float(n_ijk)/float(n_ij), 2)
            res += res_j
            self.prev_likeli_table[(i,tuple(self.bn.get_parents(i)))] = res_j
        return res

    def count_samples(self, i, j, k):
        res = 0
        candidate = self.bn.pi_table.get(i)
        for s in self.samples:
            if k is not None and s.bitstring[i] != [0, 1][k]:
                continue
            flag = True
            for p,c in zip(self.bn.get_parents(i),candidate):
                if s.gene[p] != c:
                    flag = False
                    break
            if flag:
                res += 1
        return res
        
    def get_CPT_size(self):
        K = 0
        for i in range(self.bn.num_of_var):
            if (i,tuple(self.bn.get_parents(i))) in self.prev_cpt_size_table:
                K += self.prev_cpt_size_table[(i,tuple(self.bn.get_parents(i)))]
                continue
            r = len([0, 1]) 
            q = len(list(self.bn.pi_table[i]))
            K += q * (r-1)
            self.prev_cpt_size_table[(i,tuple(self.bn.get_parents(i)))] = q * (r-1)
        return K