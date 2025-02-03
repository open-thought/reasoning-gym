from z3 import *
import collections
import solver_utils
import numpy as np

class my_solver:
    def __init__(self, id, feature_name, feature_value, numbered_cnf, num2var):
        self.id = id
        self.num2var = num2var
        self.clue_num = 0

        self.feature_name = feature_name
        self.feature_value = feature_value
        self.numbered_cnf = numbered_cnf

        self.init_solver(feature_name, feature_value, numbered_cnf)

    def init_solver(self, feature_name, feature_value, numbered_cnf):
        set_param(proof=True)
        self.s = Solver()

        self.s.set(unsat_core=True)
        house_total = len(feature_value)
        self.house_total = house_total
        features, mapping = self.define_feature(feature_name, feature_value)

        self.houses = [
            [solver_utils.instanciate_int_constrained('%s%d' % (prop, n), self.s, house_total) for prop in
             features.keys()] for n in
            range(house_total)]
        self.mapping = mapping
        self.feature = features
        # because we read feature from GT, we can make this assumption to calculate difficulty.
        self.ground_truth = self.feature
        for single_cnf in numbered_cnf:
            if len(single_cnf) == 1:
                self.add_single_clue(single_cnf)
            else:
                self.add_mul_val_clue(single_cnf)
            self.clue_num += 1

    def define_feature(self, feature_name, feature_value):
        features = collections.OrderedDict()
        for i in range(len(feature_name)):
            if feature_name[i] != 'House':
                features[feature_name[i]]=[]
                for house_j in feature_value:
                    features[feature_name[i]].append(house_j[i])

        feature_mapping = {list(features.keys())[i]: i for i in range(len(features.keys()))}
        return features, feature_mapping

    def decode_var(self, var_num):
        attribute, house_num = (self.num2var[var_num].split(' '))
        attribute_name, attribute_value = (attribute.split('.'))
        if '_' in attribute_value:
            attribute_value = attribute_value.replace('_', ' ')
        house_num = int(house_num) - 1
        return house_num, attribute_name, attribute_value


    def normal_vs_not_constraints(self, house_num, attribute_name, attribute_value):
        if attribute_name.startswith('~'):
            attribute_name = attribute_name[1:]
            return Not(self.houses[house_num][self.mapping[attribute_name]] == self.feature[attribute_name].index(attribute_value))
        else:
            return self.houses[house_num][self.mapping[attribute_name]] == self.feature[attribute_name].index(attribute_value)

    def add_single_clue(self, single_cnf):
        house_num, attribute_name, attribute_value = self.decode_var(single_cnf[0])
        self.s.assert_and_track(self.normal_vs_not_constraints(house_num, attribute_name, attribute_value), 'a'+str(self.clue_num))

    def add_mul_val_clue(self, single_cnf):
        cons = []
        for i in range(len(single_cnf)):
            house_num, attribute_name, attribute_value = self.decode_var(single_cnf[i])
            cons.append(self.normal_vs_not_constraints(house_num, attribute_name, attribute_value))
        self.s.assert_and_track(Or(*cons), 'a' + str(self.clue_num))


    def check_solution(self):
        if self.s.check() == unsat:
            c = self.s.unsat_core()
            print("Size of the unsat core:", len(c))
            print("Unsat core:", ", ".join([str(i) for i in c]))

            proof_str = str(self.s.proof())
            print(self.s.proof())
            print("Proof length:", len(proof_str))

        else:
            m = self.s.model()
            solution = [[m[case].as_long() for case in line] for line in self.houses]
            unique=True
            if count_solutions(self.s)!=1:
                print(self.id)
                unique=False
            # print("Number of solutions:", count_solutions(self.s))
            return solution, unique

    def print_solution(self, solution):
        print("Solution:")
        print(self.feature)

    def check_cell_difficulty(self):
        results = []
        clue_num=0
        for i in range(self.house_total):
            for feature in self.ground_truth:
                self.s.assert_and_track(Not(self.houses[i][self.mapping[feature]] == i), "check"+str(clue_num))
                if self.s.check() == unsat:
                    results.append("The {} in house {} is {}".format(feature, i+1, self.ground_truth[feature][i]))
                self.s.reset()
                self.init_solver(self.feature_name, self.feature_value, self.numbered_cnf)
        return results

    def check_statement_difficulty(self):
        clue_num=0
        proof_length = np.zeros((self.house_total, len(self.ground_truth)))
        for i in range(self.house_total):
            for feature in self.ground_truth:
                self.s.assert_and_track(Not(self.houses[i][self.mapping[feature]] == i), "check"+str(clue_num))
                clue_num+=1
                if self.s.check() == unsat:
                    c = self.s.unsat_core()
                proof_length[i, self.mapping[feature]] = self.s.statistics().propagations
        print(proof_length)
        return proof_length

    def check_problem_difficulty(self):
        proof_length = np.zeros((1, 1))
        proof_length[0, 0] = self.s.statistics().propagations
        return proof_length




