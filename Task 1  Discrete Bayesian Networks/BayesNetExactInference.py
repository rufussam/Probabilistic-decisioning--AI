import sys
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader
import time

class BayesNeExactInference(BayesNetReader):
    count=0
    query = {}
    prob_dist = {}


    def __init__(self):
            file_name = r"C:\Users\Rufus Sam A\Desktop\config-heart.txt"
            prob_query = "P(target|gender=0,cp=3)"
            #file_name = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\W3\BN-InfByEnumeration\config-stroke.txt"
            #prob_query = "P(stroke|gender=Male, age=2, smoking_status=formerly smoked)"
            super().__init__(file_name)
            self.query = bnu.tokenise_query(prob_query)
            self.inference_time = time.time()
            self.prob_dist = self.enumeration_ask()
            normalised_dist = bnu.normalise(self.prob_dist)
            self.inference_time = time.time() - self.inference_time
            print("unnormalised probability_distribution="+str(self.prob_dist))
            print("normalised probability_distribution="+str(normalised_dist))
            print("time="+str(self.inference_time))

    def enumeration_ask(self):
        print("\nSTARTING Inference by Enumeration...")
        Q = {}
        for value in self.bn["rv_key_values"][self.query["query_var"]]:
            value = value.split('|')[0]
            Q[value] = 0

        for value, probability in Q.items():
            value = value.split('|')[0]
            variables = self.bn["random_variables"].copy()
            evidence = self.query["evidence"].copy()
            evidence[self.query["query_var"]] = value
            probability = self.enumerate_all(variables, evidence)
            Q[value] = probability
            print("\tQ="+str(Q))

        return Q

    def enumerate_all(self, variables, evidence):
        self.count=self.count+1
        print('count---',self.count)
        #print("\nCALL to enumerate_all(): V=%s E=%s" % (variables, evidence))
        if len(variables) == 0:
            return 1.0

        V = variables[0]

        if V in evidence:
            v = evidence[V].split('|')[0]
            print("V",V)
            print("Evidence.Exact", evidence)
            p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
            variables.pop(0)
            return p*self.enumerate_all(variables, evidence)

        else:
            sum = 0
            evidence_copy = evidence.copy()
            for v in bnu.get_domain_values(V, self.bn):
                evidence[V] = v
                p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
                rest_variables = variables.copy()
                rest_variables.pop(0)
                #check = self.enumerate_all(rest_variables, evidence) #added new
                sum += p*self.enumerate_all(rest_variables, evidence)
                evidence = evidence_copy

            return sum


BayesNeExactInference()
