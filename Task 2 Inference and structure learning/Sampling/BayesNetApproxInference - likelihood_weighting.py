import sys
import random
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader
import numpy as np
from sklearn import metrics
import math
import time


class BayesNetApproxInference(BayesNetReader):
    query = {}
    prob_dist = {}
    seeds = {}
    num_samples = None
    rand_vars = []
    rv_key_values = {}
    rv_all_values = []
    predictor_variable = None
    num_data_instances = 0
    default_missing_count = 0.000001
    probabilities = {}
    predictions = []
    training_time = None
    inference_time = None
    log_probabilities = False
    verbose = False

    def __init__(self, file_name, fitted_model=None):


        if fitted_model is None:
                super().__init__(file_name)
                self.read_test_data(r'C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\supporting docs\Task 2 Inference and structure learning\Sampling\stroke-data-discretized-train.csv')
                self.calculate_scoring_functions()

        else:
                self.read_test_data(file_name)
                self.num_samples = 1000
                self.rv_key_values = self.bn["rv_key_values"]
                #self.probabilities = fitted_model.probabilities
                #self.training_time = fitted_model.training_time
                self.test_learnt_probabilities(file_name)
                self.compute_performance()

    def read_test_data(self, data_file):
        #print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                    for variable in self.rand_vars:
                        self.rv_key_values[variable] = []
                else:
                    values = line.split(',')
                    self.rv_all_values.append(values)
                    self.update_variable_key_values(values)
                    self.num_data_instances += 1

        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10]))
        print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
        print("|data instances|=%d" % (self.num_data_instances))

    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)


    def likelihood_weighting(self):
        print("\nSTARTING likelihood weighting...")
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        C = {}

        # initialise vector of counts (True and False =0)
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0

        # loop to increase counts when the sampled vector consistent w/evidence
        for i in range(0, self.num_samples):
            wt=1
            X = self.prior_sample()
            for var, val in X.items():
                for a in self.query["evidence"]:
                    # if var == 'S':
                    #     print(val)
                    if a == var:
                        like = 1
                        break
                    else:
                        like = 0
                if like == 0:
                    w=1
                else:
                    dum=wt
                    wt = wt * self.evidence_sampling(var, val, X)
                    #print(dum,'*',wt)
            for var, val in X.items():
                if var == self.query['query_var']:
                    v = val
                    break
            C[v] += wt
            # if self.is_compatible_with_evidence(X, evidence):
            #     value_to_increase = X[query_variable]
            #     C[value_to_increase] += 1
            #print(self.query['query_var'],"=",v, '| C=true, W=true')
            # if (wt==0):
            #     print('STOP')
            # else:
            #print('%.10f' % wt)
        return bnu.normalise(C)

    def prior_sample(self):
        X = {}
        
        sampled_var_values = {}
        #wt = 0
        for variable in self.bn["random_variables"]:
            like_check = 0
            for a in self.query["evidence"]:
                if a == variable:
                    like_check = 1
                    X[variable] = self.query["evidence"][a]
            if like_check == 0:            
                X[variable] = self.get_sampled_value(variable, sampled_var_values)
            sampled_var_values[variable] = X[variable]

        return X

    def evidence_sampling(self, V, val, sampled):
        # get the conditional probability distribution (cpt) of variable V
        parents = bnu.get_parents(V, self.bn)
        cpt = {}
        prob_mass = 0

        # generate a cumulative distribution for random variable V
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                if value==val:
                    return probability
                # prob_mass += probability
                # cpt[value] = prob_mass

        else:
            for v in bnu.get_domain_values(V, self.bn):
                p = bnu.get_probability_given_parents(V, v, sampled, self.bn)
                if v==val:
                    return p
                # prob_mass += p
                # cpt[v] = prob_mass

        # # check that the cpt sums to 1 (or almost)
        # if prob_mass < 0.999 and prob_mass > 1:
        #     print("ERROR: CPT=%s does not sum to 1" % (cpt))
        #     exit(0)

        #return self.sampling_from_evi_cumulative_distribution(cpt)

    def get_sampled_value(self, V, sampled):
        # get the conditional probability distribution (cpt) of variable V
        parents = bnu.get_parents(V, self.bn)
        cpt = {}
        prob_mass = 0

        # generate a cumulative distribution for random variable V
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                prob_mass += probability
                cpt[value] = prob_mass

        else:
            for v in bnu.get_domain_values(V, self.bn):
                p = bnu.get_probability_given_parents(V, v, sampled, self.bn)
                prob_mass += p
                cpt[v] = prob_mass

        # check that the cpt sums to 1 (or almost)
        if prob_mass < 0.999 and prob_mass > 1:
            print("ERROR: CPT=%s does not sum to 1" % (cpt))
            exit(0)

        return self.sampling_from_cumulative_distribution(cpt)

    def test_learnt_probabilities(self, file_name):
        print("\nEVALUATING on "+str(file_name))
        self.inference_time = time.time()
        prob_query = ""
        
        # iterate over all instances in the test data
        for instance in self.rv_all_values:
                distribution = {}
                if self.verbose:
                    print("Input vector=%s" % (instance))
                
                i =0
                prob_query = "P(" + self.predictor_variable + "|"
                for rand_vars in self.rand_vars:
                    if (rand_vars!=self.rand_vars[len(self.rand_vars)-1]) and (i!=len(self.rand_vars)-2):
                        prob_query += rand_vars + '=' + instance[i] + ','
                    else:
                        prob_query += rand_vars + '=' + instance[i] + ')'
                        break
                    i +=1

                    #for predictor_value in self.rv_key_values[self.predictor_variable]:
                    #prob_query = "P(P|O=sunny, W=weak)"
                self.query = bnu.tokenise_query(prob_query)
                normalised_dist = self.likelihood_weighting()
                #normalised_dist = bnu.normalise(self.prob_dist)
                #self.training_time = time.time() - self.training_time
                #print("unnormalised probability_distribution="+str(self.prob_dist))
                #print("normalised probability_distribution="+str(normalised_dist))
                self.predictions.append(normalised_dist)

        self.inference_time = time.time() - self.inference_time

    #return true or false
    def sampling_from_cumulative_distribution(self, cumulative): 
        random_number = random.random()
        for value, probability in cumulative.items():
            if random_number <= probability:
                random_number = random.random()
                return value.split("|")[0]

        print("ERROR couldn't do sampling from:")
        print("cumulative_dist="+str(cumulative))
        exit(0)

    def sampling_from_evi_cumulative_distribution(self, cumulative): 
        random_number = random.random()
        for value, probability in cumulative.items():
            if random_number <= probability:
                random_number = random.random()
                return value.split("|")[0]

        print("ERROR couldn't do sampling from:")
        print("cumulative_dist="+str(cumulative))
        exit(0)

    def is_compatible_with_evidence(self, X, evidence):
        for variable, value in evidence.items():
            if X[variable] != value:
                return False
        return True

    def calculate_scoring_functions(self):
        print("\nCALCULATING LL and BIC on training data...")
        LL = self.calculate_log_likelihood()
        BIC = self.calculate_bayesian_information_criterion(LL)
        print("LL score="+str(LL))
        print("BIC score="+str(BIC))


    def compute_performance(self):

        Y_true = []
        Y_pred = []
        Y_prob = []

        # obtain vectors of categorical and probabilistic predictions
        for i in range(0, len(self.rv_all_values)):
            target_value = self.rv_all_values[i][len(self.rand_vars)-1]
            if target_value == 'yes': Y_true.append(1)
            elif target_value == 'no': Y_true.append(0)
            elif target_value == '1': Y_true.append(1)
            elif target_value == '0': Y_true.append(0)
            #target_value = float(target_value) #added for gaussian
            predicted_output = self.predictions[i][target_value]
            Y_prob.append(predicted_output)

            best_key = max(self.predictions[i], key=self.predictions[i].get)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == '1': Y_pred.append(1)
            elif best_key == '0': Y_pred.append(0)

        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        # calculate metrics: accuracy, auc, brief, kl, training/inference times
        acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))
        print("PERFORMANCE:")
        print("Balanced Accuracy="+str(acc))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))
        print("training Time="+str(self.training_time)+" secs.")
        print("Inference Time="+str(self.inference_time)+" secs.")

    def calculate_log_likelihood(self):

        LL = 0
        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]
        # iterate over all instances in the training data
        for instance in self.rv_all_values:
            predictor_value = instance[len(instance)-1]

            i =0
            #prob_query = "P(" + self.predictor_variable + "|"
            prob_query = "P(" + "Dummy" + "|"
            for rand_vars in self.rand_vars:
                if (rand_vars!=self.rand_vars[len(self.rand_vars)-1]) and (i!=len(self.rand_vars)-1):
                    prob_query += rand_vars + '=' + instance[i] + ','
                else:
                    prob_query += rand_vars + '=' + instance[i] + ')'
                    break
                i +=1

            self.query = bnu.tokenise_query(prob_query)
            evidence = self.query["evidence"].copy()
            # iterate over all random variables except the predictor var.
            for value_index in range(0, len(instance)):
                variable = self.rand_vars[value_index]
                value = instance[value_index]
                #prob_dist = self.probabilities[variable]
                prob_dist = bnu.get_probability_given_parents(variable, value, evidence, self.bn)
                #prob = prob_dist[value+"|"+predictor_value]
                LL += math.log(prob_dist)


            # accumulate the log prob of the predictor variable
            #prob_dist = self.probabilities[self.predictor_variable]
            #prob = self.bn["CPT("+ self.predictor_variable +")"][predictor_value]
            #LL += math.log(prob)
			
            if self.verbose == True:
                print("LL: %s -> %f" % (instance, log_probs))

        return LL

    def calculate_bayesian_information_criterion(self, LL):
        penalty = 0

        for variable in self.rand_vars:
                parents = bnu.get_parents(variable, self.bn)
                probability = 0
                if parents is None:
                    cpt = self.bn["CPT("+variable+")"]
                    num_params = len(cpt)
                else:
                    cpt = self.bn["CPT("+variable+"|"+parents+")"]
                    num_params = len(cpt)

                local_penalty = (math.log(self.num_data_instances)*num_params)/2
                #print("BIC: n=%s, p_i%s -> penalty=%f" % (variable, num_params, local_penalty))
                penalty += local_penalty

        BIC = LL - penalty
        return BIC


file_name_train = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\supporting docs\Task 2 Inference and structure learning\Sampling\config-stroke.txt"
file_name_test = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\supporting docs\Task 2 Inference and structure learning\Sampling\stroke-data-discretized-test_only_5_rows.csv"

nb_fitted = BayesNetApproxInference(file_name_train)
nb_tester = BayesNetApproxInference(file_name_test, nb_fitted)
