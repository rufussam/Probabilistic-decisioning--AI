import sys
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader
import time
import numpy as np
from sklearn import metrics
import math


class BayesNeExactInference(BayesNetReader):
    count=0
    query = {}
    prob_dist = {}

    #added
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
                self.rv_key_values = self.bn["rv_key_values"]
                #self.probabilities = fitted_model.probabilities
                #self.training_time = fitted_model.training_time
                self.test_learnt_probabilities(file_name)
                self.compute_performance()
                
    
    def read_test_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
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
                self.prob_dist = self.enumeration_ask()
                normalised_dist = bnu.normalise(self.prob_dist)
                #self.training_time = time.time() - self.training_time
                #print("unnormalised probability_distribution="+str(self.prob_dist))
                #print("normalised probability_distribution="+str(normalised_dist))
                self.predictions.append(normalised_dist)

        self.inference_time = time.time() - self.inference_time
            


    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)

    def enumeration_ask(self):
        print("\nSTARTING Inference by Enumeration...")
        Q = {}
        try: #----------->added for Gaussian
            for value in self.bn["rv_key_values"][self.query["query_var"]]:
                value = value.split('|')[0]
                Q[value] = 0
        except Exception: #----------->added for Gaussian
            Q = {0.0: 0, 1.0: 0}

        for value, probability in Q.items():
            #value = value.split('|')[0]  #remove hash for discrete #----------->added for Gaussian
            variables = self.bn["random_variables"].copy()
            evidence = self.query["evidence"].copy()
            evidence[self.query["query_var"]] = value
            probability = self.enumerate_all(variables, evidence)
            Q[value] = probability
            #print("\tQ="+str(Q))

        return Q

    def enumerate_all(self, variables, evidence):
        self.count=self.count+1
        #print('count---',self.count)
        #print("\nCALL to enumerate_all(): V=%s E=%s" % (variables, evidence))
        if len(variables) == 0:
            return 1.0

        V = variables[0]

        if V in evidence:
            #v = evidence[V].split('|')[0] #----------->removed for Gaussian
            v = str(evidence[V]).split('|')[0] #----------->added for Gaussian
            #print("V",V)
            #print("Evidence.Exact", evidence)
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
                sum += p*self.enumerate_all(rest_variables, evidence)
                evidence = evidence_copy

            return sum
    
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
            target_value = float(target_value) #added for gaussian
            predicted_output = self.predictions[i][target_value]
            Y_prob.append(predicted_output)

            best_key = max(self.predictions[i], key=self.predictions[i].get)
            best_key = round(best_key)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == 1.0 : Y_pred.append(1)
            elif best_key == 0.0 : Y_pred.append(0)

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
                    cpt = self.bn["PDF("+variable+")"]
                    num_params = len(cpt)
                else:
                    cpt = self.bn["PDF("+variable+"|"+parents+")"]
                    num_params = len(cpt)

                local_penalty = (math.log(self.num_data_instances)*num_params)/2
                #print("BIC: n=%s, p_i%s -> penalty=%f" % (variable, num_params, local_penalty))
                penalty += local_penalty

        BIC = LL - penalty
        return BIC


    def get_counts(self, variable_index):
        counts = {}
        predictor_index = len(self.rand_vars)-1

        # accumulate countings
        for values in self.rv_all_values:
            if variable_index is None:
                # case: prior probability
                value = values[predictor_index]
            else:
                # case: conditional probability
                value = values[variable_index]+"|"+values[predictor_index]

            try:
                counts[value] += 1
            except Exception:
                counts[value] = 1


# file_name_train = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\W7\PDF_Generator\config-diabetes-PDF.txt"
# file_name_test = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\W7\PDF_Generator\diabetes-original-test.csv"


# file_name_train = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\Assesssment\prob test\config-heart.txt"
# file_name_test = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\Assesssment\heart-stroke-data\heart-data-discretized-test.csv"


# file_name_train = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\Assesssment\Task 3\config-heart.txt"
# file_name_test = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\Assesssment\Task 3\heart-data-original-test.csv"

# file_name_train = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\W3\BN-InfByEnumeration\config-playtennis.txt"
# file_name_test = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\W3\BN-InfByEnumeration\play_tennis-test.csv"

file_name_train = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\Assesssment\Gaussian prob\config-stroke.txt"
file_name_test = r"C:\Users\Rufus Sam A\Downloads\Lincoln\AAI\Assesssment\Gaussian prob\stroke-data-original-test-conv2num.csv"

#BayesNeExactInference()

nb_fitted = BayesNeExactInference(file_name_train)
nb_tester = BayesNeExactInference(file_name_test, nb_fitted)
