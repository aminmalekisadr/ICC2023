# from src.model.model import *
# from src.evaluation.evaluation import *
import copy
import logging
import math
import random
import sys
import time

import numpy as np
import yaml

from src.data_handeling.Preprocess import Preprocess
from src.ensemble.ensemble import ensemble_stacking
from src.evaluation.evaluation import evaluate_rnn
from src.model.model import create_neural_network
# from src.ensemble.ensemble import *
from src.model.model import generate_rnn


def mutate_ensemble(model):
    """
    Mutates the ensemble model
    :param model:
    :return:
    """
    # Mutate randomly the estimator
    bit_random = random.uniform(0, 1)

    if bit_random <= mutation_rate:
        # calculate mutation momentum
        mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
        # Mutate estimators
        model.n_estimators = model.n_estimators + math.ceil(model.n_estimators * mutation_momentum)

    return model


def get_stacking(train_x, train_y, test_x, test_y, model1, model2, model3):
    # First we create a list called "level0", which consists of our base models"
    # Basically, you'll want to pick a assortment of your favorite machine learning models
    # These models will get passed down to the meta-learner later
    import tensorflow as tf
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import StackingRegressor
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
    my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
    '''meta learner'''
    level0 = list()
    dim = experiment_config['data_parameters']['look_back']
    meta_learner = KerasRegressor(
        create_neural_network,  # Pass in function
        input_shape=dim,  # Pass in the dimensions to above function
        epochs=6,
        batch_size=32,
        verbose=False)

    meta_learner._estimator_type = "regressor"
    # meta_learner = KerasRegressor(build_fn= model1 , epochs=10, batch_size=32, verbose=0)
    level0.append(('rnn', meta_learner))
    model1._estimator_type = 'regressor'
    model2._estimator_type = 'regressor'
    model3._estimator_type = 'regressor'
    meta_learner._estimator_type = 'regressor'

    level0.append(('linear', LinearRegression()))
    # Create 5 neural networks using our function above
    #    pdb.set_trace()
    #    model1=KerasRegressor(build_fn=model1, epochs=100, batch_size=32, verbose=2)

    #   level0.append(('rnn', model1))

    # Wrap our neural network in a Keras Regressor to make it
    # compatible with StackingRegressor

    # Append to our list
    # level0.append(('nn_{num}'.format(num=1), keras_reg))
    # The "meta-learner" designated as the level1 model
    # In my experience Linear Regression performs best
    # but feel free to experiment with other models
    level1 = LinearRegression()
    # Create the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=2, verbose=1)
    # pdb.set_trace()
    # tf.saved_model.LoadOptions(
    #   allow_partial_checkpoint=False,
    #  experimental_io_device= None,
    # experimental_skip_checkpoint=False
    # )

    # pdb.set_trace()
    #    model.fit(train_x, train_y)
    # Make predictions
    #   pred = model.predict(test_x)
    # Calculate the RMSE

    return model


with open(sys.argv[1], "r") as yaml_config_file:
    logging.info("Loading simulation settings from %s", sys.argv[1])
    experiment_config = yaml.safe_load(yaml_config_file)
    # Load the data
    config_path = experiment_config["experiment_settings"]["config_path"]
collect_g = Preprocess(config_path)
collect_gc = collect_g.collect_gc()
mutation_rate = experiment_config["ga_parameters"]["mutation_rate"]
min_mutation_momentum = experiment_config["ga_parameters"]["min_mutation_momentum"]
max_mutation_momentum = experiment_config["ga_parameters"]["min_mutation_momentum"]
min_population = experiment_config["ga_parameters"]["min_population"]
max_population = experiment_config["ga_parameters"]["max_population"]
num_Iterations = experiment_config["ga_parameters"]["num_Iterations"]
optimisers = experiment_config["ml_parameters"]["optimizer_type"]


def crossover_rnn(model_1, model_2):
    """
    Executes crossover for the RNN in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # pdb.set_trace()

    try:

        new_model = copy.copy(model_1)
        # new_model = model_1
        # pdb.set_trace()

        # Probabilty of models depending on their RMSE test score
        # Lower RMSE score has higher prob

        test_score_total = model_1.test_score + model_2.test_score
        model_1_prob = 1 - (model_1.test_score / test_score_total)
        model_2_prob = 1 - model_1_prob
        # Probabilities of each item for each model (all items have same probabilities)
        model_1_prob_item = model_1_prob / (len(model_1.layers) - 2)
        model_2_prob_item = model_2_prob / (len(model_2.layers) - 2)

        # Number of layers of new generation depend on probability of each model
        num_layers_new_gen = int(model_1_prob * (len(model_1.layers) - 1) + model_2_prob * (len(model_2.layers) - 1))

        # Create list of int with positions of the layers of both models.
        cross_layers_pos = []
        # Create list of weights
        weights = []
        # Add positions of layers for model 1. Input and ouput layer are not added.
        for i in range(2, len(model_1.layers)):
            mod_item = type('', (), {})()
            mod_item.pos = i
            mod_item.model = 1
            cross_layers_pos.append(mod_item)
            weights.append(model_1_prob_item)

        # Add positions of layers for model 2. Input and ouput layer are not added.
        for i in range(2, len(model_2.layers)):
            mod_item = type('', (), {})()
            mod_item.pos = i
            mod_item.model = 2
            cross_layers_pos.append(mod_item)
            weights.append(model_2_prob_item)

        collect_gc

        # If new num of layers are larger than the num crossover layers, keep num of crossover layers
        if num_layers_new_gen > len(cross_layers_pos):
            num_layers_new_gen = len(cross_layers_pos)

        # Randomly choose num_layers_new_gen layers of the new list
        cross_layers_pos = list(np.random.choice(cross_layers_pos, size=num_layers_new_gen, replace=False, p=weights))

        # Add both group of hidden layers to new group of layers using previously chosen layer positions of models
        cross_layers = []
        for i in range(len(cross_layers_pos)):
            mod_item = cross_layers_pos[i]
            if mod_item.model == 1:
                cross_layers.append(model_1.layers[mod_item.pos])
            else:
                cross_layers.append(model_2.layers[mod_item.pos])

        collect_gc

        # Add input layer randomly from parent 1 or parent 2
        bit_random = random.randint(0, 1)
        if bit_random == 0:
            cross_layers.insert(0, model_1.layers[0])
        else:
            cross_layers.insert(0, model_2.layers[0])

        bit_random = random.randint(0, 1)
        if bit_random == 0:
            cross_layers.append(model_1.layers[len(model_1.layers) - 1])
        else:
            cross_layers.append(model_2.layers[len(model_2.layers) - 1])

        # Set new layers
        new_model._layers = cross_layers
    except:
        new_model = model_1

    return new_model


def crossover_lr(model_1, model_2):
    """
      Executes crossover for the RF in the GA for 2 models, modifying the first model
      :param model_1:
      :param model_2:
      :return:
      """
    # pdb.set_trace()
    try:
        new_model = copy.copy(model_1)
        # new_model = model_1
        # pdb.set_trace()

        # Probabilty of models depending on their RMSE test score
        # Lower RMSE score has higher prob

        test_score_total = model_1.test_score + model_2.test_score
        model_1_prob = 1 - (model_1.test_score / test_score_total)
        model_2_prob = 1 - model_1_prob
        # Probabilities of each item for each model (all items have same probabilities)
        model_1_prob_item = model_1_prob / (len(model_1.layers) - 2)
        model_2_prob_item = model_2_prob / (len(model_2.layers) - 2)

        # Number of layers of new generation depend on probability of each model
        num_layers_new_gen = int(model_1_prob * (len(model_1.layers) - 1) + model_2_prob * (len(model_2.layers) - 1))

        # Create list of int with positions of the layers of both models.
        cross_layers_pos = []
        # Create list of weights
        weights = []
        # Add positions of layers for model 1. Input and ouput layer are not added.
        for i in range(2, len(model_1.layers)):
            mod_item = type('', (), {})()
            mod_item.pos = i
            mod_item.model = 1
            cross_layers_pos.append(mod_item)
            weights.append(model_1_prob_item)

        # Add positions of layers for model 2. Input and ouput layer are not added.
        for i in range(2, len(model_2.layers)):
            mod_item = type('', (), {})()
            mod_item.pos = i
            mod_item.model = 2
            cross_layers_pos.append(mod_item)
            weights.append(model_2_prob_item)

        collect_gc

        # If new num of layers are larger than the num crossover layers, keep num of crossover layers
        if num_layers_new_gen > len(cross_layers_pos):
            num_layers_new_gen = len(cross_layers_pos)

        # Randomly choose num_layers_new_gen layers of the new list
        cross_layers_pos = list(np.random.choice(cross_layers_pos, size=num_layers_new_gen, replace=False, p=weights))

        # Add both group of hidden layers to new group of layers using previously chosen layer positions of models
        cross_layers = []
        for i in range(len(cross_layers_pos)):
            mod_item = cross_layers_pos[i]
            if mod_item.model == 1:
                cross_layers.append(model_1.layers[mod_item.pos])
            else:
                cross_layers.append(model_2.layers[mod_item.pos])

        collect_gc

        # Add input layer randomly from parent 1 or parent 2
        bit_random = random.randint(0, 1)
        if bit_random == 0:
            cross_layers.insert(0, model_1.layers[0])
        else:
            cross_layers.insert(0, model_2.layers[0])

        bit_random = random.randint(0, 1)
        if bit_random == 0:
            cross_layers.append(model_1.layers[len(model_1.layers) - 1])
        else:
            cross_layers.append(model_2.layers[len(model_2.layers) - 1])

        # Set new layers
        new_model._layers = cross_layers
    except:
        new_model = model_1

    return new_model


def mutate_rnn(model):
    """
    Mutates the RNN model
    :param model:
    :return:
    """
    for i in range(len(model.layers)):
        # Mutate randomly each layer
        bit_random = random.uniform(0, 1)

        if bit_random <= mutation_rate:
            weights = model.layers[i].get_weights()  # list of weights as numpy arrays
            # calculate mutation momentum
            mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
            new_weights = [x * mutation_momentum for x in weights]
            model.layers[i].set_weights(new_weights)

    collect_gc


def mutate_lr(model):
    """
    Mutates the Linear Regression model
    :param model:
    :return:
    """
    # Mutate randomly the estimator
    bit_random = random.uniform(0, 1)

    if bit_random <= mutation_rate:
        # calculate mutation momentum
        mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
        # Mutate estimators
        model.n_estimators = model.n_estimators + math.ceil(model.n_estimators * mutation_momentum)

    return model


def crossover_rf(model_1, model_2):
    """
    Executes crossover for the RF in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # pdb.set_trace()
    try:
        # new_model = copy.copy(model_1)
        new_model = model_1

        # Probabilty of models depending on their RMSE test score
        test_score_total = model_1.test_score + model_2.test_score
        model_1_prob = 1 - model_1.test_score / test_score_total
        model_2_prob = 1 - model_1_prob

        # New estimator is the sum of both estimators times their probability
        new_model.n_estimators = math.ceil(model_1.n_estimators * model_1_prob + model_2.n_estimators * model_2_prob)
    except  Exception as e:
        print('line 163 evaluate_ga', e)
        new_model = model_1

    return new_model


def mutate_rf(model):
    """
    Mutates the Random Forest
    :param model:
    :return:
    """
    # Mutate randomly the estimator
    bit_random = random.uniform(0, 1)

    if bit_random <= mutation_rate:
        # calculate mutation momentum
        mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
        # Mutate estimators
        model.n_estimators = model.n_estimators + math.ceil(model.n_estimators * mutation_momentum)

    return model


def evaluate_ga(dataset, label_data):
    """
    Evaluates and generates the ensemble model using Genetic Algorithms
    :param dataset:
    :return:
    """
    global best_lr_model
    print('#-----------------------------------------------')
    print('  ', dataset)
    print('#-----------------------------------------------')

    dataset1 = dataset
    process = Preprocess(config_path=config_path)
    # pdb.set_trace()

    dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st \
        , train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr = process.load_dataset(
        dataset)

    # pdb.set_trace()

    start = time.time()  # Start Timer

    num_population = random.randint(experiment_config["ga_parameters"]["min_population"],
                                    experiment_config["ga_parameters"]["max_population"])  # Number of RNN to evaluate
    # == 1) Generate initial population for RNN and Random Forest
    population_rnn = []
    population_rf = []
    population_lr = []
    start_ga_1 = time.time()  # Start Timer
    all_F1_rnn = []
    all_recall_rnn = []
    all_precision_rnn = []
    all_accuracy_rnn = []
    all_F1_rf = []
    all_recall_rf = []
    all_precision_rf = []
    all_accuracy_rf = []
    all_F1_lr = []
    all_recall_lr = []
    all_precision_lr = []
    all_accuracy_lr = []

    for i in range(num_population):
        # -- RNN
        # Generate random topology configuration
        num_layers = random.randint(experiment_config["ml_parameters"]["min_num_layers"],
                                    experiment_config["ml_parameters"]["max_num_layers"])
        hidden_layers = []
        for j in range(num_layers):
            num_neurons = random.randint(experiment_config["ml_parameters"]["min_num_neurons"],
                                         experiment_config["ml_parameters"]["max_num_neurons"])
            hidden_layers.append(num_neurons)

        # collect_gc()

        # Generate and add rnn model to population
        model_rnn = generate_rnn(hidden_layers)
        # print(model_rnn.summary())

        population_rnn.append(model_rnn)

        # -- RF
        # Generate random number of estimators for RF
        num_estimators = random.randint(experiment_config['ml_parameters']['min_num_estimators'],
                                        experiment_config['ml_parameters']['max_num_estimators'])

        # Generate and add rf model to population
        model_rf = generate_rnn(hidden_layers)
        population_rf.append(model_rf)
        model_lr = generate_rnn(hidden_layers)
        population_lr.append(model_lr)
    # get_stacking(  train_x_st,train_y,test_x_st,test_y,model_rnn,model_rf,model_lr)
    #        pdb.set_trace()

    end_ga_1 = time.time() - start_ga_1  # End Timer
    print('Generate Initial population Time_Taken:%.3f' % end_ga_1)

    Preprocess(sys.argv[1]).collect_gc()
    # print(len(population))

    best_F1_rnn = 0.0
    best_F1_rf = 0.0
    best_F1_lr = 0.0
    best_rnn_model = None
    best_test_predict_rnn = None
    best_rf_model = None
    best_test_predict_rf = None
    # Evaluate fitness for
    for i in range(experiment_config['ga_parameters']['num_Iterations']):
        print('=================================================================================================')
        print(' iteration: %d, total iterations: %d, population size: %d ' % (
            i + 1, experiment_config['ga_parameters']['num_Iterations'], num_population))
        print('=================================================================================================')
        # train_score, test_score = float("inf"), float("inf")
        # == 2)  Evaluate fitness for population
        start_ga_2 = time.time()  # Start Timer
        for j in range(num_population):
            # Evaluate fitness for RNN
            rnn_model = population_rnn[j]
            rnn_model, train_MC_score_rnn, test_MC_score_rnn, train_score_rnn, test_score_rnn, train_predict_rnn, test_predict_rnn, \
            test_MC_dist_predict_rnn, test_MC_predict_point_rnn, train_MC_dist_predict_rnn, train_MC_predict_point_rnn, test_MC_predict_var_rnn, \
            train_MC_predict_var_rnn, anomalies_rnn, precision_rnn, recall_rnn, Accuracy_rnn, F1_rnn = evaluate_rnn(
                rnn_model, train_x_stf,
                test_x_stf, train_y,
                test_y, scaler,
                optimisers[0], dataset1[7:-4], label_data)
            population_rnn[j] = rnn_model
            all_F1_rnn.append(F1_rnn)
            all_recall_rnn.append(recall_rnn)
            all_precision_rnn.append(precision_rnn)
            all_accuracy_rnn.append(Accuracy_rnn)

            training_time_rnn = rnn_model.train_time
            testing_time_rnn = rnn_model.test_time
            # pdb.set_trace()

            try:
                if (F1_rnn > best_F1_rnn):
                    best_rmse_rnn = test_score_rnn
                    # best_rnn_model = copy.copy(rnn_model)
                    best_rnn_model = rnn_model
                    # pdb.set_trace()
                    best_test_predict_rnn = test_predict_rnn
                    best_train_MC_score_rnn = train_MC_score_rnn
                    best_test_MC_score_rnn = test_MC_score_rnn
                    best_train_score_rnn = train_score_rnn
                    best_test_score_rnn = test_score_rnn
                    best_train_predict_rnn = train_predict_rnn
                    best_test_MC_dist_predict_rnn = test_MC_dist_predict_rnn
                    best_test_MC_predict_point_rnn = test_MC_predict_point_rnn
                    best_train_MC_dist_predict_rnn = train_MC_dist_predict_rnn
                    best_train_MC_predict_point_rnn = train_MC_predict_point_rnn
                    best_test_MC_predict_var_rnn = test_MC_predict_var_rnn
                    best_train_MC_predict_var_rnn = train_MC_predict_var_rnn
                    best_anomalies_rnn = anomalies_rnn
                    best_F1_rnn = F1_rnn
                    best_Accuracy_rnn = Accuracy_rnn
                    best_recall_rnn = recall_rnn
                    best_precision_rnn = precision_rnn
                    train_time_rnn = rnn_model.train_time
                    test_time_rnn = rnn_model.test_time

                    #              print('test predictions RNN: ', best_test_predict_rnn)
                    #             print('test_score RMSE RNN:%.3f ' % best_test_score_rnn)
                    #            print('test_score RMSE RNN_MC:%.3f ' % best_test_MC_score_rnn)
                    print('F1 RNN:%.3f ' % best_F1_rnn)
                    print('precision RNN_MC:%.3f ' % best_precision_rnn)
                    print('Recall RNN:%.3f ' % best_recall_rnn)
                    print('Accuracy RNN:%.3f ' % best_Accuracy_rnn)
                else:
                    F1_rnn = 0

            except:
                F1_rnn = 0
                pass
            # Evaluate fitness for RF
            # pdb.set_trace()

            rf_model = population_rf[j]

            rf_model, train_score_rf, test_score_rf, train_predict_rf, test_predict_rf, _, _, \
            _, _, _, _, variance_rf, \
            _, anomalies_rf, precision_rf, recall_rf, Accuracy_rf, F1_rf = evaluate_rnn(
                rf_model, train_x_stf,
                test_x_stf, train_y,
                test_y, scaler,
                optimisers[0], dataset1[7:-4], label_data)

            population_rf[j] = rf_model
            all_F1_rf.append(F1_rf)
            all_recall_rf.append(recall_rf)
            all_precision_rf.append(precision_rf)
            all_accuracy_rf.append(Accuracy_rf)

            # pdb.set_trace()
            try:
                # print('test predictions RF: ', test_predict_rf)
                #  print('test_score RMSE RF:%.3f ' % test_score_rf)
                # print('F1 RF:%.3f ' % F1_rf)

                if (F1_rf > best_F1_rf):
                    best_rmse_rf = test_score_rf
                    # best_rf_model = copy.copy(rf_model)
                    best_rf_model = rf_model
                    best_test_predict_rf = rf_model.test_predict
                    best_train_score_rf = rf_model.train_score
                    best_test_score_rf = rf_model.test_score
                    best_train_predict_rf = rf_model.train_predict
                    best_test_predict_rf = rf_model.test_predict
                    best_variance_rf = rf_model.variance
                    best_anomalies_rf = rf_model.anomalies
                    best_F1_rf = F1_rf
                    best_Accuracy_rf = Accuracy_rf
                    best_recall_rf = recall_rf
                    train_time_rf = rf_model.train_time
                    test_time_rf = rf_model.test_time
                    best_precision_rf = precision_rf
                    #               print('test predictions Rf: ', best_test_predict_rf)
                    #               print('test_score RMSE Rf:%.3f ' % best_test_score_rf)
                    print('F1 Rf:%.3f ' % best_F1_rf)
                    print('precision Rf:%.3f ' % best_precision_rf)
                    print('Recall Rf:%.3f ' % best_recall_rf)
                    print('Accuracy Rf:%.3f ' % best_Accuracy_rf)
                    training_time_rf = rf_model.train_time
                    test_time_rf = rf_model.test_time

                else:
                    F1_rf = 0
            except:
                F1_rf = 0
                pass

            lr_model = population_lr[j]
            # lr_model, train_score_lr, test_score_lr, train_predict_lr, test_predict_lr, variance_lr, anomalies_lr, precision_lr, \
            # recall_lr, Accuracy_lr, F1_lr = evaluate_lr(
            #   lr_model, train_x_lr_st,
            #  test_x_lr_st, train_y_lr, test_y_lr,
            # scaler, label_data)

            lr_model, train_score_lr, test_score_lr, _, _, train_predict_lr, test_predict_lr, \
            _, _, _, _, variance_lr, \
            _, anomalies_lr, precision_lr, recall_lr, Accuracy_lr, F1_lr = evaluate_rnn(
                rnn_model, train_x_stf,
                test_x_stf, train_y,
                test_y, scaler,
                optimisers[0], dataset1[7:-4], label_data)
            # pdb.set_trace()

            # rnn_model, train_MC_score_rnn, test_MC_score_rnn, train_score_rnn, test_score_rnn, train_predict_rnn, test_predict_rnn, \
            # test_MC_dist_predict_rnn, test_MC_predict_point_rnn, train_MC_dist_predict_rnn, train_MC_predict_point_rnn, test_MC_predict_var_rnn, \
            # train_MC_predict_var_rnn, anomalies_rnn, precision_rnn, recall_rnn, Accuracy_rnn, F1_rnn = evaluate_rnn(
            #   rnn_model, train_x_stf,
            #  test_x_stf, train_y,
            # test_y, scaler,
            # optimisers[0], dataset1[7:-4], label_data)

            population_lr[j] = lr_model
            all_F1_lr.append(F1_lr)
            all_recall_lr.append(recall_lr)
            all_precision_lr.append(precision_lr)
            all_accuracy_lr.append(Accuracy_lr)
            # pdb.set_trace()
            train_time_lr = lr_model.train_time
            test_time_lr = lr_model.test_time

            try:
                #    print('test predictions LR: ', test_predict_lr)
                #   print('test_score RMSE LR:%.3f ' % test_score_lr)
                print('F1 LR:%.3f ' % F1_lr)
                # pdb.set_trace()
                if (F1_lr > best_F1_lr):  # and (math.isnan(F1_lr) == False) and (math.isinf(F1_lr) == False) and (
                    #:
                    # best_rmse_lr = test_score_lr
                    best_lr_model = lr_model
                    best_test_predict_lr = test_predict_lr
                    best_train_score_lr = train_score_lr
                    best_test_score_lr = test_score_lr
                    best_train_predict_lr = train_predict_lr
                    best_test_predict_lr = test_predict_lr
                    best_variance_lr = variance_lr
                    best_anomalies_lr = anomalies_lr
                    best_F1_lr = F1_lr
                    best_Accuracy_lr = Accuracy_lr
                    best_recall_lr = recall_lr
                    best_precision_lr = precision_lr
                    train_time_lr = lr_model.train_time
                    test_time_lr = lr_model.test_time
                    # print('test predictions LR: ', best_test_predict_lr)
                    # print('test_score RMSE LR:%.3f ' % best_test_score_lr)
                    print('F1 LR:%.3f ' % best_F1_lr)
                    print('precision LR:%.3f ' % best_precision_lr)
                    print('Recall LR:%.3f ' % best_recall_lr)
                    print('Accuracy LR:%.3f ' % best_Accuracy_lr)
                else:
                    F1_lr = 0
            except:
                # print(Exception + 'LR')
                F1_lr = 0
                print(Exception)
                pass

        end_ga_2 = time.time() - start_ga_2  # End Timer
        print('Evaluate Fitness population Time_Taken:%.3f' % end_ga_2)

        # collect_gc()
        # == 3)  Select best individuals for next generation
        # pdb.set_trace()
        # Evaluate fitness for Leanier Regression

        # == 3) Create new population with new generations
        # Every generation will use the current best RNN and best RF to mate
        start_ga_3 = time.time()  # Start Timer
        for pop_index in range(num_population):
            # Select parents for mating
            # Element at pop_index as parent. This will be replaced with the new generation
            rnn_model_1 = population_rnn[pop_index]
            rf_model_1 = population_rf[pop_index]
            lr_model_1 = population_lr[pop_index]
            # pdb.set_trace()

            # 2 parent is the best found so far
            rnn_model_2 = best_rnn_model
            rf_model_2 = best_rf_model
            lr_model_2 = best_lr_model
            # pdb.set_trace()

            # == 4) Create new generation with crossover
            # pdb.set_trace()
            new_rnn_model = crossover_rnn(rnn_model_1, rnn_model_2)
            new_rf_model = crossover_rnn(rf_model_1, rf_model_2)
            new_lr_model = crossover_rnn(lr_model_1, lr_model_2)

            # == 5) Mutate new generation
            mutate_rnn(new_rnn_model)
            mutate_rnn(new_rf_model)
            mutate_rnn(new_lr_model)

            # Replace current model in population
            population_rnn[pop_index] = new_rnn_model
            population_rf[pop_index] = new_rf_model
            population_lr[pop_index] = new_lr_model

        end_ga_3 = time.time() - start_ga_3  # End Timer
        print('Generate new population Time_Taken:%.3f' % end_ga_3)

        collect_gc

    collect_gc

    end = time.time() - start  # End Timer
    # pdb.set_trace()

    # print('=============== BEST RNN ===============', dataset)
    # print('Best predictions: ', [x[0] for x in best_test_predict_rnn])
    # print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rnn, end))
    # save_plot_model_rnn(best_rnn_model)

    # print('=============== BEST RF ===============')
    # print('Best predictions: ', [x for x in best_test_predict_rf])
    # print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rf, end))

    # save_plot_model_rf(best_rf_model)
    # print(best_rf_model.get_params(deep=True))

    # Ensemble
    print('=============== Ensemble ===============')
    # pdb.set_trace()

    dict_ensemble = ensemble_stacking(best_rnn_model, best_lr_model, best_rf_model, best_test_predict_rnn,
                                      best_rmse_rnn, best_test_predict_lr, best_test_predict_rf, best_rmse_rf,
                                      test_y, scaler, train_x=train_x_stf, test_x=test_x_stf, train_y=train_y,
                                      test_y=test_y, train_time_rnn=train_time_rnn, test_time_rnn=test_time_rnn)
    # train_time_rf=train_time_rf,
    # test_time_rf=test_time_rf, train_time_lr=train_time_lr,
    #  test_time_lr=test_time_lr)

    # print('Ensemble averaging_values: ', averaging_values)
    #  print('Ensemble uncertainty_values: ', stacking_values_uncertainty)

    print('all_F1_rnn, all_precision_rnn, all_recall_rnn: ', all_F1_rnn, all_precision_rnn, all_recall_rnn)
    print('all_F1_rf, all_precision_rf, all_recall_rf: ', all_F1_rf, all_precision_rf, all_recall_rf)
    print('all_F1_lr, all_precision_lr, all_recall_lr: ', all_F1_lr, all_precision_lr, all_recall_lr)

    # pdb.set_trace()
    best_F1_rnn = max(all_F1_rnn)
    best_F1_rf = max(all_F1_rf)
    best_precision_rnn = max(all_precision_rnn)
    best_precision_rf = max(all_precision_rf)
    best_recall_rnn = max(all_recall_rnn)
    best_recall_rf = max(all_recall_rf)
    best_Accuracy_rnn = max(all_accuracy_rnn)
    best_Accuracy_rf = max(all_accuracy_rf)
    # pdb.set_trace()

    dict_ensemble['best_F1_rnn'] = best_F1_rnn
    dict_ensemble['best_F1_rf'] = best_F1_rf
    dict_ensemble['best_precision_rnn'] = best_precision_rnn
    dict_ensemble['best_precision_rf'] = best_precision_rf
    dict_ensemble['best_recall_rnn'] = best_recall_rnn
    dict_ensemble['best_recall_rf'] = best_recall_rf
    dict_ensemble['best_Accuracy_rnn'] = best_Accuracy_rnn
    dict_ensemble['best_Accuracy_rf'] = best_Accuracy_rf
    dict_ensemble['best_rmse_rnn'] = best_rmse_rnn
    dict_ensemble['best_rmse_rf'] = best_rmse_rf
    dict_ensemble['best_rmse_lr'] = test_score_lr
    dict_ensemble['best_test_predict_rnn'] = best_test_predict_rnn
    dict_ensemble['best_test_predict_rf'] = best_test_predict_rf
    dict_ensemble['best_test_predict_lr'] = best_test_predict_lr
    dict_ensemble['best_anomalies_rnn'] = best_anomalies_rnn
    dict_ensemble['best_anomalies_rf'] = best_anomalies_rnn
    dict_ensemble['best_anomalies_lr'] = anomalies_lr
    # dict_ensemble['variance_rf'] = best_variance_rf
    dict_ensemble['variance_rnn'] = best_test_MC_predict_var_rnn
    # dict_ensemble['variance_lr'] = best_variance_lr
    dict_ensemble['best_F1_lr'] = best_F1_lr
    dict_ensemble['best_Accuracy_lr'] = best_Accuracy_lr
    dict_ensemble['best_precision_lr'] = best_precision_lr
    dict_ensemble['best_recall_lr'] = best_recall_lr
    dict_ensemble['train_time_rnn'] = train_time_rnn
    # dict_ensemble['train_time_rf'] = train_time_rf
    # dict_ensemble['train_time_lr'] = train_time_lr
    dict_ensemble['test_time_rnn'] = test_time_rnn
    # dict_ensemble['test_time_rf'] = test_time_rf
    # dict_ensemble['test_time_lr'] = test_time_lr

    return dict_ensemble
