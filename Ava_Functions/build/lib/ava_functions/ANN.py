"""
Python module containing the artificial neural network (ANN) related functions.
"""

# imports
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import HeNormal
from keras.optimizers import Adam

# proprietary functions
from .Data_Loading import load_xlevel_preds
from .Convert_to_Categorcal import prob_to_cat

# function for setting up an ANN sequentially
def ANN(input_shape, n_in_nodes, in_dropout, n_hid_nodes, dropouts, n_out_nodes,
        activ_out="softmax", learning_rate=0.001, loss="categorical_crossentropy"):


    """
    Set up an artificial neural network (ANN) based on the Keras Sequential methodology. The ANN model is returned and
    can then be used to fit the data.

    Parameters:
        input_shape   Integer. Equals the number of features that are used to predict the target variable.
        n_in_nodes    Integer. The number of nodes in the input layer.
        in_dropout    Float. The dropout fraction of the input layer.
        n_hid_nodes   List of integers. The number of nodes in the hidden layers. The length of the list determines the
                                        number of hidden layers.
        n_out_nodes   Integer. The number of the nodes of the output layer. This must correspond to the number of
                               classes in the target variable. This corresponds to the number danger levels that are
                               predicted. Note that for a binary problem (i.e., two classes) this must be 1, while for
                               three or more classes this must equate the number of classes.
        dropouts      List of floats. The dropout fractions of the hidden layers. The legnth of this list must equate
                                      the length of n_hid_nodes.
        n_out_nodes   Integer. The number of the nodes of the output layer. This must correspond to the number of
                               classes in the target variable. This corresponds to the number danger levels that are
                               predicted. Note that for a binary problem (i.e., two classes) this must be 1, while for
                               three or more classes this must equate the number of classes.
        activ_out      String. The activation function in the output layer. For a binary target variable this should be
                               set to "sigmoid", while for a multi-class problem (i.e., n_out_nodes > 2) it should be
                               "softmax" (default).
        learning_rate  Float. Learning rate to be applied in the optimiser (which in this function is fixed to Adam).
                              Defaults to 0.001.
        loss           String. The loss function to be applied. For a binary target variable this should be
                               "binary_crossentropy" while for a multi-class problem (i.e., n_out_nodes > 2) it should
                               be "categorical_crossentropy" (default).
    """


    model = Sequential()  # build the model sequentially, kind of 'by hand'

    # first layer
    model.add(Dense(n_in_nodes, input_shape=(input_shape,), activation='relu', kernel_initializer=HeNormal()))
    Dropout(in_dropout)

    # add hidden layers using a for-loop
    for n_n, drop in zip(n_hid_nodes, dropouts):
        model.add(Dense(n_n, activation='relu', kernel_initializer=HeNormal()))
        Dropout(drop)
    # end for n_n, drop

    # the output layer has one node and uses the sigmoid activation function
    model.add(Dense(n_out_nodes, activation=activ_out))

    # Create the optimizer with a specific learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model

# end def


# function for fitting the ANN to data
def fit_ANN(train_x, train_y, test_x, test_y, test_x_all, test_y_all,
            input_shape, n_in_nodes, in_dropout, n_hid_nodes, dropouts, n_out_nodes,
            activ_out="softmax", learning_rate=0.001, loss="categorical_crossentropy",
            epochs=100, batch_size=64, stop_early=True, es_params={"monitor":"val_loss", "patience":20},
            class_weight=None, shuffle=True, verbose=-1):


    """
    Set up and fit an artificial neural network (ANN) based on the Keras Sequential methodology. The function ANN is
    used to set up the model. Both the model and the fitting history are returned.

    Parameters:
        train_x       DataFrame. The predictors on which the ANN is trained. Note that the classes should be balanced if
                                 class_weight is set to None (default).
        train_y       DataFrame. The target variable (in one-hot format! --> if not, use the to_categorical function
                                 from keras.utils to convert to one-hot) data set on which the ANN is trained. Note that
                                 the classes should be balanced if class_weight is set to None (default).
        test_x        DataFrame. The predictors which are used to validate the ANN.
        test_y        DataFrame. The target variable data set (in one-hot format! --> if not, use the to_categorical
                                 function from keras.utils to convert to one-hot) used to validate the ANN.
        input_shape   Integer. Equals the number of features that are used to predict the target variable.
        n_in_nodes    Integer. The number of nodes in the input layer.
        in_dropout    Float. The dropout fraction of the input layer.
        n_hid_nodes   List of integers. The number of nodes in the hidden layers. The length of the list determines the
                                        number of hidden layers.
        dropouts      List of floats. The dropout fractions of the hidden layers. The legnth of this list must equate
                                      the length of n_hid_nodes.
        activ_out      String. The activation function in the output layer. For a binary target variable this should be
                               set to "sigmoid", while for a multi-class problem (i.e., n_out_nodes > 2) it should be
                               "softmax" (default).
        learning_rate  Float. Learning rate to be applied in the optimiser (which in this function is fixed to Adam).
                              Defaults to 0.001.
        loss           String. The loss function to be applied. For a binary target variable this should be
                               "binary_crossentropy" while for a multi-class problem (i.e., n_out_nodes > 2) it should be
                               "categorical_crossentropy" (default).
        epochs         Integer. The number of epochs the ANN iterates over during training. Defaults to 100.
        batch_size     Integer. The batch size that the ANN uses during one iteration in the training process. Defaults
                                to 64.
        stop_early     Logical. If True, an early-stopping procedure is used which interupts the training process if the
                                loss (default) no longer reduces sufficiently after a given number of epochs (default:
                                20). The weights of the best iteration are restored.
        es_params      Dictionary of parameters to be used in the early-stopping procedure. Defaults to
                                  monitor="val_loss" (i.e., the loss with respect to the validation data is used as a
                                  heuristic for the early stopping) and patience=20, i.e., if there is no decrease of the
                                  loss after 20 iterations/epochs the training is stopped. Note that there are more
                                 parameters which are not implemented yet.
        class_weight   None or "balanced". If None the data are assumed to be balanced (i.e., external balancing), if
                                           "balanced" the data will be assumed to be unbalanced and an internal
                                           balancing via automatically calculated class weights is performed. Note that
                                           for the KNN no internal balancing is implemented as of yet, meaning that in
                                           the case of "balanced" in fact the unbalanced data are used WITHOUT internal
                                           balancing.
        shuffle        Logical. If True (default), the training data are shuffled to make sure that they are not sorted,
                                which may influence model training.
        verbose        Integer. The degree to which print statements are made during training. Set to -1 (default) for no
                                prints at all, except for the those during the fitting procedure.
    """

    # compute class weights
    if class_weight is None:
        class_weights = np.repeat(1, len(np.unique(train_y)))
    elif class_weight == "balanced":
        # since the data are in one-hot format convert them back to the standard
        temp_y = np.argmax(train_y, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(temp_y), y=temp_y)
    # end if elif
    class_weight_dict = dict(enumerate(class_weights))

    if verbose > -1:
        print("\nClass weights:")
        print(class_weight_dict)
        print()
    else:
        verbose = 0
    # end if

    # set up the early stopping
    if stop_early:
        early_stopping = [EarlyStopping(monitor=es_params["monitor"], patience=es_params["patience"], verbose=verbose,
                                        mode='min', restore_best_weights=True)]
    else:
        early_stopping = []
    # end if

    # convert the train and test data to the one-hot format
    ndlev = len(np.unique(train_y))
    if ndlev > 2:
        train_y_1h = to_categorical(train_y, num_classes=ndlev)
        test_y_1h = to_categorical(test_y, num_classes=ndlev)
        test_y_all_1h = to_categorical(test_y_all, num_classes=ndlev)
    else:
        train_y_1h = train_y
        test_y_1h = test_y
        test_y_all_1h = test_y_all
    # end if else

    # use the ANN function to set up the model
    model = ANN(input_shape, n_in_nodes, in_dropout, n_hid_nodes, dropouts, n_out_nodes, activ_out, learning_rate, loss)

    # fit the model to the data
    history = model.fit(train_x, train_y_1h, epochs=epochs, batch_size=batch_size, verbose=0,
                        callbacks=early_stopping, shuffle=shuffle, validation_data=(test_x_all, test_y_all_1h),
                        class_weight=class_weight_dict)

    return model, history

# end def


# function for applying the ANN
def apply_ANN(ndlev, n_in_nodes, in_dropout, n_hid_nodes, dropouts,
              activ_out="softmax", learning_rate=0.001, loss="categorical_crossentropy",
              epochs=100, batch_size=64, stop_early=True, es_params={"monitor":"val_loss", "patience":20},
              a_p="y", nan_handling="drop",
              class_weight=None, shuffle=True,
              balancing="external", split=0.33,
              agg_type="mean", p=90,
              h_low=400, h_hi=900, reg_code="AllReg", sel_feats=["s3", "wspeed_max"], scale_x=False, sea="full",
              data_path="", assume_path=True, balance_meth="undersample",
              return_categ=True, return_model=False, verbose=0):


    """
    Set up and apply artificial neural network (ANN) based on the Keras Sequential methodology. The function ANN is used
    to set up the model and the function fit_ANN to fit the model.

    Parameters:
        ndlev        Integer. The number of classes in the target variable, i.e., the number of danger levels. This can
                              be either 1, 2, 3, or 4.
        n_in_nodes    Integer. The number of nodes in the input layer.
        in_dropout    Float. The dropout fraction of the input layer.
        n_hid_nodes   List of integers. The number of nodes in the hidden layers. The length of the list determines the
                                        number of hidden layers.
        dropouts      List of floats. The dropout fractions of the hidden layers. The legnth of this list must equate
                                      the length of n_hid_nodes.
        activ_out      String. The activation function in the output layer. For a binary target variable this should be
                               set to "sigmoid", while for a multi-class problem (i.e., n_out_nodes > 2) it should be
                               "softmax" (default).
        learning_rate  Float. Learning rate to be applied in the optimiser (which in this function is fixed to Adam).
                              Defaults to 0.001.
        loss           String. The loss function to be applied. For a binary target variable this should be
                               "binary_crossentropy" while for a multi-class problem (i.e., n_out_nodes > 2) it should
                               be "categorical_crossentropy" (default).
        epochs         Integer. The number of epochs the ANN iterates over during training. Defaults to 100.
        batch_size     Integer. The batch size that the ANN uses during one iteration in the training process. Defaults
                                to 64.
        stop_early     Logical. If True, an early-stopping procedure is used which interupts the training process if the
                                loss (default) no longer reduces sufficiently after a given number of epochs (default:
                                20). The weights of the best iteration are restored.
        es_params      Dictionary of parameters to be used in the early-stopping procedure. Defaults to
                                  monitor="val_loss" (i.e., the loss with respect to the validation data is used as a
                                  heuristic for the early stopping) and patience=20, i.e., if there is no decrease of
                                  the loss after 20 iterations/epochs the training is stopped. Note that there are more
                                  parameters which are not implemented yet.
        a_p            String. The avalanche problem for which the danger level is chosen. Defaults to "y",
                               implying the general danger level. The names of the available avalanche problems are
                               stored in Avalanche_Problems.py under Lists_and_Dictionaries.
        nan_handling  String. Controls how NaNs are handled. Two possibilities: 1) "drop" (default), i.e., dropping all
                              rows where the danger level is NaN and 2) "zero", i.e., converting the NaNs to 0s.
        class_weight   None or "balanced". If None the data are assumed to be balanced (i.e., external balancing), if
                                           "balanced" the data will be assumed to be unbalanced and an internal
                                           balancing via automatically calculated class weights is performed. Note that
                                           for the KNN no internal balancing is implemented as of yet, meaning that in
                                           the case of "balanced" in fact the unbalanced data are used WITHOUT internal
                                           balancing.
        shuffle        Logical. If True (default), the training data are shuffled to make sure that they are not sorted,
                                which may influence model training.
        balancing String. Either none (no balancing), internal (internal balancing, i.e., balancing via weights
                          during model training; not avaialable for KNN!), or external (i.e., the balancing is done
                          during the predictor loading; refer to the documentation of the parameter balance_meth and
                          the function load_xlevel_preds).
        split     Possibility (1): Float between 0 and 1 (excluding these boundaries) to perform the train-test split
                                   using the scikit-learn function train_test_split. The value given to split then
                                   represents the fraction of the data that is used as test data.
                  Possibility (2): Integer representing the year of the winter to be excluded. Note that the year is
                                   interpreted as the year in which the season ends. That is, for the winter of 2019/20
                                   one must submit 2020.
        agg_type     String. The type of the aggregation of the grid cells. Either mean, median, or percentile.
                             Defaults to mean.
        p            Integer. The percentile if the grid-cell aggregation type is percentile. Not used if the agg_type
                              is not percentile.
        h_low        Integer. The lower altitude-threshold of the NORA3-gridcells from which the predictors are
                              calculated. Available are 400, 500, 600, and 700.
        h_hi         Integer. The upper altitude-threshold of the NORA3-gridcells from which the predictors are
                              calculated. Available are 900, 1000, 1100, and 1200.
        reg_code     Either "AllReg" to load the predictors from all 5 northern Norwegian regions or the individual
                             region code: 3009 (Nord-Troms), 3010 (Lyngen), 3011 (Tromsoe), 3012 (Soer-Troms), or 3013
                             (Indre Troms).
        sel_feats    List. A list of the features (i.e., predictors) on which the model is trained to predict the target
                           (i.e., the danger level).
        scale_x      Logical. If true, the predictors will be scaled using the scikit-learn scaler. If false (default),
                              the predictors are used without being scaled.
        sea          String. Either "full", meaning the predictors for the whole avalanche season (Dec-May) are loaded,
                             or "winter" (Dec-Feb) or "spring" (Mar-May), for the respective sub-seasons.
        data_path    String. The path to the predictors. If assume_path=True only provide the path to the directory that
                             contains Avalanche_Predictors_XLevel, where X can be 2, 3, or 4. If assume_path=False
                             provide the complete path NOT including the file name. The file name is assumed to be (in
                             f-string notation);
                             f"Features_{ndlev}Level_All_Between{h_low}_{h_hi}m_{reg_code}.csv"
        assume_path  Logical. If true, the following directory structure is assumed (using f-string notation):
                              full_path = f"{data_path}/Avalanche_Predictors_{ndlev}Level/Between{h_low}_and_{h_hi}m/"
                              Otherwise, provide the path as described in the description of data_path.
        balance_meth String. The method used for balancing the data when loading the predictors. Note that this does
                             not control what is happening during the model training, which is controlled by the
                             balancing parameter. If balancing is set to none or internal, the balance_meth parameter
                             becomes irrelevant.
                             Choices are the following:
                               -undersample: uses the custom undersample function (default)
                               -SMOTE: uses the synthetic minority oversampling method from the imbalanced-learn library
                               -ADASYN: uses the adaptive synthetic sampling method from the imbalanced-learn library
                               -ros: uses the random oversampling method from the imbalanced-learn library
                               -rus: uses the random undersampling method from the imbalanced-learn library
                             There is considerable difference in procedure between the undersample case and the other
                             cases: In the undersample case the predictors have already been balanced (script
                             Gen_Store_XLevel_Balanced_Predictors.py). In the other cases the predictors are only
                             balanced here.
        return_categ Logical. If True (default), the predicted values will be returned as categorical values instead of
                              probabilities, which is the output from the neural network (and returned if the parameter
                              is set to False).
        return_model Logical. If True, the statistical model is returned, if false (default) it is not returned.
        verbose      Integer. The degree to which print statements are made during training. Set to 0 for no prints.
    """

    # generate a name prefix/suffix depending on the gridcell aggregation
    # make sure that the percentile is 0 if the type is not percentile
    if agg_type != "percentile":
        p = 0
    # end if
    agg_str = f"{agg_type.capitalize()}{p if p != 0 else ''}"

    # set up the path
    if assume_path:
        data_path = f"{data_path}/Avalanche_Predictors_{ndlev}Level/{agg_str}/Between{h_low}_and_{h_hi}m/"
    # end if

    # load the data
    train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all = \
                                            load_xlevel_preds(data_path=data_path, sel_feats=sel_feats,
                                                              reg_code=reg_code, split=split, sea=sea, nlevels=ndlev,
                                                              a_p=a_p, nan_handling=nan_handling,
                                                              balance_meth=balance_meth, scale_x=scale_x,
                                                              verbose=verbose)

    # use the ANN function to set up the model
    input_shape = len(sel_feats)
    if ndlev == 2:
        n_out_nodes = 1
    else:
        n_out_nodes = ndlev
    # end if else

    model, history = fit_ANN(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_x_all=test_x_all,
                             test_y_all=test_y_all,
                             input_shape=input_shape, n_in_nodes=n_in_nodes, in_dropout=in_dropout,
                             n_hid_nodes=n_hid_nodes, dropouts=dropouts, n_out_nodes=n_out_nodes,
                             activ_out=activ_out, learning_rate=learning_rate, loss=loss,
                             epochs=epochs, batch_size=batch_size, stop_early=stop_early, es_params=es_params,
                             class_weight=class_weight, shuffle=shuffle, verbose=-1)

    # perform the prediction
    pred_test = model.predict(test_x, verbose=verbose)
    pred_train = model.predict(train_x, verbose=verbose)
    pred_train_all = model.predict(train_x_all, verbose=verbose)
    pred_test_all = model.predict(test_x_all, verbose=verbose)

    if return_categ:
        pred_test = prob_to_cat(pred_test)
        pred_train = prob_to_cat(pred_train)
        pred_test_all = prob_to_cat(pred_test_all)
        pred_train_all = prob_to_cat(pred_train_all)
    # end if

    if return_model:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                              train_y_all, test_x_all, test_y_all, model
    else:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                              train_y_all, test_x_all, test_y_all
    # end if else

# end def

