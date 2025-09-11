"""
Module for all the functions to train a statistical model to predict avalanche danger. This does not include the
artificial neural network (ANN). For this see ANN.py.
"""


# imports
import sys
import numpy as np
from joblib import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# import proprietary functions
from .Set_Hyperparameters import set_hyperp
from .Grid_Search import perform_grid_search
from .Data_Loading import load_xlevel_preds
from .Helpers_Load_Data import extract_sea

# function for setting up the statistical model
def stat_mod(model_ty="RF", ndlev=2, hyperp={}, grid_search=False, grid_sample=0, class_weight=None,
             train_x=None, train_y=None, cv_type="seasonal", cv=3, cv_score="accuracy", balance_meth="SMOTE",
             verbose=True):

    """
    This function implements the use of the following statistical models specifically on the avalanche predictors as
    prepared by the functions load_xlevel_preds:
        - DT  = decision tree (set as default)
        - LR  = logistic regression
        - SVM = support vector machine
        - KNN = nearest neighbour
        - RF  = random forest

    The implementation without a function can be found in Gen_and_Store_StatisticalModel.py.

    The function here is intended to be used, i.a., in the feature testing, i.e., to find out which features best
    predict avalanche danger.

    A grid search for the optimal hyperparameters can be performed if required by the user. However, it must be noted
    that likely not all possible hyperparmeters will be searched and that the possible values given may not include the
    optimal one. The parameter search will be performed using a 5-fold crossvalidation (default). Note that if a grid
    search is performed, the training data (predictors and target) must be provided. Otherwise, no data are required by
    this function.

    Parameters:
        model_ty     String. The type of model to set up. Choices and their abbreviations are given above.
        ndlev        Integer. The number of classes in the target variable, i.e., the number of danger levels. This can
                              be either 1, 2, 3, or 4. This is here only used for the SVM to determine of the kernel is
                              linear (ndlev=2) or rbf (ndlev > 2).
        hyperp       Dictionary. Dictionary containing the hyperparameters or the hyperparameter grid (if grid search
                                 is True) for the chosen statistical model. Parameters available so far are:
                                     DT: max_depth, min_leaf_samp
                                     LR: C
                                     SVM: C
                                     KNN: n_neighbors, weights
                                     RF: n_estimators, max_depth, min_leaf_samp
        grid_search  Logical. If true, a grid search for optimal hyperparameters will be performed as described above.
                              In case the hyperp dictionary is not empty, it will be assumed to provide the parameter
                              grid for the grid search.
        grid_sample  Integer. If > 0, the grid search will be randomised (RandomizedSearchCV from scikit-learn) with
                              grid_sample being the number of parameter settings sampled. If set to 0 (default) the full
                              grids will be searched.
        class_weight None, "balanced", or dictionary. If None the data are assumed to be balanced (i.e., external
                                                      balancing). If "balanced" the data will be assumed to be
                                                      unbalanced (although this should also work if the data are
                                                      balanced; this should simply imply that the weights will be the
                                                      same for each class) and an internal balancing via automatically
                                                      calculated class weights is performed. If a dictionary is
                                                      submitted, this corresponds to manually setting the class weights.
                                                      Thus, make sure that the names of the elements correspond to the
                                                      class names.
                                                      Note that for the KNN no internal balancing is implemented as of
                                                      yet, meaning that in the case of "balanced" in fact the unbalanced
                                                      data are used WITHOUT internal balancing.
        train_x      Pandas DataFrame containing the predictors. Only required if grid_search=True, that is, the grid
                                      search is performed for these data. Default is None.
                                      If class_weight is None this should be the balanced data, if it is "balanced" it
                                      should be the unbalanced (i.e., full) data.
        train_y      Pandas DataFrame containing the target variable (i.e., the avalanche danger level). Only required
                                      if grid_search=True. That is, the grid search is performed for these data. Default
                                      is None.
                                      If class_weight is None this should be the balanced data, if it is "balanced" it
                                      should be the unbalanced (i.e., full) data.
        cv_type      String. The type of folds to generated. Choices are "stratified", meaning that the class
                             frequencies of the folds will be equal to those in the original data, or "seasonal",
                             meaning that the folds are predefined based on the years of the avalanche seasons.
                             Defaults to seasonal.
        cv           Integer. The number of folds in the gridsearch crossvalidation. Defaults to 5.
        cv_score     String or score object. The score to be used in the cross-validation. For possible choices see
                                             https://scikit-learn.org/stable/modules/model_evaluation.html
        balance_meth String. The method used for balancing the data when loading the predictors. Note that this does
                             not control what is happening during the model training, which is controlled by the
                             balancing parameter. If balancing is set to none or internal, the balance_meth parameter
                             becomes irrelevant.
                             Choices are the following:
                               -undersample: uses the custom undersample function
                               -SMOTE: uses the synthetic minority oversampling method from the imbalanced-learn library
                               -SVMSMOTE: SMOTE using an SVM
                               -ADASYN: uses the adaptive synthetic sampling method from the imbalanced-learn library
                               -ros: uses the random oversampling method from the imbalanced-learn library
                               -rus: uses the random undersampling method from the imbalanced-learn library
                             Default is SMOTE.
                             There is considerable difference in procedure between the undersample case and the other
                             cases: In the undersample case the predictors have already been balanced (script
                             Gen_Store_XLevel_Balanced_Predictors.py). In the other cases the predictors are only
                             balanced here.

    Output:
        The statistical model as returned by the scikit-learn functions.
    """

    # set the hyperparameters
    if cv_type == "seasonal":
        pipe = True
    else:
        pipe = True
    # end if

    if grid_sample > 0:
        hy_ndlev = 0
    else:
        hy_ndlev = ndlev
    # end if
    param_grid = set_hyperp(model_ty=model_ty, in_hypp=hyperp, grid_search=grid_search, grid_sample=grid_sample,
                            pipe=pipe, ndlev=hy_ndlev, verbose=verbose)

    # set some model parameters:

    # kernel for the SVM
    if ndlev == 2:
        kernel = "linear"
    elif ndlev > 2:
        kernel = "rbf"
    # end if elif

    # max iterations for the LR
    max_iter = 1000


    if grid_search:
        # if grid_search=True, check if train_x and train_y are provided.
        if ((train_x is None) | (train_y is None)):
            print("\nNo predictors or traget provided. This is required for the hyperparameter search. Aborting.\n")
            sys.exit("No data for gridsearch.")
        # end if

        hyperparameters = perform_grid_search(model_ty, param_grid, ndlev, cv_type, cv, cv_score, grid_sample,
                                              balance_meth, class_weight, train_x, train_y, kernel, max_iter=max_iter,
                                              verbose=verbose)
    else:
        hyperparameters = param_grid
    # end if else

    # make the keys in hyperparameters compatible with non-pipeline usage
    try:  # Python >= 3.9
        hyperparameters = hyperparameters | {k.split("__")[-1]:hyperparameters[k] for k in hyperparameters.keys()}
    except: # Python >= 3.5
        hyperparameters = {**hyperparameters, **{k.split("__")[-1]:hyperparameters[k] for k in hyperparameters.keys()}}
    # end try except

    # select the model
    if model_ty == "DT":

        # define the decision tree model
        model = DecisionTreeClassifier(criterion="gini", max_depth=hyperparameters["max_depth"],
                                       min_samples_leaf=hyperparameters["min_samples_leaf"],
                                       class_weight=class_weight)

    elif model_ty == "LR":

        # define the logistic regression model
        model = LogisticRegression(max_iter=max_iter, class_weight=class_weight, C=hyperparameters["C"])

    elif model_ty == "SVM":

        # create the SVM Classifier
        model = SVC(kernel=kernel, class_weight=class_weight, C=hyperparameters["C"])

    elif model_ty == "KNN":

        # create a NN Classifier
        model = KNeighborsClassifier(n_neighbors=hyperparameters["n_neighbors"], weights=hyperparameters["weights"])

    elif model_ty == "RF":

        # define the random forest model according to the best hyperparameters
        model = RandomForestClassifier(n_estimators=hyperparameters["n_estimators"],
                                       max_depth=hyperparameters["max_depth"],
                                       min_samples_leaf=hyperparameters["min_samples_leaf"],
                                       min_samples_split=hyperparameters["min_samples_split"],
                                       max_features=hyperparameters["max_features"],
                                       random_state=42, class_weight=class_weight)
    # end if elif

    # print the hyperparameters
    if verbose:
        print("\nThe following set/gird of hyperparameters is used:")
        print(hyperparameters)
        print()
    # end if

    # return the model
    return model
# end def


# function for applying the statistical model to data
def apply_stat_mod(model_ty="RF", ndlev=2, a_p="y", nan_handling="drop", hyperp={}, grid_search=False, grid_sample=0,
                   cv_on_all=False, balancing="external", class_weight=None, split=0.33, h_low=400, h_hi=900,
                   agg_type="mean", p=90,
                   reg_code="AllReg", sel_feats=["s3", "wspeed_max"], scale_x=False, sea="full", data_path="",
                   assume_path=True, balance_meth="SMOTE", cv_type="seasonal", cv=3, cv_score="accuracy",
                   return_model=False, verbose=True):

    """
    This function loads the avalange-danger predictors according to the given danger-level classification (ndlev).

    The function stat_mod is then applied to set up the requested statistical model (see the documentation of stat_mod
    for the available statistical models).

    Consequently, the model is used to predict the target variable (i.e., avalanche danger) and this is returned as the
    output.

    For more information please consult the documentation of stat_mod.

    Parameters:
        model_ty     String. The type of model to set up. See the stat_mod docmentation.
        ndlev        Integer. The number of classes in the target variable, i.e., the number of danger levels. This can
                              be either 1, 2, 3, or 4.
        a_p            String. The avalanche problem for which the danger level is chosen. Defaults to "y",
                               implying the general danger level. The names of the available avalanche problems are
                               stored in Avalanche_Problems.py under Lists_and_Dictionaries.
        nan_handling  String. Controls how NaNs are handled. Two possibilities: 1) "drop" (default), i.e., dropping all
                              rows where the danger level is NaN and 2) "zero", i.e., converting the NaNs to 0s.
        hyperp       Dictionary. Dictionary containing the hyperparameters or the hyperparameter grid (if grid search
                                 is True) for the chosen statistical model. Parameters available so far are:
                                     DT: max_depth, min_leaf_samp
                                     LR: C
                                     SVM: C
                                     KNN: n_neighbors, weights
                                     RF: n_estimators, max_depth, min_leaf_samp
        grid_search  Logical. If true, a grid search for optimal hyperparameters will be performed as described above.
                              In case the hyperp dictionary is not empty, it will be assumed to provide the parameter
                              grid for the grid search.
        grid_sample  Integer. If > 0, the grid search will be randomised (RandomizedSearchCV from scikit-learn) with
                              grid_sample being the number of parameter settings sampled. If set to 0 (default) the full
                              grids will be searched.
        cv_on_all     Logical. If True, all data (not only training) are used in the cross-validation during the grid
                               search. Defaults to False.
        balancing String. Either none (no balancing), internal (internal balancing, i.e., balancing via weights
                          during model training; not avaialable for KNN!), or external (i.e., the balancing is done
                          during the predictor loading; refer to the documentation of the parameter balance_meth and
                          the function load_xlevel_preds). Note that even if balancing is set to external, one can still
                          perform an internal balancing (or "weighting") of the classes by submitting to class_weight;
                          see below.
        class_weight Dictionary. None (default) or dictionary. This parameter is used to manually weight the different
                                 classes. If None (default), no internal balancing is performed for balancing = external
                                 and an automatic balancing is performed for balancing = internal.
        split     Possibility (1): Float between 0 and 1 (excluding these boundaries) to perform the train-test split
                                   using the scikit-learn function train_test_split. The value given to split then
                                   represents the fraction of the data that is used as test data.
                  Possibility (2): Integer representing the year of the winter to be excluded. Note that the year is
                                   interpreted as the year in which the season ends. That is, for the winter of 2019/20
                                   one must submit 2020. This can be a list of multiple years.
                  Possibility (3): Integer representing the region code of the region to be extracted as test data and
                                   removed from the training data. This can be a list of multiple regions.
        h_low        Integer. The lower altitude-threshold of the NORA3-gridcells from which the predictors are
                              calculated. Available are 400, 500, 600, and 700.
        h_hi         Integer. The upper altitude-threshold of the NORA3-gridcells from which the predictors are
                              calculated. Available are 900, 1000, 1100, and 1200.
        agg_type     String. The type of the aggregation of the grid cells. Either mean, median, or percentile.
                             Defaults to mean.
        p            Integer. The percentile if the grid-cell aggregation type is percentile. Not used if the agg_type
                              is not percentile.
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
                               -undersample: uses the custom undersample function (NO LONGER AVAILABLE)
                               -SMOTE: uses the synthetic minority oversampling method from the imbalanced-learn library
                               -BSMOTE: BorderlineSMOTE
                               -SVMSMOTE: SMOTE using an SVM
                               -ADASYN: uses the adaptive synthetic sampling method from the imbalanced-learn library
                               -ros: uses the random oversampling method from the imbalanced-learn library
                               -rus: uses the random undersampling method from the imbalanced-learn library
                               -SMOTEENN combination of over and undersampling
                               -SMOTETomek combination of over and undersampling
                             Default is SMOTE.
                             There is considerable difference in procedure between the undersample case and the other
                             cases: In the undersample case the predictors have already been balanced (script
                             Gen_Store_XLevel_Balanced_Predictors.py). In the other cases the predictors are only
                             balanced here. UPDATE: The "undersample" option is no longer available!
        sample_strat Float, String, or dict: The strategy used in the class balancing algorithm.
                                             Float only for binary classification. String possibilities: auto, all,
                                             minority, not_majority, not_minority. dict: Keys indicate the targeted
                                             classes and the values to desired number of samples per class.
        cv_type      String. The type of folds to generated. Choices are "stratified", meaning that the class
                             frequencies of the folds will be equal to those in the original data, or "seasonal",
                             meaning that the folds are predefined based on the years of the avalanche seasons.
                             Defaults to seasonal.
        cv           Integer. The number of folds in the gridsearch crossvalidation. Defaults to 5.
        cv_score     String or score object. The score to be used in the cross-validation. For possible choices see
                                             https://scikit-learn.org/stable/modules/model_evaluation.html
        return_model Logical. If true, the statistical model is returned, if false (default) it is not returned.
        verbose      Logical. If True (default) print statements will made. If False they are suppressed.

    Output:
        The predicted and true target variable (i.e., the danger level) in the following order:
            pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_y, train_x_all,\
                                                                              train_y_all, test_x_all, test_y_all, model
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
    print(data_path)

    # load the data
    print("\nLoading data.\n")
    train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all = \
                                    load_xlevel_preds(data_path=data_path, sel_feats=sel_feats, a_p=a_p,
                                                      nan_handling=nan_handling,
                                                      reg_code=reg_code,
                                                      split=split, sea=sea, nlevels=ndlev,
                                                      balance_meth=balance_meth, scale_x=scale_x,
                                                      verbose=verbose)

    if grid_search:
        # determine if the grid search should be performed using the >full< or just >training< (balanced) data
        if ((balancing == "external") & (cv_type == "stratified")):
            # print('((balancing == "external") & (cv_type == "stratified"))')
            if cv_on_all:
                print("\nReloading ALL data for gridsearch.")
                gs_x, gs_y, dummy1, dummy2 = load_xlevel_preds(data_path=data_path, sel_feats=sel_feats, a_p=a_p,
                                                               nan_handling=nan_handling,
                                                               reg_code=reg_code, split=0, sea=sea,
                                                               nlevels=ndlev, balance_meth=balance_meth,
                                                               scale_x=scale_x,
                                                               verbose=True)
            else:
                print("\nUsing only the training data in the grid search.\n")
                gs_x = train_x
                gs_y = train_y
            # end if else
        elif (((balancing == "internal") | (balancing == "none")) |
                                                                 ((balancing == "external") & (cv_type == "seasonal"))):
            # print('((balancing == "external") & (cv_type == "seasonal"))')
            if cv_on_all:
                print("\nReloading ALL data for gridsearch.")
                dummy1, dummy2, gs_x, gs_y = load_xlevel_preds(data_path=data_path, sel_feats=sel_feats, a_p=a_p,
                                                               nan_handling=nan_handling,
                                                               reg_code=reg_code, split=0, sea=sea,
                                                               nlevels=ndlev, balance_meth=balance_meth,
                                                               scale_x=scale_x,
                                                               verbose=True)
            else:
                print("\nUsing only the training data in the grid search.\n")
                gs_x = train_x_all
                gs_y = train_y_all
            # end if else
        # end if elif
    else:
        gs_x = None
        gs_y = None
    # end if grid_search else

    # fit the model wit the data depending on the choice of external (None) or internal ("balanced") balancing
    if balancing == "none":
        # print("\nData not balanced...\n")

        # define the model
        model = stat_mod(model_ty=model_ty, ndlev=ndlev, hyperp=hyperp, grid_search=grid_search,
                         grid_sample=grid_sample, class_weight=None,
                         train_x=gs_x, train_y=gs_y, cv=cv)

        # fit the model
        model.fit(train_x_all, train_y_all)
    elif balancing == "internal":
        # print("\nData balanced internally...\n")

        if model_ty == "KNN":
            print("""\nInternal balancing is not (yet) available for KNN.
The result corresponds to the unbalanced case.\n""")
        # end if

        # Since the default of class_weight is None but the default should be "balanced" if balancing == "internal"
        # we here set class_weight to "balanced" if it is None.
        if class_weight is None:
            class_weight = "balanced"
        # end if

        # define the model
        model = stat_mod(model_ty=model_ty, ndlev=ndlev, hyperp=hyperp, grid_search=grid_search,
                         grid_sample=grid_sample, class_weight=class_weight,
                         train_x=gs_x, train_y=gs_y, cv=cv)
        # fit the model
        model.fit(train_x_all, train_y_all)
    elif balancing == "external":
        # print("\nData balanced externally...\n")

        # define the model
        model = stat_mod(model_ty=model_ty, ndlev=ndlev, hyperp=hyperp, grid_search=grid_search,
                         grid_sample=grid_sample, class_weight=class_weight,
                         train_x=gs_x, train_y=gs_y, cv=cv)

        # fit the model
        model.fit(train_x, train_y)
    # end if elif

    # perform the prediction
    pred_test = model.predict(test_x)
    pred_train = model.predict(train_x)
    pred_train_all = model.predict(train_x_all)
    pred_test_all = model.predict(test_x_all)

    if return_model:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                              train_y_all, test_x_all, test_y_all, model
    else:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                                     train_y_all, test_x_all, test_y_all
    # end if else
# end def


# function for applying the statistical model to gridded data
def apply_stat_mod_gridded(model_ty="RF", ndlev=4, reg_code=3009, perc=99, split=[2021, 2023], bal_meth="SVMSMOTE",
                           hyperp="", grid_search=False, grid_sample=0, class_weight=None, cv=3, return_model=False,
                           pred_path=""):

    """
    Function for applying the stat_mod to the gridded predictors.

    For now it is assumed that balancing of the data is requested.
    """

    # set the name of the y variable
    y_name = "danger_level"

    # load the predictors for the best grid cells, i.e., the cells with the highest accuracies in the intial training
    df_dl_and_pred = load(pred_path + f"NORA3_Gridded_Predictors_Best{perc}th_Perc.joblib")[reg_code]

    # get the features and remove the unneeded ones
    sel_feats = list(df_dl_and_pred.columns)

    # remove unnecessary features --> the accuracies and the danger_level
    # --> leave elevation in for now
    sel_feats.remove("test_all_acc")
    sel_feats.remove("test_bal_acc")
    sel_feats.remove("danger_level")

    # remove NaNs
    n_nan = np.sum(df_dl_and_pred.isna().any(axis=1))
    # print(f"\n{n_nan} rows with NaNs exist (={n_nan/len(df_dl_and_pred)*100:.1f}%) in the dataset. Dropping them...")
    suff = f"\n{n_nan:10} NaN-rows (={n_nan/len(df_dl_and_pred)*100:5.1f}%) removed\n"
    df_dl_and_pred.dropna(inplace=True)

    # print the number of removed NaNs
    print(suff)

    # ad-hoc adjustment to get to the 2- or 4-ADL case
    if ndlev == 4:
        df_dl_and_pred.loc[df_dl_and_pred["danger_level"] == 5, "danger_level"] = 4
        df_dl_and_pred["danger_level"] = df_dl_and_pred["danger_level"] - 1
    elif ndlev == 2:
        df_dl_and_pred.loc[df_dl_and_pred["danger_level"] < 3, "danger_level"] = 0
        df_dl_and_pred.loc[df_dl_and_pred["danger_level"] >= 3, "danger_level"] = 1
    # end if elif

    # perform the data split
    train_x, test_x, train_y, test_y, train_x_all, test_x_all, train_y_all, test_y_all = \
                                              extract_sea(df_dl_and_pred, sel_feats, split=split, balance_meth=bal_meth,
                                                                                 target_n="danger_level", k_neighbors=5)


    # define the model
    model = stat_mod(model_ty=model_ty, ndlev=ndlev, hyperp=hyperp, grid_search=grid_search,
                     grid_sample=grid_sample, class_weight=class_weight,
                     train_x=df_dl_and_pred[sel_feats], train_y=df_dl_and_pred[y_name], cv=cv)

    # fit the model
    model.fit(train_x, train_y)

    # perform the prediction
    pred_test = model.predict(test_x)
    pred_train = model.predict(train_x)
    pred_train_all = model.predict(train_x_all)
    pred_test_all = model.predict(test_x_all)

    if return_model:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                              train_y_all, test_x_all, test_y_all, model
    else:
        return pred_test, pred_train, pred_test_all, pred_train_all, train_x, train_y, test_x, test_y, train_x_all,\
                                                                                     train_y_all, test_x_all, test_y_all
    # end if else

# end def