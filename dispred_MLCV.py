import numpy as np
from sklearn.model_selection import cross_validate # allows multiple metrics
import pandas as pd
from dispred_Data import readDataset
from sklearn.metrics import *
import mlflow
import mlflow.sklearn
import os
import random
import warnings
from sklearn.model_selection import StratifiedKFold
from pytorch_tabular import TabularModel
from pytorch_tabular.models import *
from pytorch_tabular.config import (
    DataConfig, 
    OptimizerConfig,
    TrainerConfig,
)

# Set a seed value to reporduce the results
seed_value= 2515  
os.environ['PYTHONHASHSEED']=str(seed_value) 
random.seed(seed_value) 
np.random.seed(seed_value)
################################
# Reading in dataset
# CHANGE THIS TO DESIRED DATA

X_train = 
y_train = 



#y = y.to_frame()
if "ProteinID" in X.columns:
    X.drop("ProteinID", inplace = True, axis=1)


##########################################

gandalf = None
danet = None
cem = None


def setUpPytorchModels():



    data_config = DataConfig(
    target=["Target"],
    continuous_cols= X.columns.to_list()[:-1]
    )

    trainer_config = TrainerConfig(
        # Lowered batch size from 1024 for FTTransformer
        batch_size=1024,
        max_epochs=20,
    )

    trainer_config_FT = TrainerConfig(
        batch_size=64,
        max_epochs=20
    )
    
    optimizer_config = OptimizerConfig()

    gandalf_config = GANDALFConfig(
        task="classification"
    )
    global gandalf
    gandalf = TabularModel(
        data_config=data_config,
        model_config=gandalf_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=False
    )

    danet_config = DANetConfig(
        task="classification"
    )
    global danet
    danet = TabularModel(
        data_config=data_config,
        model_config=danet_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=False
    )

    # mdn_config = MDNConfig(
    #     task="classification"
    # )
    # global mdn
    # mdn = TabularModel(
    #     data_config=data_config,
    #     model_config=mdn_config,
    #     optimizer_config=optimizer_config,
    #     trainer_config=trainer_config,
    #     verbose=False
    # )

    #CategoryEmbeddingModel
    cem_config = CategoryEmbeddingModelConfig(
        task="classification"
    )
    global cem
    cem = TabularModel(
        data_config=data_config,
        model_config=cem_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=False
    )

setUpPytorchModels()

topmodels = {
    "GandalfClassifier": (gandalf, True),
    "DANetClassifier": (danet, True),
    "CEMClassifier": (cem, True),
}

#####################################################
# Custom scorers
def F1_max_calc(y_true, y_proba1):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba1)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    # max_f1_thresh = thresholds[np.argmax(f1_scores)]
    return max_f1
    #, max_f1_thresh

f1Scorer = make_scorer(F1_max_calc)

def APS_calc(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

APSscorer = make_scorer(APS_calc)

def AUC_calc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

AUCscorer = make_scorer(AUC_calc)

customScorers = {
    "F1Max": f1Scorer,
    "APS": APSscorer,
    "AUC": AUCscorer
}
####################################
# Create same splits for all models
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)

# Cross validator
def testModel(X, y, model, customScorers, not_scikit):
    
    # list of metrics to score on
    # scoringMetrics = ['accuracy', 'roc_auc']

    if not_scikit:
        if "ProteinID" in X.columns:
            X.drop("ProteinID", inplace = True, axis=1)
        def _F1(y_true, y_pred):
            return F1_max_calc(y_true, y_pred["1_probability"].values)
        
        def _APS(y_true, y_pred):
            return APS_calc(y_true, y_pred["1_probability"].values)
        
        def _AUC(y_true, y_pred):
            return AUC_calc(y_true, y_pred["1_probability"].values)
        
        X["Target"] = y["Target"] 
        # Guides say that the primary database should have a label/target column
        # https://pytorch-tabular.readthedocs.io/en/latest/tutorials/09-Cross%20Validation/
        scores = {}
        f1_metrics = []
        aps_metrics = []
        auc_metrics = []
        datamodule = None
        currentModel = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #print(X.head())
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                train_fold = X.iloc[train_idx]
                val_fold = X.iloc[val_idx]
                if datamodule is None:
                    # Initialize datamodule and model in the first fold
                    # uses train data from this fold to fit all transformers
                    datamodule = model.prepare_dataloader(train=train_fold, validation=val_fold, seed=seed_value)
                    currentModel = model.prepare_model(datamodule)
                else:
                    datamodule = datamodule.copy(train=train_fold, validation=val_fold)
                # Train Model

                model.train(currentModel, datamodule)
                pred_df = model.predict(val_fold)
                # print(pred_df.head())
                # print(val_fold.head())
                f1_metrics.append(_F1(val_fold["Target"], pred_df))
                aps_metrics.append(_APS(val_fold["Target"], pred_df))
                auc_metrics.append(_AUC(val_fold["Target"], pred_df))
                model.model.reset_weights()
        scores["test_F1Max"] = f1_metrics
        scores["test_APS"] = aps_metrics
        scores["test_AUC"] = auc_metrics
        return scores
    else:
        # For all sci-kit learn methods
        if "ProteinID" in X.columns:
            X.drop("ProteinID", inplace = True, axis=1)
        scores = cross_validate(model, X, y.values.ravel(), cv=kf, scoring = customScorers, return_train_score=False)
            

    

    return scores
   

def run(): 
    if mlflow.active_run():
        mlflow.end_run()

    #mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("model_performance")

    # for each model, start new run and log stats
    for model_name, (model, not_scikit) in topmodels.items():
        mlflow.start_run(run_name=model_name)
        print("Testing", model_name)
        try:
            # Train and evaluate model
            scores = testModel(X_train, y_train, model, customScorers, not_scikit)
            
            
            
            F1scores = scores["test_F1Max"]
            APSscores = scores["test_APS"]
            AUCscores = scores["test_AUC"]

            mean_F1Max = np.mean(F1scores)
            std_F1max = np.std(F1scores)
            mean_APS = np.mean(APSscores)
            std_APS = np.std(APSscores)
            mean_AUC = np.mean(AUCscores)
            std_AUC = np.std(AUCscores)
            

            mlflow.log_metric("F1Mean", mean_F1Max)
            mlflow.log_metric("F1Std", std_F1max)
            mlflow.log_metric("APSMean", mean_APS)
            mlflow.log_metric("APSStd", std_APS)
            mlflow.log_metric("AUCMean", mean_AUC)
            mlflow.log_metric("AUCStd", std_AUC)

            #Get average of all stats so that we have a one number catch all
            averageAll = (mean_F1Max + mean_APS + mean_AUC) / 3.0

            mlflow.log_metric("mean_all_metrics", averageAll)
            
            # Log parameters
            try:
                mlflow.sklearn.log_model(model, model_name)
            except:
                print("Couldnt log model for:", model_name)
            try:
                mlflow.log_params(model.get_params())
            except:
                print("Couldn't log params for:", model_name)
            print(f"Logged metrics for {model_name}")
        
        finally:
            # End the MLflow run explicitly
            mlflow.end_run()
        

    print("Finished logging all models.")

run()