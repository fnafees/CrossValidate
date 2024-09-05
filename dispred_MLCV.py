import numpy as np
from sklearn.model_selection import cross_validate # allows multiple metrics
import pandas as pd
# from dispred_Data import readDataset
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
def readDataset(train_path,nox_test_path,pdb_test_path,validation,test,savedFeatures):     
   

   
   
   print("Selecting features")      
   columnNames=["DispredPred","DispredProba"]
   window_size=7
   Windowed_columnList=[]
   for columnName in columnNames:
      for t in range(window_size, 0,-1):               
               Windowed_columnList.append(f'{columnName}-{t}')
      for t in range(1, window_size+1):               
               Windowed_columnList.append(f'{columnName}+{t}')
                     
   ESM2StatsColumns=["ESM2650_mean", "ESM2650_std", "ESM2650_max", "ESM2650_min", "ESM2650_median", "ESM2650_skew", "ESM2650_kurtosis", "ESM2650_kurt", "ESM2650_var", "ESM2650_sem", "ESM2650_quantile"]
            
   window_size=7   
   ESM2StatsWindowed_columnList=[]
   for columnName in ESM2StatsColumns:
      for t in range(window_size, 0,-1):               
               ESM2StatsWindowed_columnList.append(f'{columnName}-{t}')
      for t in range(1, window_size+1):               
               ESM2StatsWindowed_columnList.append(f'{columnName}+{t}')                    
   ESM2ColumnName=[f'ESM2650_{i}' for i in range(1, 1281)]                  
   ESM1_StatsColumns=["ESM1_650_mean", "ESM1_650_std", "ESM1_650_max", "ESM1_650_min", "ESM1_650_median", "ESM1_650_skew", "ESM1_650_kurtosis", "ESM1_650_kurt", "ESM1_650_var", "ESM1_650_sem", "ESM1_650_quantile"]
   # ColumnName=["ProteinID",  "No_1", "AminoAcid_1",  "Target" ]+[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[ "No_2", "AminoAcid_2", "DispredProba","DispredPred", "AminoAcid_proba" ]
   # SelectedColumns=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   
   SelectedColumns_WithESM2=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   SelectedColumns_WithoutESM2=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList
   SelectedColumns_ESM1EMS2=[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   SelectedColumns_EMS2=["DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   SelectedColumns_EMS2_with_ID=["ProteinID", "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   SelectedColumns_EMS2WihoutDispredict3=[  "Terminal_posneg"]+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   
   # in production we mege the nox dataset
   ESMDispred=SelectedColumns_WithoutESM2
   ESM2Dispred=SelectedColumns_WithESM2
   
   # in production we mege the pdb dataset
   ESM2PDBDispred=SelectedColumns_EMS2
   
   SelectedColumns=SelectedColumns_EMS2_with_ID
   print("Selected Columns length: ",len(SelectedColumns))
   

   if savedFeatures:
      SelectedColumns = SelectedColumns_EMS2
      print("Reading Modified Saved Data")
      X_train=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/X_train.feather")
      y_train=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/y_train.feather")
      X_nox_test= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/X_nox_test.feather")
      y_nox_test= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/y_nox_test.feather")
      X_pdb_test= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/X_pdb_test.feather")
      y_pdb_test=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/y_pdb_test.feather")
      
      # print(f"Modified Saved Data has {len(X_train)} amino acids")
      # print(f"The columns in Modified Saved Data are {X_train.columns.tolist()}")
      print(f"Selecting Columns on saved data")
      X_train = X_train.loc[:, SelectedColumns]  
      X_nox_test = X_nox_test.loc[:, SelectedColumns]
      X_pdb_test = X_pdb_test.loc[:, SelectedColumns]
      
      #For PDB model
         # REmove mobidb missing values
      # print("Adding Mobidb Missing Values")
      mobidb=pd.read_csv("/home/SharedFiles/Wasi/MergedFeatures/DisPred/MobidbTrainingMissing.csv")   
      X_train["MobiDBmissing"]=mobidb["MobiDBmissing"] 
      X_train["Target"]=y_train
      print("Number of amino acids before removing MobiDB missing values:", len(X_train))
      print("Removing Mobidb Missing Values")  
      X_train=X_train[X_train["MobiDBmissing"]!=1]
      y_train= X_train.loc[:,"Target"  ]
      X_train.pop("MobiDBmissing")
      X_train.pop("Target")   
      if test:
         print("Test Mode. Using a fraction of the data.")
         X_train=X_train[:10000]
         y_train=y_train[:10000]
         X_nox_test=X_nox_test[:10000]
         y_nox_test=y_nox_test[:10000]
         X_pdb_test=X_pdb_test[:10000]
         y_pdb_test=y_pdb_test[:10000]
 
   else:
      
      print("Reading and Preprocessing Data")    
      trainData =pd.read_feather(train_path)
      nox_testData=pd.read_feather(nox_test_path)
      pdb_testData=pd.read_feather(pdb_test_path)
      
      if test:
         print("Test Mode. Using a fraction of the data. Size before shortening:")
         print(trainData.shape)
         trainData=trainData[:10000]
         nox_testData=nox_testData[:10000]
         pdb_testData=pdb_testData[:10000]    
         #endingRow = 10000    
         # SelectedColumns=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]

      # Add windowed files
      
      window_size=7
      mlflow.log_param("window_size", window_size)
      train_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/TrainSet_Dispred_window_"+str(window_size)+".feather")
      pdb_testData_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/PDBSet_Dispred_window_"+str(window_size)+".feather")
      nox_testData_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/NoxSet_Dispred_window_"+str(window_size)+".feather")
      
      print("Train Data: ",trainData.shape)
      print("Nox Test Data: ",nox_testData.shape)
      print("PDB Test Data: ",pdb_testData.shape)

      print("Added Windowing Files")
      #merge windowed files
      if test:
         trainData=pd.concat([trainData,train_windowed], axis=1).dropna()
         pdb_testData=pd.concat([pdb_testData,pdb_testData_windowed], axis=1).dropna()
         nox_testData=pd.concat([nox_testData,nox_testData_windowed], axis=1).dropna()
      else:
         trainData=pd.concat([trainData,train_windowed], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_testData_windowed], axis=1)
         nox_testData=pd.concat([nox_testData,nox_testData_windowed], axis=1)
      
      
      print("after window Train Data: ",trainData.shape)
      print("after window Nox Test Data: ",nox_testData.shape)
      print("after window PDB Test Data: ",pdb_testData.shape)
   
      # Relase memory
      del train_windowed
      del pdb_testData_windowed
      del nox_testData_windowed
      
      # Add ESM2 features
      print("Adding ESM2 Features")
      train_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2650M-TrainSET_complete.feather")
      pdb_testData_esm2= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2650M-disorder_pdb_complete.feather")
      nox_testData_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2650M-disorder_NOX_complete.feather")

      train_esm2 = train_esm2.loc[:, ESM2ColumnName]
      pdb_testData_esm2 = pdb_testData_esm2.loc[:, ESM2ColumnName] 
      nox_testData_esm2 = nox_testData_esm2.loc[:, ESM2ColumnName]         

      if test:
         trainData=pd.concat([trainData,train_esm2], axis=1).dropna()
         pdb_testData=pd.concat([pdb_testData,pdb_testData_esm2], axis=1).dropna()
         nox_testData=pd.concat([nox_testData,nox_testData_esm2], axis=1).dropna()
      else:
         trainData=pd.concat([trainData,train_esm2], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_testData_esm2], axis=1)
         nox_testData=pd.concat([nox_testData,nox_testData_esm2], axis=1)
      # Relase memory
      del train_esm2
      del pdb_testData_esm2
      del nox_testData_esm2
      
      #Add ESM2 Stats
      print("Adding ESM2 Stats")
      trainstats_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2_650M_Train_stats.feather")
      pdb_stats_esm2= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2_650M_pdb_stats.feather")
      nox_stats_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2_650M_nox_stats.feather")

      if test:
         trainData=pd.concat([trainData,trainstats_esm2], axis=1).dropna()
         pdb_testData=pd.concat([pdb_testData,pdb_stats_esm2], axis=1).dropna()
         nox_testData=pd.concat([nox_testData,nox_stats_esm2], axis=1).dropna()
      else:
         trainData=pd.concat([trainData,trainstats_esm2], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_stats_esm2], axis=1)
         nox_testData=pd.concat([nox_testData,nox_stats_esm2], axis=1)
      
      # Relase memory
      del trainstats_esm2
      del pdb_stats_esm2
      del nox_stats_esm2
      
      print("Adding ESM2 stats Windowed Files")
      window_size=11
      mlflow.log_param("window_size_esm2Stats", window_size)
      train_esm2Statswindowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/TrainSet_esm2Stats_window_"+str(window_size)+".feather")
      pdb_testDataesm2Stats_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/PDBSet_esm2Stats_window_"+str(window_size)+".feather")
      nox_testDataesm2Stats_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/NoxSet_esm2Stats_window_"+str(window_size)+".feather")

      # print("Train Data: ",trainData.shape)
      # print("Nox Test Data: ",nox_testData.shape)
      # print("PDB Test Data: ",pdb_testData.shape)
   
      #merge windowed files
      if test:
         trainData=pd.concat([trainData,train_esm2Statswindowed], axis=1).dropna()
         pdb_testData=pd.concat([pdb_testData,pdb_testDataesm2Stats_windowed], axis=1).dropna()
         nox_testData=pd.concat([nox_testData,nox_testDataesm2Stats_windowed], axis=1).dropna()
      else:
         trainData=pd.concat([trainData,train_esm2Statswindowed], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_testDataesm2Stats_windowed], axis=1)
         nox_testData=pd.concat([nox_testData,nox_testDataesm2Stats_windowed], axis=1)
      
      # print("after window Train Data: ",trainData.shape)
      # print("after window Nox Test Data: ",nox_testData.shape)
      # print("after window PDB Test Data: ",pdb_testData.shape)
      
      # Relase memory
      del train_esm2Statswindowed
      del pdb_testDataesm2Stats_windowed
      del nox_testDataesm2Stats_windowed
      
      #Add ESM1 Stats
      # print("Adding ESM1 Stats")
      # trainstats_esm1 = pd.read_feather("/home/mkabir3/Research/40_CAID3/6_MergeData/ESM1_650M_Train_stats.feather")
      # pdb_stats_esm1= pd.read_feather("/home/mkabir3/Research/40_CAID3/6_MergeData/ESM1_650M_pdb_stats.feather")
      # nox_stats_esm1 = pd.read_feather("/home/mkabir3/Research/40_CAID3/6_MergeData/ESM1_650M_nox_stats.feather")
   


      # trainData=pd.concat([trainData,trainstats_esm1], axis=1)
      # pdb_testData=pd.concat([pdb_testData,pdb_stats_esm1], axis=1)
      # nox_testData=pd.concat([nox_testData,nox_stats_esm1], axis=1)
      

      # Add Mobidb missing values
      # print("Adding Mobidb Missing Values")
      # mobidb=pd.read_csv("/home/mkabir3/Research/40_CAID3/9_MobiDB/MobidbTrainingMissing.csv")   
      # trainData["MobiDBmissing"]=mobidb["MobiDBmissing"]     
         
      # Add Amino acid probabilities ## No Improvements
      # # print("Adding Amino Acid Probabilities")
      # amino_acid_prob=pd.read_csv("/home/mkabir3/Research/40_CAID3/7_ExploratoryDataAnalysis/AminoAcid_disordered_Proba.csv")
      # amino_acid_prob["proportion"]=amino_acid_prob["proportion"]*10
      # trainData["AminoAcid_proba"]= trainData["AminoAcid_2"].replace(amino_acid_prob.set_index('AminoAcid')['proportion'].to_dict())
      # nox_testData["AminoAcid_proba"]= nox_testData["AminoAcid_2"].replace(amino_acid_prob.set_index('AminoAcid')['proportion'].to_dict())
      # pdb_testData["AminoAcid_proba"]= pdb_testData["AminoAcid_2"].replace(amino_acid_prob.set_index('AminoAcid')['proportion'].to_dict())

      # Remove large proteins
      print("Removing Large Proteins")
      listofLargeProteins = trainData.groupby('ProteinID').AminoAcid_1.count()[trainData.groupby('ProteinID').AminoAcid_1.count()>2000].index
      # Remove the proteins with more than 2000 amino acids
      trainData = trainData[~trainData.ProteinID.isin(listofLargeProteins)]
      print("Number of Unique Id Removed: ",len(listofLargeProteins))
      mlflow.log_metric("LargeProteinsRemoved", len(listofLargeProteins))
      
      # add Terminal values
      print("Adding Terminal Values")
      evenly_spaced_numbers_merged=terminal_gen(trainData)
      trainData["Terminal_posneg"]=evenly_spaced_numbers_merged
      trainData["Terminal_pos"]=np.abs(evenly_spaced_numbers_merged)

      evenly_spaced_numbers_merged=terminal_gen(nox_testData)
      nox_testData["Terminal_posneg"]=evenly_spaced_numbers_merged
      nox_testData["Terminal_pos"]=np.abs(evenly_spaced_numbers_merged)

      evenly_spaced_numbers_merged=terminal_gen(pdb_testData)
      pdb_testData["Terminal_posneg"]=evenly_spaced_numbers_merged
      pdb_testData["Terminal_pos"]=np.abs(evenly_spaced_numbers_merged)
      
      # REmove mobidb missing values
      # print("Removing Mobidb Missing Values")  
      # trainData=trainData[trainData["MobiDBmissing"]!=1]
      
      
      # Selecting targets
      print("Selecting Targets")
      # trainData["Target"]=trainData["Disorder_consensus"]
      trainData["Target"]=trainData["Disorder_consensus"]+trainData["Transition_consensus"]
   

      print("Selecting features") 
      X_train = trainData.loc[:, SelectedColumns]    
      y_train = trainData.loc[:,"Target"  ]
      print("X_train: ",X_train.shape)
      print("y_train: ",y_train.shape)

      X_nox_test = nox_testData.loc[:, SelectedColumns]
      y_nox_test = nox_testData.loc[:,"Target"  ]
      X_pdb_test = pdb_testData.loc[:, SelectedColumns]
      y_pdb_test = pdb_testData.loc[:,"Target"  ]
      print("X_nox_test: ",X_nox_test.shape)
      print("y_nox_test: ",y_nox_test.shape)

      print("X_pdb_test: ",X_pdb_test.shape)
      print("y_pdb_test: ",y_pdb_test.shape)
      X_pdb_test=X_pdb_test.iloc[np.where(y_pdb_test!="-")]
      y_pdb_test=y_pdb_test.iloc[np.where(y_pdb_test!="-")]
      y_pdb_test=y_pdb_test.astype(int)

      print("After Removing - X_pdb_test: ",X_pdb_test.shape)
      print("After Removing - y_pdb_test: ",y_pdb_test.shape)

      mlflow.log_metric("traindata_UniqueProteins", trainData.ProteinID.nunique())
      mlflow.log_metric("nox_testData_UniqueProteins", nox_testData.ProteinID.nunique())
      mlflow.log_metric("pdb_testData_UniqueProteins", pdb_testData.ProteinID.nunique())

      # Relase memory
      del trainData
      del nox_testData
      del pdb_testData

      # if validation:
      #     # Create validation set
      #     validationData = trainData.sample(frac=0.2, random_state=1)
      #     trainData = trainData.drop(validationData.index)
      #     trainData.reset_index(drop=True, inplace=True)
      #     validationData.reset_index(drop=True, inplace=True)
      #     print("Validation Data: ",validationData.shape)
      #     print("Train Data: ",trainData.shape)
      #     y_valid = validationData.loc[:,"Target"  ]
      #     X_valid = validationData.loc[:, SelectedColumns]
      #     print("X_valid: ",X_valid.shape)
      #     print("y_valid: ",y_valid.shape)
      #     return X_train, y_train, X_nox_test, y_nox_test, X_pdb_test, y_pdb_test, X_valid, y_valid

      # # convert pandas dataframe to numpy array
      # train = trainData.to_numpy()
      # np.random.shuffle(train)
      # y_train = train[:,-1]
      # X_train = train[:,:-1]

      # test = testData.to_numpy()
      # y_test = test[:,-1]
      # X_test = test[:,:-1]

      # if validation: 
      #     valid = validationData.to_numpy()
      #     y_valid = valid[:,-1]
      #     X_valid = valid[:,:-1]

      # Save all into a file     

      if test==False:
         print("Saving Modified Data")
         pathlib.Path("./SavedModifiedData").mkdir(parents=True, exist_ok=True) 
         X_train.to_feather("./SavedModifiedData/X_train.feather")
         pd.DataFrame(y_train).to_feather("./SavedModifiedData/y_train.feather")
         X_nox_test.to_feather("./SavedModifiedData/X_nox_test.feather")
         pd.DataFrame(y_nox_test).to_feather("./SavedModifiedData/y_nox_test.feather")
         X_pdb_test.to_feather("./SavedModifiedData/X_pdb_test.feather")
         pd.DataFrame(y_pdb_test).to_feather("./SavedModifiedData/y_pdb_test.feather")      

  
   print("Logging Parameters")
   mlflow.log_param("TestRun", test)
   mlflow.log_param("UsingsavedFeatures", savedFeatures)
   mlflow.log_param("train_path", train_path)
   mlflow.log_param("nox_test_path", nox_test_path)
   mlflow.log_param("pdb_test_path", pdb_test_path)
   mlflow.log_metric("traindata_NumberofFeatures", X_train.shape[1])
   mlflow.log_metric("nox_testData_NumberofFeatures", X_nox_test.shape[1])
   mlflow.log_metric("pdb_testData_NumberofFeatures", X_pdb_test.shape[1])
      
   return X_train, y_train, X_nox_test, y_nox_test, X_pdb_test, y_pdb_test
# X_train = 
# y_train = 

train_path = "/home/SharedFiles/Wasi/MergedFeatures/DisPred/disorder_disprot23-CAID2-TrainSET_complete.feather"
nox_test_path = "/home/SharedFiles/Wasi/MergedFeatures/DisPred/disorder_nox_complete.feather"
pdb_test_path= "/home/SharedFiles/Wasi/MergedFeatures/DisPred/disorder_pdb_complete.feather"

X, y, _, _, _, _ = readDataset(train_path,nox_test_path,pdb_test_path,True,False,True)
y = y.to_frame()

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

    # danet_config = DANetConfig(
    #     task="classification"
    # )
    # global danet
    # danet = TabularModel(
    #     data_config=data_config,
    #     model_config=danet_config,
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