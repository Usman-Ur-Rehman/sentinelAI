import numpy as np
import torch 
import joblib
from models import Autoencoder
from feature_mapping import CICIDS_FEATURES

CONFIDENCE_THREASHHOLD=0.85

def loadModels():
    #Loading Isolation Forest Model and it's SHAP Explainer
    iso_forest=joblib.load('ML/isolation_forest.pkl')
    iso_explainer=joblib.load('ML/iso_shap_explainer.pkl')


    #------------Autoencoder already imported-----------

    #loading AE configurations
    ae_config=joblib.load('ML/autoencoder_config.pkl') 
    
    #importing model architecture from lib(creating empty AE object)and initialzing autoencoder with input featues we saved
    ae_model=Autoencoder(ae_config['input_dim'])

    #loading it's trained weights
    ae_model.load_state_dict(torch.load('ML/autoencoder.pth'))

    ae_model.eval()#setting evaluation mode

    ae_threshold=ae_config['threshold']#loaded ROC curve threshold

    #-----------------------------XGBoost Model-----------------------------

    classifier=joblib.load('ML/attack_classifier.pkl')#XGBoost Model(classifier)
    label_encoder=joblib.load('ML/label.encoder.pkl')#same label encoder used by classifier 
    scaler=joblib.load('ML/scaler.pkl')#same scaler used in datapreprocessing before training


    return iso_forest,iso_explainer,ae_model,ae_threshold,classifier,label_encoder,scaler


def scoreEvent():
    
    iso_forest, iso_explainer, ae_model, ae_threshold, clf, le, scaler = models
    
    
    features_scaled = scaler.transform([event_features])

   
    iso_score = -iso_forest.score_samples(features_scaled)[0]

    iso_is_attack = iso_forest.predict(features_scaled)[0] == -1
