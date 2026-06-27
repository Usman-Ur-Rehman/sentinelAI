import numpy as np
import torch 
import joblib
from models import Autoencoder
from feature_mapping import CICIDS_FEATURES

CONFIDENCE_THRESHOLD=0.85

def loadModels():
    #Loading Isolation Forest Model and it's SHAP Explainer
    iso_forest=joblib.load('ML/isolation_forest.pkl')
    iso_explainer=joblib.load('ML/iso_shap_explainer.pkl')

    X_scaled=np.load('ML/X_scaled.npy')
    IF_score = -iso_forest.score_samples(X_scaled)
    IF_norm_factor = float(np.percentile(IF_score, 95))
    joblib.dump(IF_norm_factor, 'ML/IF_norm_factor.pkl')
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
    label_encoder=joblib.load('ML/label_encoder.pkl')#same label encoder used by classifier 
    scaler=joblib.load('ML/scaler.pkl')#same scaler used in datapreprocessing before training


    return iso_forest,iso_explainer,IF_norm_factor,ae_model,ae_threshold,classifier,label_encoder,scaler


def scoreEvent(event_features:list,models:tuple) -> dict:

    #extracting all models from models tuple(models returned by loadModels())

    iso_forest,iso_explainer,IF_norm_factor,ae_model,ae_threshold,classifier,label_encoder,scaler=models

    #now a feature comes and we have to find confidence score for it but before finding the score we scale all 12 features of  
    #that event using the same scaler we used earlier during preprocessing of data

    features_scaled=scaler.transform([event_features])

    #----------------------Passing Event From IF And Detecting Score-----------------#

    #passing those scaled features through isolation forestmodel so it gives us thier score
    #negated so higher the score means anomly without -ve lower the score shows more anaomly so 
    # we negated for our easiness
    isolation_forest_score=-iso_forest.score_samples(features_scaled)[0]

    # Isolation Forest classification Rule :(attack=-1 & normal =1)

    IF_is_attack=iso_forest.predict(features_scaled)[0]==-1
    #it will assign it true value if prediction by model =-1 (-1 means acc to IF it is anomly)


    #----------------------Passing Event From AE And Detecting Score-----------------#

    with torch.no_grad():
        #pytorch works with it's own only tensor dtype so converting features different dtypes to only identical tensor type
        features_scaled_tensor_dtype=torch.FloatTensor(features_scaled)

        #passing features to AE model to score them it will give reconstructed event
        reconstructed_event=ae_model(features_scaled_tensor_dtype)

        #finding reconstrcution error (by comparing orugnal features_scaled_tensor_dtypes event with reconstructed_event by subtracting them)
        reconstructed_error=torch.mean(
            (features_scaled_tensor_dtype-reconstructed_event)**2 
        ).item()

        AE_is_attack=reconstructed_error>ae_threshold



    #----------------------Ensembling Both Scores-----------------#


    # normalizing both scores produced by IF & AE
    IF_score_normalized=min(isolation_forest_score/IF_norm_factor,1.0)
    AE_score_normalized=min(reconstructed_error/ae_threshold,1.0)    

    # finding avg of both scores produced by each model
    confidence_score = (IF_score_normalized + AE_score_normalized) / 2.0


    #----------------------SHAP Reasoning-----------------#
    # now shap checkls which feature contributed at how much extent in making it anomoly
    #so first finding each featues importance

    shap_features_importance_values=iso_explainer.shap_values(features_scaled)[0]
    
    shap_explaination={k:round(float(v),4) for k,v in zip(CICIDS_FEATURES,shap_features_importance_values)}

    is_attack = IF_is_attack or AE_is_attack

    if is_attack and confidence_score>CONFIDENCE_THRESHOLD:
        #CI IS ABOVE 85 SO GET ATTACK LABEL FROM XGBOOST
        attack_type_encoded_value=classifier.predict(features_scaled)[0]
        #now converting encoded number to corresponding text label using same  LabelEncoder
        attack_type=label_encoder.inverse_transform([attack_type_encoded_value])[0]
    elif is_attack:
        attack_type='UNKNOWN_LOW_CONFIDENCE'
    else:
        attack_type='BENIGN'

    
    return {
        'is_attack': bool(is_attack),
        'attack_type': attack_type,
        'confidence_score': round(float(confidence_score), 4),
        'iso_forest_flagged': bool(IF_is_attack),
        'autoencoder_flagged': bool(AE_is_attack),
        'shap_explanation': shap_explaination,
        'raw_features': dict(zip(CICIDS_FEATURES, event_features))
    }





