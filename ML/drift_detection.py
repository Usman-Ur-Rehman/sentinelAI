import numpy as np
import joblib,json
import threading,time
import redis
from collections import deque
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from feature_mapping import CICIDS_FEATURES


class driftDetector:

    def __init__(self,window_size=500,confidence_threshold=0.60,std_multiplier=2.0):
        
        self.window_size=window_size
        self.confidence_threshold=confidence_threshold
        self.std_multiplier=std_multiplier

        #so our drifts performs in such a way that it keeps check of previous 500
        #events and if avg confidence drops below  0.60 (default defined) tthen drifts action signled
        #and also even if the incoming event's any feature whose mean devaition deviated > 2 std from base line then drifts action signled

        #queue maintaining last 500 confidence scores
        self.confidence_scores=deque(maxlen=window_size)

        #queue for each feature out of 12 keeping previous 500 values of that particular feature in previous 500 events that came
        self.feature_windows={f:deque(maxlen=window_size) for f in CICIDS_FEATURES}

        #LOADING stats ie: mean,std of each individual feature across 500 previous events
        baseline_stats=joblib.load('ML/feature_baseline.pkl')

        # extracted avg mean & std of each feature across the previous 500 events
        self.baseline_avg_mean=baseline_stats['mean']
        self.baseline_avg_std=baseline_stats['std']

        #initalizing variables that handle drift detection
        self.avg_confidence_score = 1.0 #best Confidence avg initially
        self.drift_detected=False #no drift initially
        self.drift_reason=None #initally no reason bcz no drift detected
        self.drifted_features=[] #list of features due to whcih drift occoured(drifted features) initallly empty
        
        #connecting to redis so can write the drift status
        self.redis=redis.Redis(host='localhost',port=6379)

    def addResult(self,confidence: float,raw_features: dict):
        #so event came and adding recording it's results(confidence_score generate by ensemble.py and dict of features:value of that particular event)
        #add the confidence score to rolling window(queue of confidence scores)
            self.confidence_scores.append(confidence)

            #now extract all feature values from incoming event and push them into list of the corresponding feature's list to maintain previous 500 record values for each feature
            for f in CICIDS_FEATURES:
                value=raw_features.get(f,0.0)
                self.feature_windows[f].append(value)

            if len(self.confidence_scores)<50:# means if we have record for atleast 50 evetns then start checking for drifting otherwise not skip it because the model is not tested that much
                return
            

            #now if we have have data of more then 50 then start monitoring

            #-----------------------Confidence Monitoring---------------------------------#
            self.avg_confidence_score=float(np.mean(self.confidence_scores))
            confidence_drift=self.avg_confidence_score<self.confidence_threshold #if less means drifted makes it True

            #------------------------Feature's Monitoring---------------------------------#

            self.drifted_features = []  # rest
            for f in CICIDS_FEATURES:
                if len(self.feature_windows[f]) < 50:
                    continue

                rolling_mean = float(np.mean(self.feature_windows[f]))
                baseline_mean = self.baseline_avg_mean[f]
                baseline_std  = self.baseline_avg_std[f]

                if baseline_std == 0:
                    continue

                deviation = abs(rolling_mean - baseline_mean) / baseline_std
                if deviation > self.std_multiplier:
                    self.drifted_features.append(f)
            
            feature_drift = len(self.drifted_features) > 0# True if atleast 1 drift detected

            #------------------------Drift Reason---------------------------------#
            # --- Determine drift reason ---
            if confidence_drift and feature_drift:
                self.drift_detected = True
                self.drift_reason = 'both'
                print("DRIFT | confidence:", round(self.avg_confidence_score, 4), "| features:", self.drifted_features)
            elif confidence_drift:
                self.drift_detected = True
                self.drift_reason = 'confidence_drop'
                print("[DriftDetector] DRIFT (confidence) - avg:", round(self.avg_confidence_score, 4))
            elif feature_drift:
                self.drift_detected = True
                self.drift_reason = 'feature_drift'
                print("[DriftDetector] DRIFT (features) - drifted:", self.drifted_features)

            
            # write status to Redis so FastAPI /ml-stats can read it
            self.redis.set('drift_status', json.dumps({
            'avg_confidence': round(self.avg_confidence_score, 4),     
            'drift_detected': self.drift_detected,              
            'drift_reason': self.drift_reason,                  
            'drifted_features': self.drifted_features,          
            'window_size': len(self.confidence_scores)          
        }))
            
    def get_status(self) -> dict:
        return {
            'avg_confidence': round(self.avg_confidence_score, 4),
            'drift_detected': self.drift_detected,
            'drift_reason': self.drift_reason,
            'drifted_features': self.drifted_features,
            'window_size': len(self.confidence_scores)
        }
            



class AutoRetrainer:
    def __init__(self,driftDetector,model_path='ML/isolation_forest.pkl'):
        self.drift_detector = driftDetector  
        self.model_path = model_path
        self.live_model = None            
        self.swap_lock = threading.Lock()  

    def get_live_model(self):
        with self.swap_lock:
            return self.live_model
    

    def retrain_and_swap(self, X_new):
        print(f'[AutoRetrainer] Retraining — reason: {self.driftDetector.drift_reason}')

        new_model = IsolationForest(
            n_estimators=200,  
            contamination=0.2, 
            random_state=42,   
            n_jobs=-1           
        )
        new_model.fit(X_new)  
               
                              
