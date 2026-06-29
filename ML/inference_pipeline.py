import redis                
import json                 
import numpy as np        
import sys                 
import threading  

sys.path.append('ML')   # add ML folder to path so Python finds our modules

from ensemble import loadModels, scoreEvent                 
from drift_detection import driftDetector, AutoRetrainer      
from feature_mapping import generator_event_to_features      
# imported all content from ensemble.py,frift_detection.py and feature_mapping.py

r=redis.Redis(host='localhost',port=6379)

#loading all models(IF,AE,XGBoost)
print('Loading Models')
models=loadModels()

#for retraining loading data but retraining on drifting happens only on normal data
X_Scaled=np.load('ML/X_scaled.npy')
y_binary=np.load('ML/y_binary.npy')

X_normal_for_retraining=X_Scaled[y_binary==0]

#starting driftDetector
drift_detector=driftDetector(window_size=500,confidence_threshold=0.60,std_multiplier=2.0)

#starting autoRetrainer 
retrainer = AutoRetrainer(drift_detector, model_path='ML/isolation_forest.pkl')

monitor_thread = threading.Thread(
    target=retrainer.monitor_loop,   
    args=(X_normal_for_retraining,),    
    daemon=True                      
)

monitor_thread.start()  

print('Inference pipeline ready-Listening to threat_stream')

last_id = '$' 

while True:
    events = r.xread({'threat_stream': last_id}, count=5, block=1000)
    if events:
        for stream,messages in events:
            for msg_id, msg_data in messages:
                event=json.loads(msg_data[b'data'])

                try:

                    features=generator_event_to_features(event)
                    result=scoreEvent(event,models)#returns dict containing scored details
                    result['orignal_event']=event#add orignal event details(in orignal gen.py format)so that agent can block it
                    r.xadd('scored_stream',{'data':json.dumps(result)})#writing result into scored_stream that will go into agent

                    driftDetector.addResult(
                        confidence=result['confidence_score'],#current event confidence score calculated during ensembling
                        raw_features=result['raw_features']#12 cicids features of current event
                    )

                    #after scoring from ensemble showing final verdict that is it attack after decision of enembling's Rule or not
                    if result['is_attack']:
                        print(f"ATTACK: {result['attack_type']} | conf: {result['confidence_score']}")
                    else:
                        print(f"Normal | conf: {result['confidence_score']}")

                except Exception as e:
                      #  during evaluating one event if any thing like drift detection or scoring fails it simple catches that event and skips it and prints 
                      # and moves to next one so that the  whole pipeline is not disturbed and can flawlessly continue to work
                      print("Event processing failed: " + str(e))


