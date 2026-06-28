from fastapi import FastAPI
from clickhouse_driver import Client
from fastapi.middleware.cors import CORSMiddleware
import redis 
import json


#creating instance of fastAPI as app
app=FastAPI(title='sentinelAI')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#creating clickhouse client and redis connection
click_house=Client('clickhouse',user='admin',password='password123',database='threat_db')
r_client = redis.Redis(host='redis', port=6379)



#decorator for url='/health' 
@app.get('/health')
def healthCheck():
    return {'status':'ok'}



#decorator for url='/stats'
@app.get('/stats')
def getStats():
    #get total number of rows in table threat_events from clickHouse db
    total_rows=click_house.execute('SELECT count() from threat_events')[0][0]
    #get total number of rows with attack label(attacked ip rows count) in table threat_events from clickHouse db
    total_attack_rows=click_house.execute("SELECT count() from threat_events where label='attack' ")[0][0]

    return{#returing in json file format as frontend takes json file 
        'total_events':total_rows,
        'total_attacks':total_attack_rows,
        #AlertRate=how much percent of total events were attacked 
        'alert_rate':round(total_attack_rows/total_rows*100,2) if total_rows>0 else 0    
    }



 #decorator endpoint for ML inference
@app.get('/ml-stats')
def get_ml_stats():
    scored_count = r_client.xlen('scored_stream')
    drift_raw = r_client.get('drift_status')
    drift_status = json.loads(drift_raw) if drift_raw else {
        'avg_confidence': None,
        'drift_detected': False,
        'drift_reason': None,
        'drifted_features': [],
        'window_size': 0
    }
    return {
        'scored_events': scored_count,
        'drift': drift_status
    }
# Week 3 will add: /agent endpoint for RAG + LangGraph

