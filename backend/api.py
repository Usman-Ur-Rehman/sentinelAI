from fastapi import FastAPI
from clickhouse_driver import Client
from fastapi.middleware.cors import CORSMiddleware

#creating instance of fastAPI as app
app=FastAPI(title='sentinelAI')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#creating
click_house=Client('clickhouse',user='admin',password='password123',database='threat_db')

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

# Week 2 will add: /score endpoint for ML inference
# Week 3 will add: /agent endpoint for RAG + LangGraph

