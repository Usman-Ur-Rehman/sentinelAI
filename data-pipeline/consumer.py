import redis, json
from clickhouse_driver import Client
from datetime import datetime

r = redis.Redis(host='localhost', port=6379)
ch = Client('localhost', user='admin', password='password123', database='threat_db')

print('Consumer started. Waiting for events...')

last_id = '0'
while True:
    events = r.xread({'threat_stream': last_id}, count=10, block=1000)
    if events:
        for stream, messages in events:
            for msg_id, msg_data in messages:
                event = json.loads(msg_data[b'data'])
                ch.execute('INSERT INTO threat_events VALUES', [{
                    'timestamp': datetime.fromisoformat(event['timestamp']),
                    'src_ip': event['src_ip'], 'dst_ip': event['dst_ip'],
                    'src_port': event['src_port'], 'dst_port': event['dst_port'],
                    'protocol': event['protocol'], 'packet_size': event['packet_size'],
                    'duration': event['duration'],
                    'num_failed_logins': event['num_failed_logins'],
                    'attack_type': event['attack_type'], 'label': event['label']
                }])
                last_id = msg_id
                print(f"Stored: {event['label']} ({event['attack_type']})")