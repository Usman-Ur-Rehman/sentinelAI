import redis
import random
import time
import json
from datetime import datetime
from faker import Faker

r =redis.Redis(host='localhost', port=6379)
fake=Faker()

def normal_event():
    return {
        'timestamp': datetime.now().isoformat(),
        'src_ip': fake.ipv4(),
        'dst_ip': fake.ipv4(),
        'src_port': random.randint(1024, 65535),
        'dst_port': random.choice([80, 443, 8080]),
        'protocol': random.choice(['TCP', 'UDP']),
        'packet_size': random.randint(64, 512),
        'duration': random.uniform(0.1, 2.0),
        'num_failed_logins': 0,
        'attack_type': 'none',
        'label': 'normal'
    }

def brute_force_event(src_ip):
    return {
        'timestamp': datetime.now().isoformat(),
        'src_ip': src_ip,
        'dst_ip': fake.ipv4(),
        'src_port': random.randint(1024, 65535),
        'dst_port': 22,
        'protocol': 'TCP',
        'packet_size': random.randint(1400, 1500),
        'duration': random.uniform(0.001, 0.01),
        'num_failed_logins': random.randint(3, 10),
        'attack_type': 'brute_force',
        'label': 'attack'
    }

def port_scan_event(src_ip, target_port):
    return {
        'timestamp': datetime.now().isoformat(),
        'src_ip': src_ip,
        'dst_ip': fake.ipv4(),
        'src_port': random.randint(1024, 65535),
        'dst_port': target_port,
        'protocol': 'TCP',
        'packet_size': random.randint(40, 60),
        'duration': random.uniform(0.0001, 0.001),
        'num_failed_logins': 0,
        'attack_type': 'port_scan',
        'label': 'attack'
    }

def exfiltration_event(src_ip):
    return {
        'timestamp': datetime.now().isoformat(),
        'src_ip': src_ip,
        'dst_ip': fake.ipv4(),
        'src_port': random.randint(1024, 65535),
        'dst_port': 443,
        'protocol': 'TCP',
        'packet_size': random.randint(8000, 10000),
        'duration': random.uniform(5.0, 15.0),
        'num_failed_logins': 0,
        'attack_type': 'exfiltration',
        'label': 'attack'
    }

attack_ip=fake.ipv4()

while (True):
    roll=random.random()

    if(roll<0.85):
        event=normal_event()
    elif(roll<0.92):
        event=brute_force_event(attack_ip)
    elif(roll<0.92):
        event=port_scan_event(attack_ip,random.randint(1, 1024))
    else:
        event=exfiltration_event(attack_ip)
    
    r.xadd('threat_stream',{'data':json.dumps(event)})

    