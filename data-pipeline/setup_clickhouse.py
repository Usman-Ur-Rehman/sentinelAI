from clickhouse_driver import Client

client = Client('localhost', user='admin', password='password123', database='threat_db')

client.execute('''
    CREATE TABLE IF NOT EXISTS threat_events (
        timestamp DateTime,
        src_ip String,
        dst_ip String,
        src_port UInt16,
        dst_port UInt16,
        protocol String,
        packet_size UInt32,
        duration Float32,
        num_failed_logins UInt16,
        attack_type String,
        label String
    ) ENGINE = MergeTree()
    ORDER BY timestamp
''')

print('threat_events table created')