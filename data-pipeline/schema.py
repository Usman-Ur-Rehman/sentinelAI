EVENT_SCHEMA = {
    'timestamp': 'string (ISO 8601 format)', #YYYY MM DD : HH MM SS....
    'src_ip': 'string',
    'dst_ip': 'string',
    'src_port': 'int',
    'dst_port': 'int',
    'protocol': 'string (TCP | UDP | ICMP)',
    'packet_size': 'int (bytes)',
    'duration': 'float (seconds)',
    'num_failed_logins': 'int',
    'attack_type': 'string (none | brute_force | port_scan | exfiltration)',
    'label': 'string (normal | attack)'
}