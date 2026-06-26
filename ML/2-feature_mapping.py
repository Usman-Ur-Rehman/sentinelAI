# Maps generator event fields to approximate CICIDS feature equivalents.
# Not a perfect 1-to-1 mapping since generator produces simplified events.
# Good enough for Week 2 scoring — Week 3 onwards use real CICIDS data.

CICIDS_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean',
    'Bwd Packet Length Mean', 'Flow IAT Mean',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count'
]

def generator_event_to_features(event: dict) -> list:
    '''
    Convert a generator event dict to a feature vector
    in the same order as CICIDS_FEATURES.
    Returns a list of floats ready for model.predict().
    '''
    duration = float(event.get('duration', 0))
    packet_size = float(event.get('packet_size', 0))
    num_failed = float(event.get('num_failed_logins', 0))
    dst_port = float(event.get('dst_port', 0))

    # Derive approximate CICIDS features from generator fields
    flow_duration = duration * 1_000_000        # convert seconds to microseconds
    total_fwd_pkts = packet_size / 100 if packet_size > 0 else 1
    total_bwd_pkts = total_fwd_pkts * 0.8
    flow_bytes_s = packet_size / duration if duration > 0 else 0
    flow_pkts_s = 1 / duration if duration > 0 else 0
    fwd_pkt_len_mean = packet_size
    bwd_pkt_len_mean = packet_size * 0.6
    flow_iat_mean = duration * 500_000
    syn_flag = 1.0 if dst_port == 22 else 0.0   # brute force targets SSH
    rst_flag = 1.0 if num_failed > 0 else 0.0
    psh_flag = 1.0 if packet_size > 1000 else 0.0
    ack_flag = 1.0 if duration > 0.1 else 0.0

    return [
        flow_duration, total_fwd_pkts, total_bwd_pkts,
        flow_bytes_s, flow_pkts_s, fwd_pkt_len_mean,
        bwd_pkt_len_mean, flow_iat_mean,
        syn_flag, rst_flag, psh_flag, ack_flag
    ]