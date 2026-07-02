from typing import TypedDict, List, Optional
class AgentState(TypedDict):
    scored_event: dict # input from Week 2 scored_stream
    routing_decision: str # 'VALIDATION' or 'FULL_PIPELINE'
    rewritten_query: str # Person 2 output
    expanded_queries: List[str] # Person 3 output
    retrieved_chunks: List[dict] # Person 1 output (hybrid)
    multi_hop_chunks: List[dict] # Person 1 output (multi-hop)
    merged_ranked_chunks: List[dict] # Person 2 output
    reranked_chunks: List[dict] # Person 2 output
    compressed_context: str # Person 3 output
    validation_result: Optional[str] # 'CONFIRMED' or 'CONTRADICTED'
    llm_explanation: str # Person 2 output
    coordinated_attack_detected: bool # Person 2 output
    severity: str # Person 3 output
    action_taken: str # Person 3 output
    embedded_back: bool # Person 3 output
