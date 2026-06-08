import sys
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class APDataset:
    n_nodes: int
    coords: List[Tuple[float, float]]
    flow_matrix: List[List[float]]
    p_hubs: int
    collection_cost: float
    transfer_cost: float
    distribution_cost: float

def parse_ap_dataset(filepath: str) -> APDataset:
    """Parses an AP Dataset file and returns a strongly-typed APDataset dataclass."""
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Using split() without arguments splits by any whitespace (spaces, tabs, newlines)
    # This is robust against inconsistent formatting from the C scripts.
    tokens = content.split()
    
    if not tokens:
        raise ValueError(f"File {filepath} is empty or unreadable.")
        
    # Use an iterator to sequentially consume tokens
    token_iter = iter(tokens)
    
    def next_float():
        return float(next(token_iter))
    
    def next_int():
        return int(next(token_iter))
        
    try:
        n_nodes = next_int()
        
        coords = []
        for _ in range(n_nodes):
            x = next_float()
            y = next_float()
            coords.append((x, y))
            
        flow_matrix = []
        for _ in range(n_nodes):
            row = [next_float() for _ in range(n_nodes)]
            flow_matrix.append(row)
            
        p_hubs = next_int()
        
        collection_cost = next_float()
        transfer_cost = next_float()
        distribution_cost = next_float()
        
    except StopIteration:
        raise ValueError("Unexpected end of file while parsing dataset.")
        
    return APDataset(
        n_nodes=n_nodes,
        coords=coords,
        flow_matrix=flow_matrix,
        p_hubs=p_hubs,
        collection_cost=collection_cost,
        transfer_cost=transfer_cost,
        distribution_cost=distribution_cost
    )

if __name__ == '__main__':
    # Test on a known dataset
    test_file = 'AP Dataset/APdata200'
    try:
        dataset = parse_ap_dataset(test_file)
        print(f"Successfully parsed: {test_file}")
        print(f"Number of nodes: {dataset.n_nodes}")
        print(f"Number of hubs (p): {dataset.p_hubs}")
        print(f"Coords (first 3): {dataset.coords[:3]}...")
        print(f"Flow Matrix dimension: {len(dataset.flow_matrix)}x{len(dataset.flow_matrix[0])}")
        print(f"Collection Cost: {dataset.collection_cost}")
        print(f"Transfer Cost: {dataset.transfer_cost}")
        print(f"Distribution Cost: {dataset.distribution_cost}")
    except Exception as e:
        print(f"Error parsing {test_file}: {e}")
        sys.exit(1)
