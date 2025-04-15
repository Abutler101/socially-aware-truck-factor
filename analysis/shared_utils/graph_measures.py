from typing import Dict

import networkx as nx


def weighted_in_deg_cent(g: nx.DiGraph, weight: str) -> Dict[str, float]:
    total_nodes = len(g)
    weighted_in_degrees = list(g.in_degree(weight=weight))
    return {file: in_deg / total_nodes for file, in_deg in weighted_in_degrees}


def weighted_out_deg_cent(g: nx.DiGraph, weight: str) -> Dict[str, float]:
    total_nodes = len(g)
    weighted_out_degrees = list(g.out_degree(weight=weight))
    return {file: in_deg / total_nodes for file, in_deg in weighted_out_degrees}


def weighted_deg_cent(g: nx.DiGraph, weight: str) -> Dict[str, float]:
    total_nodes = len(g)
    weighted_degrees = list(g.degree(weight=weight))
    return {file: in_deg / total_nodes for file, in_deg in weighted_degrees}
