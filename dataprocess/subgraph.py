import numpy as np
from torch import index_put_

'''
remove subgraph_edges
'''
def get_subgraph_edge(psg,graph_matrix,subgraph_num=8,max_subgraph_node=128):
    '''
    Args:
        psg: [list] a list consisting of token_ids representing the original long passage
        graph_matrix: [numpy.ndarray] the adjacency matrix to seperate
        subgraph_num: [int] the number of subgraphs you want
        max_subgraph_node:[int] the max node number of a subgraph
    Return:
        a list consisting of subgraphs, and each subgraph is represented by a list like the input psg
    '''
    node_num=graph_matrix.shape[0]
    subgraph_list = []  

    # the passage is too short to split into the given number of subgraphs
    if subgraph_num>node_num:
        for i in range(node_num):
            subgraph_list.append(psg[i])
        for i in range(subgraph_num-1):
            subgraph_list.append([])
        return subgraph_list

    # print sparsity of the original graph
    print("sparsity：",np.sum(graph_matrix)/node_num/node_num)

    # remove self_loop
    for i in range(node_num):
        graph_matrix[i][i]=0
    
    # extract subgraphs
    for _ in range(subgraph_num): 
        # no available node left
        if np.sum(graph_matrix)==0:
            subgraph_list.append([])
            continue    

        degree=[np.sum(graph_matrix[i]) for i in range(node_num)]

        # get node with the biggest degree
        cur_node=degree.index(max(degree))

        # get all the neighboring nodes of cur_node
        subgraph=[cur_node]
        for i in range(node_num):
            if graph_matrix[cur_node][i]==1 and i!=cur_node:
                subgraph.append(i)
                if len(subgraph)>=max_subgraph_node-1: break

        # delete the select subgraph from original graph
        for i in range(len(subgraph)):
            for j in range(i,len(subgraph)):
                u,v=subgraph[i],subgraph[j]
                if graph_matrix[u][v]==1:
                    # delete the edge from the original graph
                    graph_matrix[u][v]=0
                    graph_matrix[v][u]=0

        # convert the index of node into corresponding token_id 
        subgraph_list.append([psg[idx] for idx in subgraph])
    return subgraph_list

'''
remove subgraph_node
'''
def get_subgraph_node(psg,graph_matrix,subgraph_num=8,max_subgraph_node=128):
    '''
    Args:
        psg: [list] a list consisting of token_ids representing the original long passage
        graph_matrix: [numpy.ndarray] the adjacency matrix to seperate
        subgraph_num: [int] the number of subgraphs you want
        max_subgraph_node:[int] the max node number of a subgraph
    Return:
        a list consisting of subgraphs, and each subgraph is represented by a list like the input psg
    '''
    node_num=graph_matrix.shape[0]
    subgraph_list = []  

    # the passage is too short to split into the given number of subgraphs
    if subgraph_num>node_num:
        for i in range(node_num):
            subgraph_list.append(psg[i])
        for i in range(subgraph_num-1):
            subgraph_list.append([])
        return subgraph_list

    # print sparsity of the original graph
    print("sparsity：",np.sum(graph_matrix)/node_num/node_num)

    # remove self_loop
    for i in range(node_num):
        graph_matrix[i][i]=0

    degree=[np.sum(graph_matrix[i]) for i in range(node_num)]

    # extract subgraphs
    for _ in range(subgraph_num):   
        # no available node left
        if np.sum(graph_matrix)==0:
            subgraph_list.append([])
            continue    

        degree=[np.sum(graph_matrix[i]) for i in range(node_num)]

        # get node with the biggest degree
        cur_node=degree.index(max(degree))
        subgraph=[cur_node]
        for i in range(node_num):
            if graph_matrix[cur_node][i]==1 and i!=cur_node:   
                if len(subgraph)<max_subgraph_node:            
                    # delete the node from the original graph
                    graph_matrix[:,i]=0
                    graph_matrix[i,:]=0           
                    subgraph.append(i)
            
        subgraph_list.append([psg[idx] for idx in subgraph])

    return subgraph_list



