import numpy as np

# remove subgraph_edges
def get_subgraph_edge(psg,graph,n,k,max_subgraph_node):
    if k>n:
        ans=[psg[i] for i in range(n)]
        for i in range(k-1):ans.append([])
        return ans

    # print("sparsity：",np.sum(graph)/n/n)
    ans = []
    
    # remove self_loop
    for i in range(n):
        graph[i][i]=0
    
    # get k subgraph
    for _ in range(k): 
        if np.sum(graph)==0:
            ans.append([])
            continue    
        
        degree=[np.sum(graph[i]) for i in range(n)]

        # get node with the biggest degree
        cur_node=degree.index(max(degree))
        g=[cur_node]
        for j in range(n):
            if graph[cur_node][j]==1 and j!=cur_node:
                g.append(j)
                if len(g)>=max_subgraph_node-1:
                    break

        # clean the select subgraph in original graph
        for i in range(len(g)):
            for j in range(i,len(g)):
                u,v=g[i],g[j]
                if graph[u][v]==1:
                    graph[u][v]=0
                    graph[v][u]=0
            
        ans.append([psg[tt] for tt in g])
    return ans

# remove subgraph_node
def get_subgraph_node(psg,graph,n,k,max_subgraph_node):

    if k>n:
        ans=[psg[i] for i in range(n)]
        for i in range(k-1):ans.append([])
        return ans

    # print("sparsity：",np.sum(graph)/n/n)

    # remove self_loop
    for i in range(n):
        graph[i][i]=0

    degree=[np.sum(graph[i]) for i in range(n)]
    ans = []
    # print("------------------------------------------------")

    for _ in range(k):   
        if np.sum(graph)==0:
            ans.append([])
            continue    

        degree=[np.sum(graph[i]) for i in range(n)]

        cur_node=degree.index(max(degree))
        g=[cur_node]
        for j in range(n):
            if graph[cur_node][j]==1 and j!=cur_node:   
                if len(g)<max_subgraph_node:            
                    # delete the node from the original graph
                    graph[:,j]=0
                    graph[j,:]=0           
                    g.append(j)
            
        ans.append([psg[tt] for tt in g])

    return ans



