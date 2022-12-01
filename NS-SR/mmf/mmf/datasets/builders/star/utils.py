import torch
import numpy as np

def absolute_postion_embedding(token_ids,embeding_dim):
    try:
        token_ids = token_ids.tolist()
    except:
        pass

    base=10000
    i_max=int((embeding_dim-1)/2)
    all_embeddings_sin, all_embeddings_cos = [], []
    
    for pos in token_ids:
        posithon_embeding_sin, posithon_embeding_cos = [], []
        for i in range(i_max+1):
            posithon_embeding_sin.append(np.sin(pos/(pow(base,(2*i)/embeding_dim))))
            posithon_embeding_cos.append(np.cos(pos/(pow(base,(2*i)/embeding_dim))))
        all_embeddings_sin.append(posithon_embeding_sin)
        all_embeddings_cos.append(posithon_embeding_cos)

    sin_embedding = torch.tensor(all_embeddings_sin,dtype=torch.float32)
    cos_embedding = torch.tensor(all_embeddings_cos,dtype=torch.float32)
    all_embeddings = torch.cat((sin_embedding, cos_embedding), dim=-1)

    return all_embeddings

# def init_hyper_token(max_act, max_rel, max_situation):
#     # 0: Hyperedge (action) 1: node (object) / edges (relationship)
#     hyper_tokens = []
#     hyperedge = [1 for i in range(max_act)]
#     graph = [0 for i in range(max_rel*4+1)]
#     hypergraph = hyperedge + graph
#     for j in range(max_situation):
#         hyper_tokens += hypergraph

#     return hyper_tokens

# def init_type_token(max_act, max_rel, max_situation):
#     # 0: Action, 1: Object, 2: Relationship, 3: Special, 4: Frame
#     type_tokens = []
#     hyperedge = [0 for i in range(max_act)]
#     triplet = [3, 1, 2, 1] # [[SEP], OBJ1, REL, OBJ2]
#     for i in range(max_rel):
#         hyperedge += triplet
#     hypergraph = hyperedge + [4]
#     for j in range(max_situation):
#         type_tokens += hypergraph

#     return type_tokens

# def init_triplet_token(max_act, max_rel, max_situation):

#     triplet_tokens = []
#     max_id = max_rel + 2
#     hypergraph = [ 0 for i in range(max_act)]
#     for i in range(max_rel):
#         hypergraph += [max_id, i+1,i+1,i+1]
#     hypergraph += [max_rel + 1] # frame
#     for j in range(max_situation):
#         triplet_tokens += hypergraph

#     return triplet_tokens

# def init_situation_token(max_act, max_rel, max_situation):

#     situation_tokens = []
#     for i in range(max_situation):
#         situation_tokens += [ i for j in range(max_act + max_rel*4 + 1)]

#     return situation_tokens

# def init_special_token(max_rel, max_frame):
#     special_id_ = [2 for i in range(max_rel)]
#     special_target_ = [-1 for i in range(max_rel)]

#     special_id = torch.tensor([special_id_ for i in range(max_frame)], dtype=torch.long)
#     special_target = torch.tensor([special_target_ for j in range(max_frame)])

#     return special_id, special_target

def init_hyper_token(max_act, max_rel, max_situation):
    # 0: Hyperedge (action) 1: node (object) / edges (relationship)
    hyper_tokens = []
    hyperedge = [1 for i in range(max_act)]
    graph = [0 for i in range(max_rel*3)]
    hypergraph = hyperedge + graph
    for j in range(max_situation):
        hyper_tokens += hypergraph

    return hyper_tokens

def init_type_token(max_act, max_rel, max_situation):
    # 0: Action, 1: Object, 2: Relationship, 3: Special, 4: Frame
    type_tokens = []
    hyperedge = [0 for i in range(max_act)]
    triplet = [1, 2, 1] # [[SEP], OBJ1, REL, OBJ2]
    for i in range(max_rel):
        hyperedge += triplet
    hypergraph = hyperedge
    for j in range(max_situation):
        type_tokens += hypergraph

    return type_tokens

def init_triplet_token(max_act, max_rel, max_situation):

    triplet_tokens = []
    max_id = max_rel + 2
    hypergraph = [ 0 for i in range(max_act)]
    for i in range(max_rel):
        hypergraph += [i+1,i+1,i+1]
    #hypergraph += [max_rel + 1] # frame
    for j in range(max_situation):
        triplet_tokens += hypergraph

    return triplet_tokens

def init_situation_token(max_act, max_rel, max_situation):

    situation_tokens = []
    for i in range(max_situation):
        situation_tokens += [ i for j in range(max_act + max_rel*3)]

    return situation_tokens

def init_special_token(max_rel, max_frame):
    special_id_ = [2 for i in range(max_rel)]
    special_target_ = [-1 for i in range(max_rel)]

    special_id = torch.tensor([special_id_ for i in range(max_frame)], dtype=torch.long)
    special_target = torch.tensor([special_target_ for j in range(max_frame)])

    return special_id, special_target


def init_hyper_token_nr(max_act, max_rel, max_situation):
    # 0: Hyperedge (action) 1: node (object) / edges (relationship)
    hyper_tokens = []
    hyperedge = [1 for i in range(max_act)]
    graph = [0 for i in range(max_rel*3)]
    hypergraph = hyperedge + graph
    for j in range(max_situation):
        hyper_tokens += hypergraph

    return hyper_tokens

def init_type_token_nr(max_act, max_rel, max_situation):
    # 0: Action, 1: Object, 2: Relationship, 3: Special, 4: Frame
    type_tokens = []
    hyperedge = [0 for i in range(max_act)]
    triplet = [1, 2, 1] # [OBJ1, REL, OBJ2]
    for i in range(max_rel):
        hyperedge += triplet
    hypergraph = hyperedge
    for j in range(max_situation):
        type_tokens += hypergraph

    return type_tokens

def init_triplet_token_nr(max_act, max_rel, max_situation):

    triplet_tokens = []
    max_id = max_rel + 2
    hypergraph = [ 0 for i in range(max_act)]
    for i in range(max_rel):
        hypergraph += [i+1,i+1,i+1]
    #hypergraph += [max_rel + 1] # frame
    for j in range(max_situation):
        triplet_tokens += hypergraph

    return triplet_tokens

def init_situation_token_nr(max_act, max_rel, max_situation):

    situation_tokens = []
    for i in range(max_situation):
        situation_tokens += [ i for j in range(max_act + max_rel*3)]

    return situation_tokens

    