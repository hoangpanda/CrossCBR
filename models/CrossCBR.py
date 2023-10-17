#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GraphConv
from GraphGAT import GraphGAT


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    # rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    # colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    # graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats


class GAT(nn.Module): 
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 4
        self.in_head = 4
        self.out_head = 1
        self.embedding_input_size = 64
        self.embedding_output_size = 64
        self.conv1 = GATv2Conv(self.embedding_input_size, self.embedding_output_size, heads=1)
        self.conv2 = GATConv(self.hid*self.in_head, self.embedding_output_size, concat=False, heads=self.out_head)
        self.GCNconv1 = GCNConv(self.embedding_input_size, self.embedding_output_size)
        self.GraphGCN_conv1 = GraphConv(self.embedding_input_size, sle.embedding_output_size)

    def forward(self, features, graph):
        x, edge_index = features, graph._indices()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.GraphGCN_conv1(x, edge_index)
        return x;
        #x = F.relu(x)
        #x = F.dropout(x, p=0.3, training=self.training)
        #x = self.conv2(x, edge_index)
        #return F.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, dim_id, dim_latent=None):
        super(GNN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features

        self.preference = nn.Embedding(num_user, self.dim_latent)
        nn.init.xavier_normal_(self.preference.weight).cuda()
        if self.dim_latent:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)

            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = GraphGAT(self.dim_feat, self.dim_feat, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, aggr='add')
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
        nn.init.xavier_normal_(self.g_layer2.weight)

    def forward(self, id_embedding):
        temp_features = torch.tanh(self.MLP(self.features)) if self.dim_latent else self.features
        x = torch.cat((self.preference.weight, temp_features), dim=0)
        x = F.normalize(x).cuda()

        #layer-1
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
        x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
        return x_1
        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
        x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        x = torch.cat((x_1, x_2), dim=1)

        return x

class CrossCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]


    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
        print(f'self.item_level_graph: {self.item_level_graph}')


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph # user item
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])

        # normalize layer LightGCN
        #print(f'BEFORE: {item_level_graph}')

        #x = laplace_transform(item_level_graph)
        # print(f'LAPLACE TRANSFORM: {x}')
        # print(f'shape of LAPLACE TRANSFORM: {x.shape}')
        # print(f'type of LAPLACE TRANSFORM: {type(x)}')

        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)
        

    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        #print(f'ub_graph: {ub_graph}')
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        #print(f'bundle_level_graph: {bundle_level_graph}')

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)
        #print(f'bundle_level_graph: {bundle_level_graph}')


    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)




    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        print(f'graph: {graph}')
        graph_indices = graph._indices()
        print(f'graph indices: {graph_indices}   ')
        print(f'A_feature shape: {A_feature.shape}')
        print(f'A_feature: {A_feature}')
        print(f'B_feature shape: {B_feature.shape}')
        print(f'B_feature: {B_feature}')

        #layer = GATLayer(64, 64, num_heads=1)

        #indices = graph._indices()
        #A_feature = torch.unsqueeze(A_feature, 0)
        #B_feature = torch.unsqueeze(B_feature, 0)
        #max_v = 0
        # for i in range(indices.shape[0]):
        #     for j in range(indices.shape[1]):
        #         if indices[i][j] > max_v:
                    # max_v = indices[i][j]
        #max_v = 40807
        # indices_temp = torch.zeros((max_v, max_v))

        # row_1 = indices[0]
        # row_2 = indices[1]

        # for i in range(max_v):
            # indices_temp[indices[row_1][i]][indices[row_2][i]] = 1

        #indices = torch.unsqueeze(indices_temp, 0)
        # print(f'indices: {indices}')
        # print(f'max: {max_v}')
        #A_feature = layer(A_feature.to('cpu'), indices.to('cpu'), print_attn_probs=True)
        #B_feature = layer(B_feature.to('cpu'), indices.to('cpu'), print_attn_probs=True)
        features = torch.cat((A_feature, B_feature), 0).to('cpu')
        print(f'device features: {features.device}')
        all_features = [features]
        print(f'all_features: {all_features}')

        print(f'shape all_features: {features.shape}')
        for i in range(self.num_layers):
            # spmm <=> torch.sparse.mm -> multiply two matrix
            #gnn = GNN(64, graph._indices().to('cpu'))
            #features = torch.spmm(graph.to('cpu'), features)
            embedding_input = 64
            embedding_output= 64
            layerGAT = GAT().to('cpu')
            #print(f'device layerGAT: {layerGAT.device}')
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            #features = features / (i+2)
            features = layerGAT(features, graph.to('cpu'))
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        A_feature = A_feature.to('cuda:0')
        B_feature = B_feature.to('cuda:0')
        print(f'device A_feature: {A_feature.device}')
        print(f'device B_feature: {B_feature.device}')

        return A_feature, B_feature



    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, test=False):
        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            # item_level_graph: ui_matrix
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)
            print(f'users_feature: {self.users_feature}')
            print(f'users_feature shape: {self.users_feature.shape}')
            print(f'items_feature: {self.items_feature}')
            print(f'items_feature shape: {self.items_feature.shape}')
            print(f'IL_users_feature: {IL_users_feature}')
            print(f'IL_users_feature shape: {IL_users_feature.shape}')
            print(f'IL_items_feature: {IL_items_feature}')
            print(f'IL_items_feature: {IL_items_feature.shape}')

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)
        print(f'IL_items_feature: {IL_bundles_feature}')

        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)

        users_feature = [IL_users_feature, BL_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]

        #print(f'user_feature shape: {users_feature.shape}')
        #print(f'type of users_feature: {type(users_feature)}')
        #print(f'users_feature: {users_feature}')

        #print(f'bundles_feature shape: {bundles_feature.shape}')
        # print(f'type of bundles_feature: {type(bundles_feature)}')
        # print(f'bundles_feature: {bundles_feature}')

        return users_feature, bundles_feature


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature, alpha_c_loss=0.5):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature
        # [bs, 1+neg_num]
        alpha_pred = 1.5
        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2)*alpha_pred + torch.sum(BL_users_feature * BL_bundles_feature, 2)*(2-alpha_pred)
        bpr_loss = cal_bpr_loss(pred)   
        
        # cl is abbr. of "contrastive loss"
        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)

        c_losses = [u_cross_view_cl, b_cross_view_cl]
        
       # c_loss = u_cross_view_cl*alpha_c_loss + b_cross_view_cl*(1-alpha_c_loss)
       # c_loss = c_loss*0.5
        alpha_c_loss = 0.8
        c_loss = (u_cross_view_cl*alpha_c_loss + (1-alpha_c_loss)*b_cross_view_cl) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
    #    print(f'BATCH: {batch}')
        print('BAT DAU CHAY HAM SELF.PROPAGATE()')
        users_feature, bundles_feature = self.propagate()

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores
