#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GraphConv, TransformerConv
from torch_geometric.nn import SuperGATConv, SSGConv
 

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
    # chuẩn hóa 
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph 


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values

class GAT(nn.Module): 
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 12
        self.in_head = 12
        self.out_head = 1
        self.embedding_input_size = 64
        self.embedding_output_size = 64
        self.conv1 = GATv2Conv(self.embedding_input_size, self.in_head, heads=self.hid)
        self.conv2 = GATv2Conv(self.hid*self.in_head, self.embedding_output_size, heads=self.out_head)
        self.transformer_conv_1 = TransformerConv(self.embedding_input_size, self.in_head, heads=self.hid)
        self.transformer_conv_2 = TransformerConv(self.hid*self.in_head, self.embedding_output_size, heads=self.out_head)
        self.SuperGATConv_1 = SuperGATConv(self.embedding_input_size, self.in_head, heads=self.hid)
        self.SuperGATConv_2 = SuperGATConv(self.hid*self.in_head, self.embedding_output_size, heads=self.out_head)
        self.SSGConv = SSGConv(self.embedding_input_size, self.embedding_output_size, alpha=0.5)

    def forward(self, features, graph):
        x, edge_index = features, graph._indices()
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.SuperGATConv_1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.SuperGATConv_2(x, edge_index)
        x = self.SSGConv(x, edge_index)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x;
        #return F.log_softmax(x, dim=1)

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

        # khởi tạo embedding
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
        # edit layer
        self.GAT_model = GAT()
        self.GraphConv = GraphConv(64, 64)


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
        # print(f'self.item_level_graph: {self.item_level_graph}')


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph # user item
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])

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
        # print(f'graph: {graph}')
        graph_indices = graph._indices()
        # print(f'graph indices: {graph_indices}   ')
        # print(f'A_feature shape: {A_feature.shape}')
        # print(f'A_feature: {A_feature}')
        # print(f'B_feature shape: {B_feature.shape}')
        # print(f'B_feature: {B_feature}')

        features = torch.cat((A_feature, B_feature), 0).to('cpu') # user feature and bundle feature 
        all_features = [features] 
        for i in range(self.num_layers):
            #layerGAT = self.GAT_model.to('cpu')
            #features = layerGAT(features, graph.to('cpu'))
            #layerGraphConv = self.GraphConv().to('cpu')
            #features = layerGraphConv(features, graph.to('cpu')._indices())
            features = torch.spmm(graph.to('cpu'), features) # spmm <=> torch.sparse.mm -> multiply two matrix
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                # không test thì sẽ tạo data augmentation
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) 
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        A_feature = A_feature.to('cuda:0')
        B_feature = B_feature.to('cuda:0')

        return A_feature, B_feature

    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature) # (B x I) x (I x d) -> (B x d)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, test=False):
        # lightGCN with item view and bundle view
        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            # item_level_graph: ui_matrix
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)
            # print(f'users_feature: {self.users_feature}')
            # print(f'users_feature shape: {self.users_feature.shape}')
            # print(f'items_feature: {self.items_feature}')
            # print(f'items_feature shape: {self.items_feature.shape}')
            # print(f'IL_users_feature: {IL_users_feature}')
            # print(f'IL_users_feature shape: {IL_users_feature.shape}')
            # print(f'IL_items_feature: {IL_items_feature}')
            # print(f'IL_items_feature: {IL_items_feature.shape}')

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)
        # print(f'IL_items_feature: {IL_bundles_feature}')

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
        c_loss = u_cross_view_cl + b_cross_view_cl
        c_loss = c_loss*0.5
        #alpha_c_loss = 0.8
        #c_loss = (u_cross_view_cl*alpha_c_loss + (1-alpha_c_loss)*b_cross_view_cl) / len(c_losses)
        
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
        users_feature, bundles_feature = self.propagate()


        print(f'users_feature: {users_feature}')
        # print(f'shape of user_feature: {torch.tensor(users_feature).shape}')
        # print(f'shape of bundle_feature: {torch.tensor(bundles_feature).shape}')

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
