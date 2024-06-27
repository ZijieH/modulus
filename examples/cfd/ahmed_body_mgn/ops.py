import torch.nn as nn
import torch
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP


class GMP(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MeshGraphMLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer)
        edge_info_in_len =  2 * latent_dim + pos_dim + 1
        self.mlp_edge_info = MeshGraphMLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer)
        self.pos_dim = pos_dim

    def forward(self, x, g, pos):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            pi = pos[:, i]
            pj = pos[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim
        norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
        fiber = torch.cat([dir, norm], dim=-1)

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        edge_embedding = self.mlp_edge_info(tmp)

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x


class WeightedEdgeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', flow='target_to_source')

    def forward(self, x, g, ew, aggragating=True):
        # aggregating: False means returning
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter(weighted_info, target_index, dim=-2, dim_size=x.shape[-2], reduce="sum")
        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i = g[0]
        j = g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter(w_to_send, j, dim=-1, dim_size=normed_w.size(0), reduce="sum") + eps
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


class GMPEdgeAggregatedRes(MessagePassing):
    def __init__(self, in_dim, latent_dim, hidden_layer):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_edge_info = MeshGraphMLP(in_dim, latent_dim, latent_dim, hidden_layer)

    def forward(self, x, g, pos, pos_w, use_mat=True, use_world=True):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            if use_mat:
                pi = pos[:, i]
                pj = pos[:, j]
            if use_world:
                pwi = pos_w[:, i]
                pwj = pos_w[:, j]
        elif len(pos.shape) == 2:
            if use_mat:
                pi = pos[i]
                pj = pos[j]
            if use_world:
                pwi = pos_w[i]
                pwj = pos_w[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if use_mat:
            dir = pi - pj  # in shape (T),N,dim
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
        if use_world:
            dir_w = pwi - pwj  # in shape (T),N,dim
            norm_w = torch.norm(dir_w, dim=-1, keepdim=True)  # in shape (T),N,1

        if use_mat and use_world:
            fiber = torch.cat([dir, norm, dir_w, norm_w], dim=-1)
        elif not use_mat and use_world:
            fiber = torch.cat([dir_w, norm_w], dim=-1)
        elif use_mat and not use_world:
            fiber = torch.cat([dir, norm], dim=-1)
        else:
            raise NotImplementedError("at least one pos needs to cal fiber info")

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        edge_embedding = self.mlp_edge_info(tmp)

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        return aggr_out


class Unpool(nn.Module):
    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h
        return new_h
