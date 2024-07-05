import torch.nn as nn
from ops import GMP, Unpool, WeightedEdgeConv

class BSGMP(nn.Module):
    """Bistride Graph Message Passing (BSGMP) network for hierarchical graph processing."""

    def __init__(self, unet_depth, latent_dim, hidden_layer, pos_dim):
        """
        Initialize the BSGMP network.

        Parameters
        ----------
        unet_depth : int
            Number of unet depth in the network.
            NOTE this excepts the top level
        latent_dim : int
            Latent dimension for the graph nodes and edges.
        hidden_layer : int
            Number of hidden layers in the MLPs.
        pos_dim : int
            Dimension of the physical position (in Euclidean space).
        """
        super(BSGMP, self).__init__()
        self.bottom_gmp = GMP(latent_dim, hidden_layer, pos_dim)
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.unet_depth = unet_depth
        self.edge_conv = WeightedEdgeConv()
        for _ in range(self.unet_depth):
            self.down_gmps.append(GMP(latent_dim, hidden_layer, pos_dim))
            self.up_gmps.append(GMP(latent_dim, hidden_layer, pos_dim))
            self.unpools.append(Unpool())

    def forward(self, h, m_ids, m_gs, pos):
        """
        Forward pass for the BSGMP network.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape [B, N, F] or [N, F].
        m_ids : list of torch.Tensor
            Indices for pooling/unpooling nodes at each level.
        m_gs : list of torch.Tensor
            Graph connectivity (edges) at each level.
        pos : torch.Tensor
            Node positional information of shape [B, N, D] or [N, D].

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        # Shape: h is in (B, N, F) or (N, F)
        # m_gs is in shape: Level,(Set),2,Edges, where 0th Set is main/material graph
        # pos is in (B, N, D) or (N, D)

        down_outs = []  # to store output features at each level during down pass
        down_ps = []  # to store positional information at each level during down pass
        cts = []  # to store edge weights for convolution at each level

        w = pos.new_ones((pos.shape[-2], 1))  # Initialize weights

        # Down pass
        for i in range(self.unet_depth):
            h = self.down_gmps[i](h, m_gs[i], pos)
            down_outs.append(h)
            down_ps.append(pos)

            # Calculate edge weights
            ew, w = self.edge_conv.cal_ew(w, m_gs[i])
            h = self.edge_conv(h, m_gs[i], ew)
            pos = self.edge_conv(pos, m_gs[i], ew)
            cts.append(ew)

            # Pooling
            if len(h.shape) == 3:
                h = h[:, m_ids[i]]
            elif len(h.shape) == 2:
                h = h[m_ids[i]]

            if len(pos.shape) == 3:
                pos = pos[:, m_ids[i]]
            elif len(pos.shape) == 2:
                pos = pos[m_ids[i]]

            w = w[m_ids[i]]

        # Bottom pass
        h = self.bottom_gmp(h, m_gs[self.unet_depth], pos)

        # Up pass
        for i in range(self.unet_depth):
            depth_idx = self.unet_depth - i - 1
            g, idx = m_gs[depth_idx], m_ids[depth_idx]
            h = self.unpools[i](h, down_outs[depth_idx].shape[-2], idx)
            # aggregate is False as we are returning the information to previous out degrees.
            h = self.edge_conv(h, g, cts[depth_idx], aggragating=False)
            h = self.up_gmps[i](h, g, down_ps[depth_idx])
            h = h.add(down_outs[depth_idx])

        return h
