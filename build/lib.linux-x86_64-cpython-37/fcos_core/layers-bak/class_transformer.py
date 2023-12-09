# import torch
# import torch.nn as nn
# import numpy as np
#
# class dot_attention(nn.Module):
#
#     def __init__(self, attention_dropout=0.0):
#         super(dot_attention, self).__init__()
#         self.dropout = nn.Dropout(attention_dropout)
#         self.softmax = nn.Softmax(dim=2)
#
#     def forward(self, q, k, v, scale=None, attn_mask=None):
#         attention = torch.bmm(q, k.transpose(1, 2))
#         if scale:
#             attention = attention * scale
#         if attn_mask:
#             attention = attention.masked_fill(attn_mask, -np.inf)
#         attention = self.softmax(attention)
#         attention = self.dropout(attention)
#         context = torch.bmm(attention, v)
#         return context
#
# class class_MultiHeadAttention(nn.Module):
#     def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
#         super(class_MultiHeadAttention, self).__init__()
#
#         self.dim_per_head = model_dim//num_heads
#         self.num_heads = num_heads
#         self.linear_k = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])
#         self.linear_v = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])
#         self.linear_q = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])
#
#         self.dot_product_attention = dot_attention(dropout)
#
#         self.linear_final = nn.Linear(model_dim, model_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(model_dim)
#         self.version  = version
#
#     def forward(self, key, value, query, label, attn_mask=None):
#
#         if self.version == 'v2':
#             label_lbl_first = []
#             nodes_lbl_first = []
#             context_lbl_first = []
#             label = label.long()
#             for lbl in label.unique():
#                 lbl_idx = label == lbl
#                 B =1
#                 key_lbl = key[lbl_idx].unsqueeze(1)
#                 value_lbl = value[lbl_idx].unsqueeze(1)
#                 query_lbl = query[lbl_idx].unsqueeze(1)
#                 residual = query_lbl
#                 dim_per_head = self.dim_per_head
#                 num_heads = self.num_heads
#                 key_lbl = self.linear_k[lbl](key_lbl)
#                 value_lbl = self.linear_v[lbl](value_lbl)
#                 query_lbl = self.linear_q[lbl](query_lbl)
#
#                 key_lbl = key_lbl.view(key_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)
#                 value_lbl = value_lbl.view(value_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)
#                 query_lbl = query_lbl.view(query_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)
#
#                 scale = (key_lbl.size(-1) // num_heads) ** -0.5
#                 context = self.dot_product_attention(query_lbl, key_lbl, value_lbl, scale, attn_mask)
#                 # (query, key, value, scale, attn_mask)
#                 context = context.transpose(0, 1).contiguous().view(query_lbl.size(1), B, dim_per_head * num_heads)
#                 output = self.linear_final(context)
#                 # dropout
#                 output = self.dropout(output)
#                 output = self.layer_norm(residual + output)
#                 label_lbl_first.append(label[lbl_idx])
#                 nodes_lbl_first.append(output.squeeze())
#                     # output = residual + output
#
#         elif self.version == 'v1': # some difference about the place of torch.view fuction
#             key = key.unsqueeze(0)
#             value = value.unsqueeze(0)
#             query = query.unsqueeze(0)
#             residual = query
#             B, L, C = key.size()
#             dim_per_head = self.dim_per_head
#             num_heads = self.num_heads
#             batch_size = key.size(0)
#
#             key = self.linear_k(key)
#             value = self.linear_v(value)
#             query = self.linear_q(query)
#
#             key = key.view(batch_size * num_heads, -1, dim_per_head)
#             value = value.view(batch_size * num_heads, -1, dim_per_head)
#             query = query.view(batch_size * num_heads, -1, dim_per_head)
#
#             if attn_mask:
#                 attn_mask = attn_mask.repeat(num_heads, 1, 1)
#             scale = (key.size(-1) // num_heads) ** -0.5
#             context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
#             context = context.view(batch_size, -1, dim_per_head * num_heads)
#             output = self.linear_final(context)
#             output = self.dropout(output)
#             output = self.layer_norm(residual + output)
#
#         return torch.cat(nodes_lbl_first, dim=0), torch.cat(context_lbl_first, dim=0), torch.cat(label_lbl_first, dim=0)
#
# class CrossGraph(nn.Module):
#     """ This class hasn't been used"""
#     def __init__(self, model_dim=256,  dropout=0.0,):
#         super(CrossGraph, self).__init__()
#
#
#         self.linear_node1 = nn.Linear(model_dim,model_dim)
#         self.linear_node2 = nn.Linear(model_dim,model_dim)
#
#         self.dot_product_attention = dot_attention(dropout)
#
#         self.linear_final = nn.Linear(model_dim, model_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(model_dim)
#
#
#     def forward(self, node_1, node_2,  attn_mask=None):
#         node_1_r = node_1
#         node_2_r = node_2
#
#         edge1 = self.linear_edge(node_1)
#         edge2 = self.linear_edge(node_2)
#
#         node_1_ = self.linear_node1(node_1)
#         node_2_ = self.linear_node1(node_2)
#
#         attention = torch.mm(edge1,edge2.t())
#
#         node_1 = torch.mm(attention.softmax(-1), node_2_)
#         node_2 = torch.mm(attention.t().softmax(-1), node_1_)
#
#
#         node_1 = self.linear_final(node_1)
#         node_2 = self.linear_final(node_2)
#
#         node_1 = self.dropout(node_1)
#         node_2  = self.dropout(node_2)
#         node_1 = self.layer_norm(node_1_r + node_1)
#
#         node_2 = self.layer_norm(node_2_r + node_2)
#
#
#         return node_1, node_2
#
# if __name__ == "__main__":
#  node_1 = torch.rand(241,256)
#  label = torch.randint(0,9,[241])
#  model = class_MultiHeadAttention(256, 1, dropout=0.1, version='v2')
#  print(model(node_1, node_1, node_1, label)[0],)
#  # a = nn.ModuleList([nn.Linear(256, 256) for i in range(8)])
import torch
import torch.nn as nn
import numpy as np

class dot_attention(nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context

class class_self_MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
        super(class_self_MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])
        self.linear_v = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])
        self.linear_q = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.version  = version

    def forward(self, key, label, attn_mask=None):
        key_ori = key.clone()
        if self.version == 'v2':
            label = label.long()
            for lbl in label.unique():
                lbl_idx = label == lbl
                B =1
                key_lbl = key[lbl_idx].unsqueeze(1)
                value_lbl = key[lbl_idx].unsqueeze(1)
                query_lbl = key[lbl_idx].unsqueeze(1)
                residual = query_lbl
                dim_per_head = self.dim_per_head
                num_heads = self.num_heads
                key_lbl = self.linear_k[lbl](key_lbl)
                value_lbl = self.linear_v[lbl](value_lbl)
                query_lbl = self.linear_q[lbl](query_lbl)

                key_lbl = key_lbl.view(key_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)
                value_lbl = value_lbl.view(value_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)
                query_lbl = query_lbl.view(query_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)

                scale = (key_lbl.size(-1) // num_heads) ** -0.5
                context = self.dot_product_attention(query_lbl, key_lbl, value_lbl, scale, attn_mask)
                # (query, key, value, scale, attn_mask)
                context = context.transpose(0, 1).contiguous().view(query_lbl.size(1), B, dim_per_head * num_heads)
                output = self.linear_final(context)
                # dropout
                output = self.dropout(output)
                output = self.layer_norm(residual + output)
                key_ori[lbl_idx] = output.squeeze()
                    # output = residual + output

        elif self.version == 'v1': # some difference about the place of torch.view fuction
            key = key.unsqueeze(0)
            value = key.unsqueeze(0)
            query = key.unsqueeze(0)
            residual = query
            B, L, C = key.size()
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)

            if attn_mask:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)
            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            context = context.view(batch_size, -1, dim_per_head * num_heads)
            output = self.linear_final(context)
            output = self.dropout(output)
            output = self.layer_norm(residual + output)

        return key_ori


class class_cross_MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
        super(class_cross_MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])
        self.linear_v = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])
        self.linear_q = nn.ModuleList([nn.Linear(model_dim, self.dim_per_head * num_heads) for num in range(9)])

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.version  = version

    def forward(self, key, value, query, label_1, label_2, attn_mask=None):
        query_ori = query.clone()
        if self.version == 'v2':
            label_1 = label_1.long()
            label_2 = label_2.long()
            all_label = torch.cat([label_1, label_2]).unique()
            for lbl in all_label:
                lbl_idx_1 = label_1 == lbl
                lbl_idx_2 = label_2 == lbl
                B =1
                key_lbl = key[lbl_idx_1].unsqueeze(1)
                value_lbl = value[lbl_idx_1].unsqueeze(1)
                query_lbl = query[lbl_idx_2].unsqueeze(1)
                residual = query_lbl
                dim_per_head = self.dim_per_head
                num_heads = self.num_heads
                key_lbl = self.linear_k[lbl](key_lbl)
                value_lbl = self.linear_v[lbl](value_lbl)
                query_lbl = self.linear_q[lbl](query_lbl)

                key_lbl = key_lbl.view(key_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)
                value_lbl = value_lbl.view(value_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)
                query_lbl = query_lbl.view(query_lbl.size(0), B * num_heads, dim_per_head).transpose(0,1)

                scale = (key_lbl.size(-1) // num_heads) ** -0.5
                context = self.dot_product_attention(query_lbl, key_lbl, value_lbl, scale, attn_mask)
                # (query, key, value, scale, attn_mask)
                context = context.transpose(0, 1).contiguous().view(query_lbl.size(1), B, dim_per_head * num_heads)
                output = self.linear_final(context)
                # dropout
                output = self.dropout(output)
                output = self.layer_norm(residual + output)
                query_ori[lbl_idx_2] = output.squeeze()
                    # output = residual + output

        elif self.version == 'v1': # some difference about the place of torch.view fuction
            key = key.unsqueeze(0)
            value = key.unsqueeze(0)
            query = key.unsqueeze(0)
            residual = query
            B, L, C = key.size()
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)

            if attn_mask:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)
            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            context = context.view(batch_size, -1, dim_per_head * num_heads)
            output = self.linear_final(context)
            output = self.dropout(output)
            output = self.layer_norm(residual + output)

        return query_ori

class CrossGraph(nn.Module):
    """ This class hasn't been used"""
    def __init__(self, model_dim=256,  dropout=0.0,):
        super(CrossGraph, self).__init__()


        self.linear_node1 = nn.Linear(model_dim,model_dim)
        self.linear_node2 = nn.Linear(model_dim,model_dim)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)


    def forward(self, node_1, node_2,  attn_mask=None):
        node_1_r = node_1
        node_2_r = node_2

        edge1 = self.linear_edge(node_1)
        edge2 = self.linear_edge(node_2)

        node_1_ = self.linear_node1(node_1)
        node_2_ = self.linear_node1(node_2)

        attention = torch.mm(edge1,edge2.t())

        node_1 = torch.mm(attention.softmax(-1), node_2_)
        node_2 = torch.mm(attention.t().softmax(-1), node_1_)


        node_1 = self.linear_final(node_1)
        node_2 = self.linear_final(node_2)

        node_1 = self.dropout(node_1)
        node_2  = self.dropout(node_2)
        node_1 = self.layer_norm(node_1_r + node_1)

        node_2 = self.layer_norm(node_2_r + node_2)


        return node_1, node_2

if __name__ == "__main__":
    node_1 = torch.rand(241,256)
    label_1 = torch.randint(0,9,[241])
    node_2 = torch.rand(211,256)
    label_2 = torch.randint(0,9,[211])
    model = class_cross_MultiHeadAttention(256, 1, dropout=0.1, version='v2')


 # a = nn.ModuleList([nn.Linear(256, 256) for i in range(8)])