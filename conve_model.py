import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_



class BaseModel(torch.nn.Module):
    def __init__(self, pre_data):
        super(BaseModel, self).__init__()
        self.pre_data = pre_data
        self.ent_embed = torch.nn.Embedding(len(self.pre_data.ent2id), 200, padding_idx=None)
        xavier_normal_(self.ent_embed.weight.data)
        self.rel_embed = torch.nn.Embedding(len(self.pre_data.rel2id), 200, padding_idx=None)
        xavier_normal_(self.rel_embed.weight.data)
        self.bceloss = torch.nn.BCELoss()

    def concat(self, e1_embed, rel_embed, form='plain'):
        if form == 'plain':
            e1_embed = e1_embed.view(-1, 1, 10, 20)
            rel_embed = rel_embed.view(-1, 1, 10, 20)
            stack_inp = torch.cat([e1_embed, rel_embed], 2)

        elif form == 'alternate':
            e1_embed = e1_embed.view(-1, 1, 200)
            rel_embed = rel_embed.view(-1, 1, 200)
            stack_inp = torch.cat([e1_embed, rel_embed], 1)
            stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * 10, 20))

        else:
            raise NotImplementedError
        return stack_inp

    def loss(self, pred, true_label=None):
        pos_scr = pred[:, 0]
        neg_scr = pred[:, 1:]
        label_pos = true_label[0]
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss


class ConvE(BaseModel):
    def __init__(self, pre_data):
        super(ConvE, self).__init__(pre_data)
        inp_drop = 0.2
        feat_drop = 0.3
        hid_drop = 0.3
        k_w = 10
        k_h = 20
        num_filt = 32
        embed_dim = k_w * k_h
        ker_sz = 3
        bias = False
        self.input_drop = torch.nn.Dropout(inp_drop)
        self.feature_drop = torch.nn.Dropout2d(feat_drop)
        self.hidden_drop = torch.nn.Dropout(hid_drop)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(num_filt)
        self.bn2 = torch.nn.BatchNorm1d(embed_dim)

        self.conv1 = torch.nn.Conv2d(1, out_channels=num_filt, kernel_size=(ker_sz, ker_sz),
                                     stride=1, padding=0, bias=bias)
        flat_sz_h = int(2 * k_w) - ker_sz + 1
        flat_sz_w = k_h - ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * num_filt
        self.fc = torch.nn.Linear(self.flat_sz, embed_dim)

        self.register_parameter('bias', Parameter(torch.zeros(len(self.pre_data.ent2id))))

    def forward(self, sub, rel):
        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)
        stk_inp = self.concat(sub_emb, rel_emb, 'alternate')
        x = self.bn0(stk_inp)
        x = self.input_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
        x += self.bias.expand_as(x)

        pred = torch.sigmoid(x)

        return pred

    def predict(self, sub, rel, top_m):
        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)
        stk_inp = self.concat(sub_emb, rel_emb, 'alternate')
        x = self.bn0(stk_inp)
        x = self.input_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.ent_embed(top_m).transpose(1, 0))
        x += self.bias[top_m].expand_as(x)
        pred = torch.sigmoid(x)

        return pred