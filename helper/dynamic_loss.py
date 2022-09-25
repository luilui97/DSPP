import torch.nn as nn
import torch
class MultiLossLayer(nn.Module):
    """
           计算自适应损失权重
           implementation of "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
       """
    def __init__(self, num_loss):
        super(MultiLossLayer, self).__init__()
        self.sigmas_dota = nn.Parameter(nn.init.uniform_(torch.empty(num_loss), a=0.2, b=1.0), requires_grad=True)

    def get_loss(self, loss_set):
        factor = torch.div(1.0, torch.mul(2.0, self.sigmas_dota)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        loss_part = torch.sum(torch.mul(factor, loss_set)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        regular_part = torch.sum(torch.log(self.sigmas_dota)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        loss = loss_part + regular_part
        return loss