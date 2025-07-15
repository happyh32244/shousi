import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self,x):
        return self.fc(x)
    
class MoeLayer(nn.Module):
    def __init__(self, num_experts, in_feature, out_feature):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(in_feature,out_feature) for _ in range(num_experts)])
        self.gate = nn.Linear(in_feature, num_experts)

    def forward(self,x):
        gate_score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts],dim=1)
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        return output
    
if __name__ =='__main__':
    input_size = 5
    output_size = 3
    num_experts = 4
    batch_size = 10

    model = MoeLayer(num_experts,input_size, output_size)
    demo = torch.randn(batch_size, input_size)
    output = model(demo)
    print(output.shape)


