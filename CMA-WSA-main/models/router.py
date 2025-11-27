import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from .modules import CrossAttention
from torch import softmax

# Add CAM module
class CAM(nn.Module):
    """Cross-Modal Attention Mechanism"""
    def __init__(self, C=8):
        super().__init__()
        # Define a learnable parameter gamma, initialized to 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.4)
        # Define the number of channels C
        self.C = C

    def forward(self, img_feat, text_feat):
        # Use raw features with shape (batch, feature_dim)
        B = img_feat.shape[0]  # batch size
        
        # Compute N1 and N2; if not divisible by C, discard the excess dimensions
        N1 = img_feat.shape[1] // self.C
        N2 = text_feat.shape[1] // self.C
        
        # Reshape to (batch, C, N)
        img_feat_reshaped = img_feat[:, :self.C*N1].view(B, self.C, N1)
        text_feat_reshaped = text_feat[:, :self.C*N2].view(B, self.C, N2)
        
        # Use text features as query q, and its transpose as key k
        q = text_feat_reshaped  # (B, C, N2)
        k = text_feat_reshaped.permute(0, 2, 1)  # (B, N2, C)
        
        # Compute attention map via matrix multiplication to get (B, C, C)
        perception = torch.bmm(q, k)
        
        # Compute perception matrix by max-minus to highlight important regions
        perception = torch.max(perception, -1, keepdim=True)[0].expand_as(perception) - perception
        # Apply softmax to perception matrix if needed
        #perception = self.softmax(perception)   
        
        # Use image features as values v
        v = img_feat_reshaped  # (B, C, N1)
        # Weight image features by the perception matrix to obtain perception info (B, C, N1)
        perception_info = torch.bmm(perception.float(), v.float())
        
        # Reshape perception info back to (B, C*N1)
        perception_info = perception_info.view(B, -1)
        # Dropout layer
        perception_info = self.dropout(perception_info)
        
        # Weighted fusion of perception info with original image features
        # Ensure dimension matching; only use the first C*N1 dimensions
        output = self.gamma * perception_info + img_feat[:, :self.C*N1]
        
        return output

# CAM module wrapper
class CAMModule(nn.Module):
    """Wrapper for CAM to match the existing interface"""
    def __init__(self, config):
        super().__init__()
        self.embed_size = config["FUSION_IN"]
        
        # Read C from config (default 8)
        self.C = config.get("CAM_C", 8)
        
        # Read image and text feature dimensions, or use defaults
        self.image_dim = config.get("CAM_IMAGE_DIM", self.embed_size)
        self.text_dim = config.get("CAM_TEXT_DIM", self.embed_size)
        
        # Initialize CAM
        self.cam = CAM(C=self.C)
        
        # Compute N1 and N2
        self.N1 = self.image_dim // self.C
        self.N2 = self.text_dim // self.C
        
        # Input projection to target dimensions
        self.image_proj = nn.Linear(self.embed_size, self.image_dim) if self.embed_size != self.image_dim else nn.Identity()
        self.text_proj = nn.Linear(self.embed_size, self.text_dim) if self.embed_size != self.text_dim else nn.Identity()
        
        # Output projection back to embed size
        self.output_proj = nn.Linear(self.C * self.N1, self.embed_size)
        
        # Add dropout layer
        self.dropout = nn.Dropout(config.get("DROPOUT", 0.3))
        
    def forward(self, x, q):
        """Forward method for CAM
        Args:
            x: vision features (batch, feature_dim)
            q: text features (batch, feature_dim)
        Returns:
            output_features: processed features
            gate_prob: None (to keep the interface consistent)
            importance_loss: 0 (to keep the interface consistent)
        """
        # Project features to specified dimensions
        x_proj = self.image_proj(x)
        # Optional SiLU activation and dropout
        #x_proj = nn.SiLU()(x_proj)
        #x_proj = self.dropout(x_proj)
        
        q_proj = self.text_proj(q)
        # Optional SiLU activation and dropout
        #q_proj = nn.SiLU()(q_proj)
        #q_proj = self.dropout(q_proj)
        
        # Apply CAM to projected features
        cam_output = self.cam(x_proj, q_proj)
        
        # Project output back to embed size
        output_features = self.output_proj(cam_output)
        # Optional SiLU activation and dropout
        #output_features = nn.SiLU()(output_features)
        #output_features = self.dropout(output_features)
        
        # Keep interface consistent with MoE
        gate_prob = None
        importance_loss = torch.tensor(0.0, device=x.device)
        
        return output_features, gate_prob, importance_loss


class Expert(nn.Module):
    def __init__(self, emb_size):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, x):
        return self.seq(x)


# Mixture of Experts
class MoE(nn.Module):
    def __init__(self, config, experts, top, emb_size, w_importance=0.01):
        super().__init__()
        self.attConfig = config["attnConfig"]
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(experts)])
        self.top = top
        self.crossAtt = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        self.gate = nn.Linear(emb_size, experts)
        self.noise = nn.Linear(emb_size, experts)  # add noise to gate
        self.w_importance = w_importance  # expert balance(for loss)

    def forward(self, x, q):  # x: (batch,seq_len,emb)
        x_shape = x.shape

        x = x.reshape(-1, x_shape[-1])  # (batch*seq_len,emb)

        # gates
        att = self.crossAtt(x.unsqueeze(1), q.unsqueeze(1)).squeeze(1)
        gate_logits = self.gate(att)  # (batch*seq_len,experts)
        gate_prob = softmax(gate_logits, dim=-1)  # (batch*seq_len,experts)

        # 2024-05-05 Noisy Top-K Gating
        if self.training:
            noise = torch.randn_like(gate_prob) * nn.functional.softplus(
                self.noise(x))  # https://arxiv.org/pdf/1701.06538 , StandardNormal()*Softplus((x*W_noise))
            gate_prob = gate_prob + noise

        # top expert
        top_weights, top_index = torch.topk(gate_prob, k=self.top,
                                            dim=-1)  # top_weights: (batch*seq_len,top), top_index: (batch*seq_len,top)
        top_weights = softmax(top_weights, dim=-1)

        top_weights = top_weights.view(-1)  # (batch*seq_len*top)
        top_index = top_index.view(-1)  # (batch*seq_len*top)

        x = x.unsqueeze(1).expand(x.size(0), self.top, x.size(-1)).reshape(-1, x.size(-1))  # (batch*seq_len*top,emb)
        y = torch.zeros_like(x)  # (batch*seq_len*top,emb)

        # run by per expert
        for expert_i, expert_model in enumerate(self.experts):
            x_expert = x[top_index == expert_i]  # (...,emb)
            y_expert = expert_model(x_expert)  # (...,emb)

            add_index = (top_index == expert_i).nonzero().flatten()
            y = y.index_add(dim=0, index=add_index,
                            source=y_expert)  # y[top_index==expert_i]=y_expert

        # weighted sum experts
        top_weights = top_weights.view(-1, 1).expand(-1, x.size(-1))  # (batch*seq_len*top,emb)
        y = y * top_weights
        y = y.view(-1, self.top, x.size(-1))  # (batch*seq_len,top,emb)
        y = y.sum(dim=1)  # (batch*seq_len,emb)

        # experts balance loss
        # https://arxiv.org/pdf/1701.06538 BALANCING EXPERT UTILIZATION
        if self.training:
            importance = gate_prob.sum(dim=0)  # sum( (batch*seq_len,experts) , dim=0)
            # Coefficient of Variation(CV), CV = standard deviation / mean
            importance_loss = self.w_importance * (torch.std(importance) / torch.mean(importance)) ** 2
        else:
            importance = gate_prob.sum(dim=0)
            importance_loss = self.w_importance * (torch.std(importance) / torch.mean(importance)) ** 2
            # importance_loss = None
        return y.view(x_shape), gate_prob, importance_loss


# CrossAttentionModule
class CrossAttentionModule(nn.Module):
    """Improved CrossAttention module"""
    def __init__(self, config):
        super().__init__()
        self.attConfig = config["attnConfig"]
        self.embed_size = self.attConfig["embed_size"]
        
        # Cross Attention
        self.crossAtt = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        
        # Feed-forward network (similar to Expert)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size * 4),
            nn.SiLU(),
            nn.Dropout(self.attConfig["attn_dropout"]),
            nn.Linear(self.embed_size * 4, self.embed_size),
            nn.Dropout(self.attConfig["attn_dropout"])
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.embed_size, self.embed_size)
        
    def forward(self, x, q):
        """Improved forward method
        Args:
            x: vision features (batch, feature_dim)
            q: text features (batch, feature_dim)
        Returns:
            output_features: processed features
            gate_prob: None (to keep the interface consistent)
            importance_loss: 0 (to keep the interface consistent)
        """
        # Residual connection + layer normalization + Cross Attention
        attn_output = self.crossAtt(x.unsqueeze(1), q.unsqueeze(1)).squeeze(1)
        x = self.norm1(x + attn_output)  # Residual connection
        
        # Residual connection + layer normalization + FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual connection
        
        # Final output projection
        output_features = self.output_proj(x)
        
        # Keep interface consistent with MoE
        gate_prob = None
        importance_loss = torch.tensor(0.0, device=x.device)
        
        return output_features, gate_prob, importance_loss


# FusionModule
class FusionModule(nn.Module):
    """Module for direct fusion of visual and text features"""
    def __init__(self, config):
        super().__init__()
        self.embed_size = config["FUSION_IN"]
        self.hidden_size = config["FUSION_HIDDEN"]
        self.dropout = 0.4
        self.fusion_dropout = nn.Dropout(self.dropout)
        
        
    def forward(self, x, q):
        """Directly fuse visual and text features
        Args:
            x: vision features (batch, feature_dim)
            q: text features (batch, feature_dim)
        Returns:
            fused_features: fused features
            gate_prob: None (to keep the interface consistent)
            importance_loss: 0 (to keep the interface consistent)
        """
        # Element-wise addition without dimension changes or extra operations
        fused_features = x + q
        fused_features = self.fusion_dropout(fused_features)
        
        # Keep interface consistent with MoE; return None and 0
        gate_prob = None
        importance_loss = torch.tensor(0.0, device=x.device)

        return fused_features, gate_prob, importance_loss


class RouterGate(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(RouterGate, self).__init__()
        self.config = config
        self.embed_size = self.config["FUSION_IN"]
        self.attConfig = self.config["attnConfig"]
        self.output = int(self.attConfig["embed_size"] / 4)
        
        # 获取融合模式配置
        self.fusion_mode = self.config.get('fusion_mode', 'moe').lower()
        
        # 根据融合模式初始化相应的模块
        if self.fusion_mode == 'moe':
            experts = self.config["EXPERTS"]
            top = self.config["TOP"]
            self.moe = MoE(config, experts, top, self.attConfig["embed_size"])
        elif self.fusion_mode == 'crossattention':
            self.cross_attention_module = CrossAttentionModule(config)
        elif self.fusion_mode == 'fusion':
            self.fusion_module = FusionModule(config)
        elif self.fusion_mode == 'cam':
            self.cam_module = CAMModule(config)
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}. Supported modes: 'moe', 'crossattention', 'fusion', 'cam'")
            
        # 共享的CNN编码器和线性层
        self.cnnEncoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnnEncoder.fc.in_features
        self.cnnEncoder.fc = torch.nn.Linear(num_ftrs, self.output)
        self.cnnEncoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.linerImg = nn.Linear(self.attConfig["embed_size"], self.output)
        self.out = nn.Linear(int(self.embed_size * 2), self.embed_size)

    def forward(self, source, target, background, image, text):
        # Extract visual features (common across all modes)
        s = self.cnnEncoder(source)
        t = self.cnnEncoder(target)
        b = self.cnnEncoder(background)
        img = self.linerImg(image)
        img = nn.SiLU()(img)
        visionFeatures = torch.cat((s, t, b, img), dim=1)

        # Select fusion strategy based on configuration
        if self.fusion_mode == 'moe':
            # MoE flow
            moeFeatures, gate_prob, importance_loss = self.moe(visionFeatures, text)
            return moeFeatures, gate_prob, importance_loss
        elif self.fusion_mode == 'crossattention':
            # CrossAttention flow
            crossAttFeatures, gate_prob, importance_loss = self.cross_attention_module(visionFeatures, text)
            return crossAttFeatures, gate_prob, importance_loss
        elif self.fusion_mode == 'fusion':
            # Direct Fusion flow
            fusionFeatures, gate_prob, importance_loss = self.fusion_module(visionFeatures, text)
            return fusionFeatures, gate_prob, importance_loss
        elif self.fusion_mode == 'cam':
            # CAM flow
            camFeatures, gate_prob, importance_loss = self.cam_module(visionFeatures, text)
            return camFeatures, gate_prob, importance_loss
