import torch
import torch.nn as nn
import pywt
import numpy as np

"""
Wavelet-Enhanced IIA (WIIA) module
"""

class AttentionWeight(nn.Module):
    """Attention weight module"""
    def __init__(self, channel, kernel_size):
        super(AttentionWeight, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size, padding=padding, groups=channel, bias=False)
        self.bn = nn.BatchNorm1d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, w, c, h = x.size()
        x_weight = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_weight = self.conv1(x_weight).view(b, c, h)
        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        x_weight = x_weight.view(b, 1, c, h)
        return x * x_weight


class WaveletTransform(nn.Module):
    """Wavelet transform module"""
    
    def __init__(self, wavelet='db4'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
    
    def dwt2d(self, x):
        """2D wavelet transform"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Ensure even dimensions
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Apply wavelet transform per channel
        coeffs_list = []
        for i in range(channels):
            channel_data = x[:, i, :, :].detach().cpu().numpy()
            batch_coeffs = []
            for j in range(batch_size):
                coeffs = pywt.dwt2(channel_data[j], self.wavelet)
                batch_coeffs.append(coeffs)
            coeffs_list.append(batch_coeffs)
        
        # Reorganize data format
        LL_list, (LH_list, HL_list, HH_list) = [], ([], [], [])
        for channel_coeffs in coeffs_list:
            channel_LL, channel_LH, channel_HL, channel_HH = [], [], [], []
            for coeffs in channel_coeffs:
                LL, (LH, HL, HH) = coeffs
                channel_LL.append(LL)
                channel_LH.append(LH)
                channel_HL.append(HL)
                channel_HH.append(HH)
            
            LL_list.append(np.stack(channel_LL))
            LH_list.append(np.stack(channel_LH))
            HL_list.append(np.stack(channel_HL))
            HH_list.append(np.stack(channel_HH))
        
        # Convert back to tensor
        LL = torch.from_numpy(np.stack(LL_list, axis=1)).float().to(device)
        LH = torch.from_numpy(np.stack(LH_list, axis=1)).float().to(device)
        HL = torch.from_numpy(np.stack(HL_list, axis=1)).float().to(device)
        HH = torch.from_numpy(np.stack(HH_list, axis=1)).float().to(device)
        
        return LL, LH, HL, HH
    
    def idwt2d(self, LL, LH, HL, HH, target_shape):
        """2D inverse wavelet transform"""
        batch_size, channels = LL.shape[:2]
        device = LL.device
        
        # Convert to NumPy for inverse transform
        LL_np = LL.detach().cpu().numpy()
        LH_np = LH.detach().cpu().numpy()
        HL_np = HL.detach().cpu().numpy()
        HH_np = HH.detach().cpu().numpy()
        
        reconstructed_list = []
        for i in range(channels):
            channel_reconstructed = []
            for j in range(batch_size):
                coeffs = (LL_np[j, i], (LH_np[j, i], HL_np[j, i], HH_np[j, i]))
                reconstructed = pywt.idwt2(coeffs, self.wavelet)
                channel_reconstructed.append(reconstructed)
            reconstructed_list.append(np.stack(channel_reconstructed))
        
        result = torch.from_numpy(np.stack(reconstructed_list, axis=1)).float().to(device)
        
        # Crop to target size
        target_h, target_w = target_shape[-2:]
        result = result[:, :, :target_h, :target_w]
        
        return result


class WaveletIIA(nn.Module):
    """Wavelet-Enhanced IIA module"""
    def __init__(self, channel, kernel_size, wavelet='db4', ablation_enable=False, ablation_type='none'):
        super(WaveletIIA, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet)
        
        # Create attention modules for LH (vertical details) and HL (horizontal details)
        self.attention_lh = AttentionWeight(channel, kernel_size)  # vertical details
        self.attention_hl = AttentionWeight(channel, kernel_size)  # horizontal details
        
        # Ablation control
        self.ablation_enable = ablation_enable
        self.ablation_type = (ablation_type.lower() if isinstance(ablation_type, str) else 'none')

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input feature map of shape (batch_size, channels, height, width)
        Returns:
            output feature map with the same shape as input
        """
        original_shape = x.shape
        
        # 1. Wavelet decomposition
        LL, LH, HL, HH = self.wavelet_transform.dwt2d(x)

        # 2. Apply attention based on ablation type
        should_attend_lh = True
        should_attend_hl = True
        if self.ablation_enable:
            if self.ablation_type == 'disable_attention_lh':
                should_attend_lh = False
            elif self.ablation_type == 'disable_attention_hl':
                should_attend_hl = False

        # LH (vertical details)
        if should_attend_lh:
            LH_processed = LH.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
            LH_processed = self.attention_lh(LH_processed).permute(0, 2, 3, 1)  # (B, C, H, W)
        else:
            LH_processed = LH  # no attention

        # HL (horizontal details)
        if should_attend_hl:
            HL_processed = HL.permute(0, 2, 1, 3).contiguous()  # (B, H, C, W)
            HL_processed = self.attention_hl(HL_processed).permute(0, 2, 1, 3)  # (B, C, H, W)
        else:
            HL_processed = HL  # no attention

        # 3. Subband masking ablation: zero specified subband
        if self.ablation_enable:
            if self.ablation_type == 'mask_ll':
                LL = torch.zeros_like(LL)
            elif self.ablation_type == 'mask_lh':
                LH_processed = torch.zeros_like(LH)
            elif self.ablation_type == 'mask_hl':
                HL_processed = torch.zeros_like(HL)
            elif self.ablation_type == 'mask_hh':
                HH = torch.zeros_like(HH)

        # 4. Wavelet reconstruction (mask LL/HH as needed; apply attention/masking to LH/HL)
        reconstructed = self.wavelet_transform.idwt2d(LL, LH_processed, HL_processed, HH, x.shape)

        # 5. Residual connection
        output = x + reconstructed

        return output
