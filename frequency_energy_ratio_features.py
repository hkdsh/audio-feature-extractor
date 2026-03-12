import numpy as np
import librosa
from config import CONFIG


class FrequencyEnergyRatioExtractor:
    """七段频率能量比特征提取器 - 修复版"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        
    def extract_all_features(self, y, sr):
        """
        提取七段频率能量比 (Ratio)
        计算：(该频段能量 / 总能量) 的均值
        """
        # 计算功率谱 (Power Spectrum)
        # S shape: (n_fft/2 + 1, n_frames)
        S = np.abs(librosa.stft(y, n_fft=self.config.n_fft, hop_length=self.config.hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)
        
        # 计算每一帧的总能量
        total_energy_per_frame = np.sum(S, axis=0)
        # 避免除以零 (将0替换为一个极小值)
        total_energy_per_frame[total_energy_per_frame == 0] = 1e-10
        
        # 定义频段 (Hz)
        bands = [
            (60, 150),
            (150, 350),
            (350, 700),
            (700, 1500),
            (1500, 3000),
            (3000, 6000),
            (6000, 10000) # 注意：如果预处理做了低通滤波，这个频段可能为空，建议预处理保留高频
        ]
        
        features = {}
        
        for low, high in bands:
            # 找到该频段对应的频率 Bin 索引
            idx = np.where((freqs >= low) & (freqs < high))[0]
            
            if len(idx) > 0:
                # 计算该频段每一帧的能量和
                band_energy_per_frame = np.sum(S[idx, :], axis=0)
                
                # [核心修复] 计算比例：该频段能量 / 该帧总能量
                ratios = band_energy_per_frame / total_energy_per_frame
                
                # 取时间上的平均值
                features[f'Energy_Ratio_{low}_{high}Hz_mean'] = float(np.mean(ratios))
            else:
                features[f'Energy_Ratio_{low}_{high}Hz_mean'] = 0.0
        
        return features

# 测试代码
if __name__ == "__main__":
    sr = CONFIG.sr
    y = np.random.randn(sr) # 白噪声，理论上各频段能量应该比较均匀
    extractor = FrequencyEnergyRatioExtractor()
    feats = extractor.extract_all_features(y, sr)
    print("Energy Ratio Test:")
    for k, v in feats.items():
        print(f"{k}: {v:.4f}")