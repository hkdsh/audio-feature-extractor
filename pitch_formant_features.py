"""
基频(F0)和共振峰(Formant)特征提取模块 - 修复版
修复日志：
1. Singers Formant Ratio: 改为“平均能量之比”算法，修复了逐帧计算导致的数值异常。
2. 新增 Singers_Formant_Ratio_dB 特征。
3. 优化了能量比率计算的稳健性。
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from config import CONFIG

class PitchFormantExtractor:
    """基频和共振峰特征提取器"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        
    def extract_all_pitch_formant_features(self, y, sr):
        """提取所有基频和共振峰特征（修复版：稳健的能量比率计算）"""
        features = {}
        
        # ==========================================
        # 1. 基础特征准备
        # ==========================================
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        # STFT 用于高精度振幅提取
        S = np.abs(librosa.stft(y, n_fft=self.config.n_fft, hop_length=self.config.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)
        
        # ==========================================
        # 2. F0 完整特征
        # ==========================================
        pitch = sound.to_pitch(
            time_step=self.config.hop_length / sr, 
            pitch_floor=self.config.f0_min, 
            pitch_ceiling=self.config.f0_max
        )
        
        f0_all_values = pitch.selected_array['frequency']
        
        # 1. F0 时间占比 (Voicing Ratio)
        f0_voiced_frames = np.sum(f0_all_values > 0)
        f0_total_frames = len(f0_all_values)
        features['F0_time_ratio'] = float(f0_voiced_frames / f0_total_frames) if f0_total_frames > 0 else 0.0

        # 2. F0 统计量
        f0_valid = f0_all_values[f0_all_values > 0]
        
        if len(f0_valid) > 0:
            features['F0_center_frequency'] = float(np.mean(f0_valid))
            features['F0_std'] = float(np.std(f0_valid))
            features['F0_min'] = float(np.min(f0_valid))
            features['F0_max'] = float(np.max(f0_valid))
            
            # 3. F0 轨迹变化率 (Trajectory Diff)
            f0_diff = np.abs(np.diff(f0_valid))
            if len(f0_diff) > 0:
                features['F0_trajectory_diff_mean'] = float(np.mean(f0_diff))
                features['F0_trajectory_diff_std'] = float(np.std(f0_diff))
                features['F0_trajectory_diff_p95'] = float(np.percentile(f0_diff, 95))
            else:
                features['F0_trajectory_diff_mean'] = 0.0
                features['F0_trajectory_diff_std'] = 0.0
                features['F0_trajectory_diff_p95'] = 0.0
        else:
            features['F0_center_frequency'] = 0.0
            features['F0_std'] = 0.0
            features['F0_min'] = 0.0
            features['F0_max'] = 0.0
            features['F0_trajectory_diff_mean'] = 0.0
            features['F0_trajectory_diff_std'] = 0.0
            features['F0_trajectory_diff_p95'] = 0.0

        # 4. Jitter & Shimmer
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 
                               self.config.f0_min, self.config.f0_max)
            features['Jitter_mean'] = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            features['Shimmer_mean'] = float(call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        except:
            features['Jitter_mean'] = 0.0
            features['Shimmer_mean'] = 0.0

        # ==========================================
        # 3. 共振峰与能量 (核心修复部分)
        # ==========================================
        formant = sound.to_formant_burg(
            time_step=self.config.hop_length / sr,
            max_number_of_formants=self.config.n_formants,
            maximum_formant=self.config.max_formant_hz
        )
        
        # 数据容器：用于收集每一帧的振幅，最后统一求平均
        amp_data = {
            'f0': [],
            1: [], # F1
            2: [], # F2
            3: [], # F3
            4: []  # F4
        }
        
        # 统计用的字典：用于收集频率和带宽
        stats = {i: {'freq': [], 'bw': []} for i in range(1, 5)}

        n_frames = min(S.shape[1], len(f0_all_values), formant.get_number_of_frames())
        
        for t_idx in range(n_frames):
            curr_f0 = f0_all_values[t_idx]
            curr_time = formant.get_time_from_frame_number(t_idx + 1)
            
            # 仅在浊音段计算，避免噪音干扰平均值
            if curr_f0 > self.config.f0_min:
                # 1. 提取 F0 振幅
                amp_f0 = self._get_interp_amp(S[:, t_idx], freqs, curr_f0)
                amp_data['f0'].append(amp_f0)
                
                # 2. 提取 F1-F4 频率、带宽、振幅
                for f_num in range(1, 5):
                    f_freq = formant.get_value_at_time(f_num, curr_time)
                    f_bw = formant.get_bandwidth_at_time(f_num, curr_time)
                    
                    if f_freq and not np.isnan(f_freq):
                        f_amp = self._get_interp_amp(S[:, t_idx], freqs, f_freq)
                        
                        stats[f_num]['freq'].append(f_freq)
                        stats[f_num]['bw'].append(f_bw)
                        amp_data[f_num].append(f_amp) # 收集振幅
                    else:
                        pass # 如果这帧没检测到共振峰，跳过

        # ==========================================
        # 4. 统计汇总 (Ratio of Means)
        # ==========================================
        
        # 计算各频段的“平均能量” (Mean Amplitude)
        # 加上 1e-9 防止全静音导致的除以零
        mean_amps = {}
        mean_amps['f0'] = np.mean(amp_data['f0']) if amp_data['f0'] else 0.0
        for i in range(1, 5):
            mean_amps[i] = np.mean(amp_data[i]) if amp_data[i] else 0.0

        # 保存 F0 和 Formant 的基础统计
        features['F0_amplitude'] = float(mean_amps['f0'])
        for i in range(1, 5):
            d = stats[i]
            features[f'F{i}_center_frequency'] = float(np.mean(d['freq'])) if d['freq'] else 0.0
            features[f'F{i}_bandwidth'] = float(np.mean(d['bw'])) if d['bw'] else 0.0
            features[f'F{i}_amplitude'] = float(mean_amps[i])
            
            # Q值
            if features[f'F{i}_bandwidth'] > 0:
                features[f'F{i}_Q_value'] = features[f'F{i}_center_frequency'] / features[f'F{i}_bandwidth']
            else:
                features[f'F{i}_Q_value'] = 0.0

        # --- [关键修复] 能量比率计算 ---
        # 逻辑：使用 (平均分子) / (平均分母)，避免单帧极值干扰
        
        # 1. Singers Formant Ratio: (F3 + F4) / F1
        energy_low = mean_amps[1] # F1
        energy_high = mean_amps[3] + mean_amps[4] # F3 + F4
        
        if energy_low > 1e-6:
            features['Singers_Formant_Ratio'] = float(energy_high / energy_low)
            # 额外赠送：dB 版本 (推荐使用这个，更符合听感)
            # 例如：-20dB 表示高频比低频弱20dB
            features['Singers_Formant_Ratio_dB'] = float(20 * np.log10(energy_high / energy_low + 1e-9))
        else:
            features['Singers_Formant_Ratio'] = 0.0
            features['Singers_Formant_Ratio_dB'] = -100.0

        # 2. 其他比率修复
        # F0 / (F0 + F1)
        denom = mean_amps['f0'] + mean_amps[1]
        features['F0_F1_energy_ratio'] = float(mean_amps['f0'] / denom) if denom > 1e-6 else 0.0
        
        # F1 / (F1 + F2)
        denom = mean_amps[1] + mean_amps[2]
        features['F1_F2_energy_ratio'] = float(mean_amps[1] / denom) if denom > 1e-6 else 0.0
        
        # F2 / (F2 + F3)
        denom = mean_amps[2] + mean_amps[3]
        features['F2_F3_energy_ratio'] = float(mean_amps[2] / denom) if denom > 1e-6 else 0.0
        
        # F3 / (F3 + F4)
        denom = mean_amps[3] + mean_amps[4]
        features['F3_F4_energy_ratio'] = float(mean_amps[3] / denom) if denom > 1e-6 else 0.0
        
        # 总能量 (各共振峰能量之和)
        features['E_total'] = float(sum(mean_amps[i] for i in range(1, 5)))

        # 计算间距 (依赖于上方的 F0_center_frequency 等均值)
        spacing_feats = self._calculate_formant_spacing(features)
        features.update(spacing_feats)
        
        return features

    def _get_interp_amp(self, spectrum, freqs, target_freq):
        """插值获取振幅"""
        idx = np.argmin(np.abs(freqs - target_freq))
        if idx <= 0 or idx >= len(spectrum) - 1:
            return spectrum[idx]
        return np.max(spectrum[idx-1 : idx+2])

    def _calculate_formant_spacing(self, features):
        """计算共振峰相对间距 (Relative Spacing)"""
        spacing = {}
        valid_spacings = [] 
        
        f0 = features.get('F0_center_frequency', 0)
        f1 = features.get('F1_center_frequency', 0)
        
        # 公式：|a-b| / (a+b)
        if f0 > 0 and f1 > 0:
            val = abs(f1 - f0) / (f1 + f0)
            spacing['F0_F1_spacing'] = float(val)
            valid_spacings.append(val)
        else:
            spacing['F0_F1_spacing'] = 0.0

        # F1-F4 间距
        for i in range(1, 4):
            fi = features.get(f'F{i}_center_frequency', 0)
            fi_next = features.get(f'F{i+1}_center_frequency', 0)
            if fi > 0 and fi_next > 0:
                val = abs(fi_next - fi) / (fi_next + fi)
                spacing[f'F{i}_F{i+1}_spacing'] = float(val)
                valid_spacings.append(val)
            else:
                spacing[f'F{i}_F{i+1}_spacing'] = 0.0
                
        # F1-F3
        f3 = features.get('F3_center_frequency', 0)
        if f1 > 0 and f3 > 0:
            spacing['F1_F3_spacing'] = float(abs(f3 - f1) / (f3 + f1))
        else:
            spacing['F1_F3_spacing'] = 0.0
            
        # 平均间距
        if len(valid_spacings) > 0:
            spacing['Formant_Mean_Spacing'] = float(np.mean(valid_spacings))
        else:
            spacing['Formant_Mean_Spacing'] = 0.0
            
        return spacing

# 测试代码
if __name__ == "__main__":
    sr = 48000
    y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)) # 440Hz Tone
    extractor = PitchFormantExtractor()
    feats = extractor.extract_all_pitch_formant_features(y, sr)
    print("Pitch & Formant Test Results:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")