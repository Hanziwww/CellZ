use crate::bitstream::{BitReader, BitWriter};
use crate::golomb_rice::select_optimal_gr_k;

/// 自适应混合值编码器统计信息
#[derive(Debug, Clone)]
pub struct EncodeStats {
    pub n_value_1: usize,
    pub n_small: usize,
    pub n_large: usize,
    pub threshold: u32,
    pub k_large: u32,
    pub small_bits: u32,
}

/// 自适应混合值编码器
pub struct AdaptiveHybridValueEncoder {
    pub threshold: u32,
    pub k_large: u32,
    pub small_bits: u32,
}

impl AdaptiveHybridValueEncoder {
    pub fn new() -> Self {
        Self {
            threshold: 15,
            k_large: 0,
            small_bits: 4,
        }
    }

    /// 计算给定阈值所需的最小位数
    fn min_small_bits_for_threshold(thr: u32) -> u32 {
        let max_symbol = if thr >= 2 { thr - 2 } else { 0 };
        if max_symbol == 0 {
            return 1;
        }
        // 预留 small_bits==7 作为截断二进制模式的哨兵
        let bits = 32 - max_symbol.leading_zeros();
        bits.max(1).min(6)
    }

    /// 编码值数组（已转换为 v-1 形式）
    pub fn encode_values(&mut self, values: &[i64], bw: &mut BitWriter) -> EncodeStats {
        let mut stats = EncodeStats {
            n_value_1: 0,
            n_small: 0,
            n_large: 0,
            threshold: 0,
            k_large: 0,
            small_bits: 0,
        };

        if values.is_empty() {
            // 空值：写入默认参数
            bw.write_bits(15, 5); // threshold
            bw.write_bits(0, 5);  // k_large
            bw.write_bits(4, 3);  // small_bits
            return stats;
        }

        let n_zero = values.iter().filter(|&&v| v == 0).count();

        let mut best_cost = u64::MAX;
        let mut best_threshold = 15u32;
        let mut best_small_bits = 4u32;
        let mut best_k_large = 0u32;

        // 阈值搜索范围：1..31（5bit 头部允许的范围）
        for thr in 1..32 {
            let small_vals: Vec<i64> = values.iter()
                .filter(|&&v| v > 0 && v < thr as i64)
                .copied()
                .collect();
            let large_vals: Vec<i64> = values.iter()
                .filter(|&&v| v >= thr as i64)
                .copied()
                .collect();
            
            let n_small = small_vals.len();
            let n_large = large_vals.len();

            let sb = Self::min_small_bits_for_threshold(thr);

            // large 值：求最优 k_large
            let (k_opt, large_cost) = if !large_vals.is_empty() {
                let shifted: Vec<i64> = large_vals.iter()
                    .map(|&v| v - thr as i64)
                    .collect();
                let k = select_optimal_gr_k(&shifted, Some(31));
                let q_sum: u64 = shifted.iter().map(|&v| (v as u64) >> k).sum();
                let mut cost = q_sum + n_large as u64; // unary q+1
                if k > 0 {
                    cost += (n_large as u64) * (k as u64);
                }
                (k, cost)
            } else {
                (0, 0)
            };

            // 小值域成本：比较定长 vs 截断二进制
            let cost_small_fixed = (n_small as u64) * (2 + sb as u64);
            
            let cost_small_trunc = if n_small > 0 && thr > 1 {
                let m = thr - 1;
                let small_symbols: Vec<i64> = small_vals.iter()
                    .map(|&v| v - 1)
                    .collect();
                let b = if m > 0 { 31 - m.leading_zeros() } else { 0 };
                // For truncated binary, t = 2^{b+1} - M always; when b==0 and M==1, t==1
                let t = (1u32 << (b + 1)) - m;
                
                let payload_bits_trunc = if b == 0 {
                    if t == 1 { 0 } else { n_small as u64 }
                } else {
                    let cnt_lt_t = small_symbols.iter()
                        .filter(|&&s| (s as u32) < t)
                        .count();
                    (cnt_lt_t as u64) * (b as u64) + 
                    ((n_small - cnt_lt_t) as u64) * ((b + 1) as u64)
                };
                (n_small as u64) * 2 + payload_bits_trunc
            } else {
                (n_small as u64) * 2
            };

            // 选择更优的小值方案
            let (small_bits_selected, cost_small) = if cost_small_trunc < cost_small_fixed {
                (7u32, cost_small_trunc) // 哨兵：表示截断二进制
            } else {
                (sb, cost_small_fixed)
            };

            // 总成本
            let cost_zero = (n_zero as u64) * 1;
            let cost_large = (n_large as u64) * 2 + large_cost;
            let total_bits = cost_zero + cost_small + cost_large + 13; // 13 bits 头部

            if total_bits < best_cost {
                best_cost = total_bits;
                best_threshold = thr;
                best_small_bits = small_bits_selected;
                best_k_large = k_opt;
            }
        }

        self.threshold = best_threshold;
        self.small_bits = best_small_bits;
        self.k_large = best_k_large;

        // 写入编码参数到流头部
        bw.write_bits(self.threshold as u64, 5);
        bw.write_bits(self.k_large as u64, 5);
        bw.write_bits(self.small_bits as u64, 3);

        stats.threshold = self.threshold;
        stats.k_large = self.k_large;
        stats.small_bits = self.small_bits;

        // 逐个编码值
        for &val in values {
            if val == 0 { // 对应原始值=1
                bw.write_bit(false);
                stats.n_value_1 += 1;
            } else if val < self.threshold as i64 { // 小值
                bw.write_bit(true);
                bw.write_bit(false);
                
                // 小值编码：定长或截断二进制
                if self.small_bits == 7 {
                    let m = self.threshold - 1;
                    let s = (val - 1) as u32;
                    
                    if m > 0 {
                        let b = 31 - m.leading_zeros();
                        let t = if b > 0 { (1u32 << (b + 1)) - m } else { 0 };
                        
                        if b > 0 {
                            if s < t {
                                bw.write_bits(s as u64, b);
                            } else {
                                let y = s + t;
                                bw.write_bits(y as u64, b + 1);
                            }
                        }
                    }
                } else {
                    bw.write_bits((val - 1) as u64, self.small_bits);
                }
                stats.n_small += 1;
            } else { // 大值 >=threshold
                bw.write_bit(true);
                bw.write_bit(true);
                
                let val_shifted = (val - self.threshold as i64) as u64;
                let q = if self.k_large > 0 {
                    val_shifted >> self.k_large
                } else {
                    val_shifted
                };
                let r = if self.k_large > 0 {
                    val_shifted & ((1u64 << self.k_large) - 1)
                } else {
                    0
                };
                bw.write_unary_ones_then_zero(q);
                if self.k_large > 0 {
                    bw.write_bits(r, self.k_large);
                }
                stats.n_large += 1;
            }
        }

        stats
    }
}

/// 自适应混合值解码器
pub struct AdaptiveHybridValueDecoder {
    pub threshold: u32,
    pub k_large: u32,
    pub small_bits: u32,
}

impl AdaptiveHybridValueDecoder {
    pub fn new() -> Self {
        Self {
            threshold: 15,
            k_large: 0,
            small_bits: 4,
        }
    }

    /// 读取编码参数
    pub fn read_header(&mut self, br: &mut BitReader) {
        self.threshold = br.read_bits(5) as u32;
        self.k_large = br.read_bits(5) as u32;
        self.small_bits = br.read_bits(3) as u32;
    }

    /// 解码单个值（返回 v-1 形式）
    pub fn decode_value(&self, br: &mut BitReader) -> i64 {
        let flag = br.read_bit();
        if !flag { // 值=0（原始值1）
            return 0;
        }
        
        let small_flag = br.read_bit();
        if !small_flag { // 小值：1到threshold-1
            if self.small_bits == 7 {
                // 截断二进制解码
                let m = self.threshold - 1;
                if m == 0 {
                    return 1;
                }
                
                let b = if m > 0 { 31 - m.leading_zeros() } else { 0 };
                let t = if b > 0 { (1u32 << (b + 1)) - m } else { 0 };
                
                let s = if b == 0 {
                    0
                } else {
                    let x = br.read_bits(b) as u32;
                    if x < t {
                        x
                    } else {
                        let y = (x << 1) | (if br.read_bit() { 1 } else { 0 });
                        y - t
                    }
                };
                (s + 1) as i64
            } else {
                // 定长读取
                let val_minus_1 = br.read_bits(self.small_bits);
                (val_minus_1 + 1) as i64
            }
        } else { // 大值 >=threshold
            let q = br.read_unary_ones_then_zero_count();
            let r = if self.k_large > 0 {
                br.read_bits(self.k_large)
            } else {
                0
            };
            let val_shifted = (q << self.k_large) | r;
            (val_shifted + self.threshold as u64) as i64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_value_roundtrip() {
        let values = vec![0, 0, 1, 2, 3, 5, 10, 15, 20, 100];
        
        let mut encoder = AdaptiveHybridValueEncoder::new();
        let mut bw = BitWriter::new();
        let stats = encoder.encode_values(&values, &mut bw);
        
        println!("Stats: {:?}", stats);
        
        let bytes = bw.finish();
        let mut br = BitReader::new(bytes);
        
        let mut decoder = AdaptiveHybridValueDecoder::new();
        decoder.read_header(&mut br);
        
        assert_eq!(decoder.threshold, encoder.threshold);
        assert_eq!(decoder.k_large, encoder.k_large);
        assert_eq!(decoder.small_bits, encoder.small_bits);
        
        for &expected in &values {
            let decoded = decoder.decode_value(&mut br);
            assert_eq!(decoded, expected);
        }
    }
}

