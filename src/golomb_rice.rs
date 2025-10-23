use crate::bitstream::BitWriter;

/// 使用 Golomb-Rice 编码数字数组
pub fn encode_golomb_rice_numbers(numbers: &[i64], k: u32, bw: &mut BitWriter) {
    let shift = k;
    let mask = if k > 0 { (1u64 << k) - 1 } else { 0 };
    
    for &n in numbers {
        if n < 0 {
            panic!("Golomb-Rice supports only non-negative integers");
        }
        let n = n as u64;
        let q = n >> shift;
        let r = if k > 0 { n & mask } else { 0 };
        bw.write_unary_ones_then_zero(q);
        if k > 0 {
            bw.write_bits(r, k);
        }
    }
}

/// 为给定的非负整数选择最优的 k 值（0..k_max），最小化 GR 位成本
pub fn select_optimal_gr_k(numbers: &[i64], k_max: Option<u32>) -> u32 {
    if numbers.is_empty() {
        return 0;
    }
    
    let max_val = *numbers.iter().max().unwrap();
    if max_val <= 0 {
        return 0;
    }
    
    let k_max = k_max.unwrap_or_else(|| {
        let bits = 64 - (max_val as u64).leading_zeros();
        bits.min(30)
    });
    
    let mut best_k = 0u32;
    let mut best_cost = u64::MAX;
    let count = numbers.len() as u64;
    
    for k in 0..=k_max {
        // q = numbers >> k, sum of q
        let q_sum: u64 = numbers.iter().map(|&n| (n as u64) >> k).sum();
        // Unary cost q+1 per number, plus remainder bits if k>0
        let mut cost = q_sum + count; // q + 1
        if k > 0 {
            cost += count * (k as u64);
        }
        if cost < best_cost {
            best_cost = cost;
            best_k = k;
        }
    }
    
    best_k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;

    #[test]
    fn test_golomb_rice_encoding() {
        let numbers = vec![0, 1, 2, 3, 10, 15, 20];
        let k = select_optimal_gr_k(&numbers, Some(5));
        
        let mut bw = BitWriter::new();
        encode_golomb_rice_numbers(&numbers, k, &mut bw);
        let bytes = bw.finish();
        
        let mut br = BitReader::new(bytes);
        for &expected in &numbers {
            let q = br.read_unary_ones_then_zero_count();
            let r = if k > 0 { br.read_bits(k) } else { 0 };
            let decoded = ((q << k) | r) as i64;
            assert_eq!(decoded, expected);
        }
    }
}
