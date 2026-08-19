
use std::collections::HashSet;

use rand::Rng;

pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl SamplingConfig {
    pub const GEMMA_RECOMMENDED: Self = Self {
        temperature: 1.0,
        top_k: 64,
        top_p: 0.95,
    };
}

pub fn sample(logits: &[f32], config: &SamplingConfig, banned: &HashSet<usize>, rng: &mut impl Rng) -> usize {
    assert!(!logits.is_empty(), "sample requires at least one logit");
    let temperature = config.temperature.max(1e-5);

    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    if config.top_k > 0 {
        let keep = config.top_k.saturating_add(banned.len()).min(ranked.len());
        ranked.select_nth_unstable_by(keep - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
        ranked.truncate(keep);
    }
    ranked.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    if !banned.is_empty() {
        let filtered: Vec<usize> = ranked.iter().copied().filter(|id| !banned.contains(id)).collect();
        if !filtered.is_empty() {
            ranked = filtered;
        }
    }
    if config.top_k > 0 {
        ranked.truncate(config.top_k.min(ranked.len()));
    }

    let max_logit = logits[ranked[0]];
    let mut probs: Vec<f32> = ranked
        .iter()
        .map(|&i| ((logits[i] - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    let mut nucleus_len = probs.len();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= config.top_p {
            nucleus_len = i + 1;
            break;
        }
    }
    let nucleus_sum: f32 = probs[..nucleus_len].iter().sum();

    let threshold = rng.random::<f32>() * nucleus_sum;
    let mut acc = 0.0;
    for (i, &p) in probs[..nucleus_len].iter().enumerate() {
        acc += p;
        if acc >= threshold {
            return ranked[i];
        }
    }
    ranked[nucleus_len - 1]
}
