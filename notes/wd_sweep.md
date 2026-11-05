# Weight-decay sweep (partial: 4 of 8 runs)

**Status:** partial results. Wave 2 (wd 0.3 and 0.5 seed 1, wd 0.03 seeds 0 and 1) is still training. The full write-up replaces this file.

| weight decay | seed | source | grokking step (test acc ≥ 95%) | final train acc | final test acc |
|---|---|---|---|---|---|
| 0 | 0 | existing `no_decay` | did not grok within 25,000 steps | 100.00% | 3.02% |
| 0 | 1 | existing `no_decay` | did not grok within 25,000 steps | 100.00% | 4.05% |
| 0 | 2 | existing `no_decay` | did not grok within 25,000 steps | 100.00% | 4.33% |
| 0.1 | 0 | sweep (this PR) | did not grok within 25,000 steps | 100.00% | 5.50% |
| 0.1 | 1 | sweep (this PR) | did not grok within 25,000 steps | 100.00% | 8.11% |
| 0.3 | 0 | sweep (this PR) | did not grok within 25,000 steps | 100.00% | 12.80% |
| 0.5 | 0 | sweep (this PR) | 20,514 | 100.00% | 100.00% |
| 1 | 0 | existing `baseline` | 9,868 | 100.00% | 100.00% |
| 1 | 1 | existing `baseline` | 5,785 | 100.00% | 100.00% |
| 1 | 2 | existing `baseline` | 4,216 | 100.00% | 100.00% |
