Queries evaluated: 73 (synthetic, labeled)

| metric | router | vector_only | fan_out |
|---|---|---|---|
| outcome_accuracy | 0.918 | 0.274 | 0.699 |
| answerable_accuracy | 0.911 | 0.321 | 0.875 |
| non_answerable_accuracy | 0.941 | 0.118 | 0.118 |
| answered_when_it_should_not | 1 | 3 | 4 |
| tool_calls_per_query | 0.9 | 1.0 | 2.3 |
| usd_per_1k_queries | 0.924 | 0.252 | 0.57 |
| usd_per_1k_queries_without_llm_routing | 0.468 | 0.252 | 0.57 |
| synthesis_input_tokens_per_query | 295.4 | 173.3 | 319.7 |
| latency_ms_p50 | 1496.7 | 180.0 | 1146.7 |
| latency_ms_p95 | 1726.2 | 1148.8 | 1360.4 |
| context_precision_mean | 0.8 | 0.407 | 0.711 |

router by category: clarify 6/6, compound 5/6, directory 13/13, metrics 18/20, no_answer 2/3, out_of_scope 4/4, policy 15/17, refused 4/4
vector_only by category: clarify 0/6, compound 0/6, directory 2/13, metrics 0/20, no_answer 2/3, out_of_scope 0/4, policy 16/17, refused 0/4
fan_out by category: clarify 0/6, compound 2/6, directory 13/13, metrics 18/20, no_answer 2/3, out_of_scope 0/4, policy 16/17, refused 0/4
