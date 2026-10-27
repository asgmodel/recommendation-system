# Experimental Protocol

1. Fix ontology, catalog, aliases, data partitions and random seeds.
2. Review question-SPARQL pairs and reference executions. Reserve query families for validation and testing.
3. Select mT5 settings on validation execution F1 for nonempty references. Report empty-reference cases separately.
4. Fit BKT by bounded maximum likelihood on training assessment sequences. Evaluate next-response log loss, Brier score, AUC and calibration on separate learners.
5. Compare template/static, template/adaptive, mT5/static and mT5/adaptive arms on identical test records. The two adaptation arms for each generator share retrieval outputs and eligibility rules.
6. Report Top-1, Hit@3, Hit@5, MRR@k, MAP@k, NDCG@k and Precision@k. Bootstrap by question family. Family-level intervals from the small development set are integration diagnostics.
7. Evaluate learner outcomes separately using independent pre/post tasks and a prespecified controlled design. Retrieval and ranking results do not estimate learning gains.

The default experiment runs template/static and template/adaptive. Selecting `both` requires a trained mT5 checkpoint; a missing model never triggers baseline substitution.

The supplied historical ontology checks and article ranking metrics describe different tasks. Preserve their original definitions and do not combine their percentages with neural query accuracy.

Optional policy experiments use Dueling Double DQN on recorded transitions with episode-disjoint validation. Define the behavior policy, action masks and reward before collection. Prospective or justified off-policy evaluation is required to estimate policy benefit.
