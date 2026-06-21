# Paper Outline

## Workshop-style paper: A Causal Atlas of a Small Language Model

### Abstract
TBD

### 1. Introduction
- Motivation: understanding what training changes inside LLMs
- Gap: most interpretability focuses on frontier models
- Contribution: systematic causal atlas of a 0.5B model

### 2. Related Work
- Mechanistic interpretability (Elhage et al., Wang et al.)
- Activation patching (Geiger et al.)
- Training perturbation studies
- Small model interpretability
- LoRA and adapter analysis

### 3. Methods
- Model and task suite
- Ablation methodology
- Activation patching
- Steering vectors
- Training regimes
- Evidence hierarchy and confidence scoring

### 4. Baseline Behaviour
- Per-family performance
- Failure modes
- Tokenization effects

### 5. Causal Localisation
- Layer ablation results
- Head and MLP ablation
- Activation patching
- Path patching and candidate circuits

### 6. Training Perturbation Analysis
- CPT vs SFT internal differences
- LoRA rank and target-module effects
- Dataset shard attribution
- Checkpoint timeline

### 7. Component Atlas
- Methodology
- Key entries
- Confidence distribution

### 8. Case Studies
- Strong causal component
- Training-induced skill
- Negative result / failure

### 9. Discussion
- Implications for fine-tuning practice
- Limitations
- Future work

### 10. Conclusion

### References

---

*Fill with actual results. Workshop paper, not NeurIPS.*
