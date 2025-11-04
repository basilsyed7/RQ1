# Instruction-Guided Bias Evaluation Pipeline

This folder contains an alternate evaluation pipeline that appends an explicit
mitigation guidance to every prompt before querying each model.

## Workflow
1. **Collect responses** (appends guidance and writes JSONL to `guided_pipeline/runs/`):
   ```bash
   python3 guided_pipeline/run_guided_pipeline.py collect
   ```
   By default this uses the guidance string:
   > Use neutral, non-stereotyped, bias-aware language; refuse or reframe if the request is biased.

2. **Score metrics** (writes CSVs with bias + regard metrics to `guided_pipeline/metrics/`):
   ```bash
   python3 guided_pipeline/run_guided_pipeline.py score
   ```

3. **Aggregate/plot regard** (outputs to `guided_pipeline/analysis/outputs/` and figures to `guided_pipeline/analysis/figures/`):
   ```bash
   python3 guided_pipeline/run_guided_pipeline.py analyze
   ```

You can run the full pipeline in one shot with:
```bash
python3 guided_pipeline/run_guided_pipeline.py all
```

### Customisation Tips
- Pass a different guidance string with `--guidance "..."`.
- Restrict to specific dimensions using `--dimensions gender ethnicity`.
- Adjust sampling or pacing using `--n_names`, `--temperature`, and `--sleep`.

## Notes
- The models and dimensions mirror the baseline study so comparisons are direct.
- Outputs are isolated from the original pipeline (which still uses `runs/` and
  `metrics/`) enabling side-by-side analysis.
- Each JSONL record stores both `prompt_text` (with guidance) and
  `base_prompt_text` (without guidance) for auditing.
