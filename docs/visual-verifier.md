# Visual verifier and target diagnostic

`scripts/visual_acceptance.py` checks that the pinned Qwen VLMs actually receive
image tensors and return a completed, constrained optical reading on the target
machine. It constructs its own numeric PNGs. It accepts no user document, dataset,
ground truth file, prompt override, or remote image URL.

The default run uses CUDA with BF16 and executes Qwen 0.8B, then Qwen 4B in one
supervised model owner. There is one generation per model. The previous Qwen is
evicted before the next loads. Each arm compares a correct proposal and an
intentionally wrong proposal against the **same sealed optical capture**; the
second comparison makes no model call. An incorrect, ambiguous, truncated or
otherwise incomplete optical reading fails the functional check. There are no
automatic retries.

## Run in the supplied Docker service

Prepare and verify both pinned Qwen bundles using the normal model preparation
command first. Use the existing Compose configuration and its shared data,
reports and model-lock mounts. Stop the HTTP model owner before this diagnostic:

```sh
docker compose stop ocr
docker compose run --rm --no-deps --entrypoint python ocr \
  /app/scripts/visual_acceptance.py \
  --models /models --output /reports/visual-target-001 --seconds 600
```

The report and constructed fixtures are written to the existing `harness_reports`
volume. The output directory must be new. The script updates `report.json`
atomically during preflight, each generation, comparison and shutdown, retaining
partial results on failure or timeout. It does not publish or send the report.
The script needs no API tokens and records no environment credentials.

After inspecting the report, restart the service when desired:

```sh
docker compose start ocr
```

For an explicit CPU check in an installed inference environment, select the
small model. CPU always uses float32:

```sh
python scripts/visual_acceptance.py --device cpu --qwen qwen-0.8b \
  --models ./models --output ./reports/visual-cpu-001 --seconds 600
```

`--qwen` may select either pinned model or both. The total budget is finite,
defaults to 600 seconds and is capped at 3600 seconds. Each supervised call gets
only the remaining budget, including worker startup, loading and generation.
The Supervisor admission and compute leases prevent another model owner in the
same configured lock namespace from overlapping this run. The diagnostic uses
the production lease mechanism; it does not touch a research `active_run.lock`.
Keep `HARNESS_LOCK_DIR` shared with other containers using the same GPU.

## What a passing report proves

A successful CLI exit requires `status: "pass"`, confirmed model-owner shutdown
and `target_runtime_accepted: true`. `default_both_models_accepted` is true only
when both default models completed the check. A CPU or single-model result does
not assert that the other model or CUDA executed successfully.

The report includes the pinned model revision and bundle SHA, verified
processor/tokenizer file SHAs, source/raster/crop and actual-pixel SHAs, source
coordinates and transform proof, processor class, actual image grid and pixel
tensor SHA, constrained schema SHA, effective EOS, fresh-generation flag and
effective parameter dtypes. All model-facing region identifiers are fixed
positional aliases. Original region IDs, paths, field names, proposed strings
and scorer labels are absent from the blind model prompt. Both original-file
and crop pixels are replayed and checked before the model consumes them.

This is a **synthetic functional self-test**, not an independently annotated
quality benchmark. It provides no production accuracy estimate, no calibrated
acceptance threshold and no evidence that Qwen text and Qwen vision errors are
statistically independent. `quality_qualified`,
`automatic_acceptance_authority` and `statistical_independence_claimed` remain
false in every report. Tests with injected backends cannot mark actual target
execution as accepted.

## Production verification behavior

The first stage transcribes the full approved ROI literally, without a proposed
answer. The second stage compares the sealed optical literal deterministically
using `exact-v1`. Spaces, punctuation, digits and line breaks are preserved;
there is no substring acceptance, digit stripping, spelling repair or silent
replacement of a prior OCR value or human correction. A full line containing a
label is compared as a full line. It does not become a value-only ROI through
text matching.

`correct` means optical literal concordance within that comparison scope. It
does not verify the field's semantic role or authorize acceptance. Disagreement
and abstention remain in document history and request review. Human corrections
remain explicit revisions. Existing validation, conflicts, runtime binding and
risk-policy checks continue to apply. `HARNESS_VISUAL_VERIFIER=1` enables the
bounded production hook; the Compose candidate enables it by default.
