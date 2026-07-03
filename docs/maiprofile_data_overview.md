# MAI Profile Data Overview for DSpark

This document summarizes the MAI Profile raw prompt data under:

```text
$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/raw_data/20260615
```

The raw files are prompt-only JSONL records with the common schema:

```json
{
  "user_id": "...",
  "prompt_hash": "...",
  "prompt_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

There are no assistant responses yet. Before DSpark target-cache creation, these prompts must be sent through the target model to generate assistant messages.

## Token Statistics

Token counts below were produced by `scripts/data/analyze_maiprofile_prompts.py` using explicit DeepSpec Gemma4-style prompt rendering:

```text
<|turn>user
<system prompt + user prompt>
<turn|>
<|turn>model
```

| layer | rows | MiB | p50 tokens | p95 tokens | p99 tokens | max tokens | assistant records | >2048 | >4096 | >8192 | >16384 | >32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| layer1_actual | 6,614 | 45.11 | 1,531 | 2,584 | 3,211 | 7,760 | 0 | 1,356 | 5 | 0 | 0 | 0 |
| layer1_delta | 7,675 | 254.507 | 5,927 | 22,886 | 28,826 | 46,928 | 0 | 6,070 | 4,674 | 3,093 | 1,631 | 25 |
| layer1_intent | 6,689 | 51.903 | 1,715 | 2,156 | 2,475 | 3,724 | 0 | 631 | 0 | 0 | 0 | 0 |
| layer2_coarse_interest | 8,887 | 119.263 | 2,987 | 4,387 | 5,182 | 7,049 | 0 | 7,440 | 790 | 0 | 0 | 0 |
| layer2_temporal | 6,626 | 55.04 | 1,955 | 2,725 | 3,284 | 7,436 | 0 | 2,762 | 15 | 0 | 0 | 0 |
| layer3_commercial_interests | 8,875 | 224.681 | 5,627 | 8,133 | 9,203 | 13,018 | 0 | 8,875 | 7,500 | 414 | 0 | 0 |
| layer3_persona | 8,867 | 139.961 | 3,209 | 5,423 | 6,421 | 10,589 | 0 | 6,823 | 2,258 | 5 | 0 | 0 |
| layer3_seasonality | 8,865 | 23.266 | 583 | 852 | 993 | 1,558 | 0 | 0 | 0 | 0 | 0 | 0 |
| layer4_biography | 8,907 | 143.0 | 4,118 | 7,815 | 9,795 | 13,558 | 0 | 7,215 | 4,490 | 325 | 0 | 0 |
| layer4_commercial_preference | 8,309 | 100.446 | 2,859 | 3,930 | 4,627 | 6,367 | 0 | 8,309 | 287 | 0 | 0 | 0 |

Total non-empty prompt rows: **80,314**.

## Layer Semantics

### `layer1_delta` — Layer 1: Delta Interest Extraction

Extracts user interests from denoised interaction signals. Each interest should group signals about the same underlying entity, product, destination, concern, person, or event, with topics as facets of that entity.

Input shape:

- User demographics/facts when available.
- Today's denoised signals/events.
- The model is expected to output newly extracted fine-grained interests and topics.

DSpark note:

- This is the longest layer by far: p50 5.9k tokens, p95 22.9k, max 46.9k.
- A small `max_length` such as 2k/4k is not representative for this layer.
- This layer needs special handling: longer-context target cache, short-prompt bucket only, or separate analysis before full DSpark training.

### `layer1_actual` — Layer 1: Interest Activity Description

Given extracted interests, topics, and evidence, generates an `actual_activity` field for each interest. The output should be a one-sentence factual summary of what the user actually did, grounded in observed evidence.

Input shape:

- Delta interests with topic evidence.
- The model writes factual activity summaries.

DSpark note:

- p50 1.5k tokens, p95 2.6k, max 7.8k.
- Good candidate for an initial DSpark pilot. A 4k context covers most but not all samples; 8k covers almost all based on max.

### `layer1_intent` — Layer 1: Interest Intent Inference

Given interests, actual activity summaries, and supporting topics/evidence, generates an `inferred_intent` for each interest. The intent should describe the likely user goal, need, or decision behind the observed activity.

Input shape:

- Delta interests.
- Existing `actual_activity` summaries.
- Topics/evidence with empty intent fields to fill.

DSpark note:

- p50 1.7k, p95 2.2k, max 3.7k.
- Very suitable for early DSpark pilot with 4k context.

### `layer2_coarse_interest` — Layer 2: Coarse Interest Clustering

Clusters fine interests into durable coarse parent interests. Existing clusters act as anchors; thin clusters and ungrouped fine interests are merged or grouped into stable domains. The output is a patch-like set of changed clusters.

Input shape:

- Existing coarse clusters.
- Thin clusters.
- Ungrouped fine interests with confidence, temporal type, and topics.

DSpark note:

- p50 3.0k, p95 4.4k, max 7.0k.
- 4k truncates a meaningful tail; 8k is more appropriate for a robust pilot.

### `layer2_temporal` — Layer 2: Temporal Interest Classification

Classifies each interest/activity by temporal pattern based on the nature of the interest itself, not just frequency or recency of observed behavior.

Input shape:

- Interests with aggregation stats.
- Existing/previous temporal labels where available.

DSpark note:

- p50 2.0k, p95 2.7k, max 7.4k.
- Good candidate for early pilot; 4k covers most samples.

### `layer3_commercial_interests` — Layer 3: Commercial Interest Enrichment

Determines whether each active interest is commercial and enriches commercial interests with product/service purchase intent details. It distinguishes true product/service evaluation from news, investment research, career interest, or general brand awareness.

Input shape:

- Active interests with actual activity, inferred intent, topics, and sources.
- The model outputs commercial flags and commercial metadata.

DSpark note:

- p50 5.6k, p95 8.1k, max 13.0k.
- 4k is too short for the median. 8k covers many but not all; 16k likely covers nearly all.

### `layer3_persona` — Layer 3: Persona Prompt

Synthesizes an interest persona for each active interest, capturing motivations, preferences, and behavioral patterns grounded in topics, signals, and intent patterns.

Input shape:

- User facts.
- Active interests with actual activity, inferred intent, and topics.
- The model writes concise per-interest persona descriptions.

DSpark note:

- p50 3.2k, p95 5.4k, max 10.6k.
- 8k is likely a better pilot target than 4k.

### `layer3_seasonality` — Layer 3: Seasonality Prompt

Determines the inherent seasonality of each active interest. Categories include NotApplicable, MultiYear, Annually, Quarterly, Monthly, and Weekly.

Input shape:

- A list of active interest names.
- The model assigns seasonality categories.

DSpark note:

- Shortest layer: p50 583, p95 852, max 1.6k.
- Easy to support with small context, but output may be relatively short; DSpark decode-speed benefit should be validated by generated response length.

### `layer4_biography` — Layer 4: Biography Prompt

Infers user-level life stage and writes a concise biography from behavioral patterns and active interests. It should avoid unsupported speculation and summarize who the user appears to be based on evidence.

Input shape:

- User facts.
- Active interests with actual activity, inferred intent, persona, confidence, counts, dates, and sources.
- The model outputs life stage plus a biography.

DSpark note:

- p50 4.1k, p95 7.8k, max 13.6k.
- 4k truncates around median; 8k covers most but not all.

### `layer4_commercial_preference` — Layer 4: User-Level Commercial Preferences

Synthesizes user-level commercial preferences from all commercial interests, biography, and life stage. Also generates cross-interest predicted commercial queries that combine signals from multiple interests.

Input shape:

- Biography and life stage.
- Commercial interests with personas, brands, retailers, products, topics, and predicted queries.
- The model outputs user-level commercial preference summaries and cross-interest predicted queries.

DSpark note:

- p50 2.9k, p95 3.9k, max 6.4k.
- Good pilot candidate; 4k covers most, 8k covers nearly all.
