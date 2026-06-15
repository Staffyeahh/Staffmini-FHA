# V14.3 Anchor-Predictive FHA Test - 2026-06-15

## Formaal

V14.3 tester en intern model-aendring, ikke en ekstern repair- eller eval-addon.

Hypotesen var, at FHA-broen bliver staerkere, hvis hvert anchor ikke kun komprimerer den aktuelle 16-token blok, men ogsaa tvinges til at baere information, der kan forudsige naeste anchor. Det skulle goere Gating Anchor mere semantisk og mindre som simpel pooling.

## Design

Base: V14.2 Gated Delta Anchor.

Aendring:

- `fha.anchor_type: gated_delta`
- `fha.anchor_prediction_weight: 0.05`
- Hvert `anchor_t` sendes gennem en intern predictor.
- Predictor traenes med cosine loss mod `anchor_{t+1}.detach()`.
- Loss laegges internt oven i LM-loss: `loss = lm_loss + 0.05 * anchor_prediction_loss`.

Dette ligger i modellen/traeningen. Der er ingen ekstern viden, retrieval, repair inference eller post-processing.

## Implementering

Nye/aendrede filer:

- `configs/model_config_v14_3_anchor_predictive_50m_sample25.yaml`
- `src/staffyeahh_mini/configuration_mini_step_moe.py`
- `src/staffyeahh_mini/modeling_mini_fha_v14.py`
- `scripts/train.py`

Samtidig blev en vigtig config-fejl rettet:

- V14 baseline bruger nu eksplicit `fha.anchor_type: mean`.
- V14.1 bruger nu eksplicit `fha.anchor_type: multi_slot`.
- V14.2/V14.3 bruger nu eksplicit `fha.anchor_type: gated_delta`.

Det betyder, at fremtidige baseline-sammenligninger bliver rene. Den tidligere V14 baseline vs V14.2 1500-step sammenligning skal laeses med caveat, fordi baseline-konfigurationen foer rettelsen kunne route til gated-delta-adfaerd via default-logikken.

## Modelstoerrelse

Kommando:

```powershell
python scripts/count_model_params.py configs/model_config_v14_3_anchor_predictive_50m_sample25.yaml
```

Resultat:

| metric | value |
|---|---:|
| total params | 47.51M |
| embedding params | 11.63M |
| attention/FHA params | 5.90M |
| other params | 29.97M |

## Smoke Test

Kommando:

```powershell
python scripts/smoke_test.py --config configs/model_config_v14_3_anchor_predictive_50m_sample25.yaml --seq-len 512 --batch-size 4
```

Resultat:

- logits shape: `(4, 512, 30294)`
- initial loss: `10.4429`
- smoke test passed

## 300-Step Traening

Kommando:

```powershell
python scripts/train.py --model-config configs/model_config_v14_3_anchor_predictive_50m_sample25.yaml --train-config configs/train_config_v14_fha_lite_300.yaml --token-bin data/tokenized/cosmopedia_mix_100m.bin --output-dir outputs/v14_3_anchor_predictive_scratch_300
```

Checkpoint:

```text
outputs/v14_3_anchor_predictive_scratch_300/step_0000300/model.pt
```

Udvalgte traeningspunkter:

| step | total loss | lm loss | anchor pred loss | tok/s |
|---:|---:|---:|---:|---:|
| 1 | 10.4606 | 10.4108 | 0.996 | 24,742 |
| 50 | 7.2446 | 7.2422 | 0.047 | - |
| 100 | 6.5606 | 6.5545 | 0.122 | - |
| 150 | 6.2288 | 6.2214 | 0.147 | - |
| 200 | 5.8326 | 5.8226 | 0.199 | - |
| 250 | 5.6423 | 5.6313 | 0.220 | - |
| 300 | 5.4203 | 5.4099 | 0.207 | 50,228 |

Runtime var cirka 3m41s. Max VRAM var cirka 10.54GB.

## Held-Out Causal Eval

Kommando:

```powershell
python scripts/eval_causal_suite.py --model-config configs/model_config_v14_3_anchor_predictive_50m_sample25.yaml --checkpoint outputs/v14_3_anchor_predictive_scratch_300/step_0000300/model.pt --token-bin data/tokenized/cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-json outputs/v14_3_anchor_predictive_scratch_300/eval_causal_300.json --output-md outputs/v14_3_anchor_predictive_scratch_300/eval_causal_300.md
```

Resultat:

| seq | loss | ppl | top-1 | tok/s |
|---:|---:|---:|---:|---:|
| 512 | 5.3781 | 216.6 | 22.92% | 22,834 |
| 1024 | 5.4394 | 230.3 | 22.21% | 45,010 |
| 2048 | 5.4912 | 242.6 | 21.79% | 83,800 |

## Sammenligning Mod V14.2 300

Tidligere V14.2 Gated Delta Anchor ved 300 steps:

| model | seq 512 loss | seq 1024 loss | seq 2048 loss |
|---|---:|---:|---:|
| V14.2 gated delta | 5.3406 | 5.4031 | 5.4594 |
| V14.3 anchor-predictive | 5.3781 | 5.4394 | 5.4912 |

V14.3 er daarligere paa alle tre eval-laengder i denne 300-step test.

## Real-Life Generation Test

Sampling:

- temperature: `0.8`
- top-k: `40`
- max new tokens: `60`

Prompts:

- `The capital of France is`
- `If a glass falls from a table, it will usually`
- `A good reason to save money is`
- `In Python, a function is defined with`

Kvalitativt resultat:

- Output er ikke bare tilfaeldige tokens.
- Den har laert en del generel tekstform, afsnitsrytme og hyppige fraser.
- Den svarer stadig ikke robust paa fakta, kausalitet eller kode.
- Eksempel: `The capital of France is` fortsaetter med generisk tekst om "importance" i stedet for `Paris`.
- Eksempel: `If a glass falls from a table...` rammer ikke klart "break/shatter".

Det er acceptabelt for kun 300 steps, men ikke et tegn paa, at V14.3 er en bedre retning end V14.2.

## Konklusion

V14.3 virker mekanisk: aux-loss traener, checkpointet loader, eval koerer, og modellen genererer tekst.

Men hypotesen er ikke bekraeftet ved 300 steps. Den interne next-anchor cosine loss goer ikke modellen bedre paa held-out LM-loss, og generationen viser ikke tydeligere semantik end forventet.

Min vurdering: lad vaere med at skalere praecis denne V14.3-form til 1500/15k endnu. Hvis vi vil forfoelge ideen, boer naeste variant aendre selve supervisionen:

- Predict naeste bloks token-distribution eller top-k noegleord i stedet for naeste anchor-vektor.
- Brug lavere weight, fx `0.01`, eller warmup efter 200-500 steps.
- Maal anchor-kvalitet direkte med en lille held-out probe: kan anchor forudsige naeste blok bedre end mean/gated-delta uden at skade LM-loss?

Aktuel bedste praktiske FHA-retning er fortsat V14.2 Gated Delta Anchor som baseline, men med den nye `anchor_type`-rettelse boer baseline koeres rent igen, foer vi tager en stor beslutning.
