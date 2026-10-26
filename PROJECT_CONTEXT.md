# Project context — handoff

BTech Project CSQ301 · **Efficient algorithms for machine unlearning (theory and implementation)**.
This file summarises earlier work on the project so a new assistant session can continue without the old chat.
Last updated 2026-10-24.

## Status

- The professor shared 11 papers in four batches (`papers/1` to `papers/4`). No project direction is fixed yet, but folder 4 (MUFSC) points strongly at exact unlearning for federated fair clustering.
- Each paper was read in full (including appendices) and turned into structured notes, then merged into one knowledge graph.
- The graph is live at **https://sandstrompl.github.io/BTP/** (`index.html` in this repo). It has three tabs: Network, Overlap matrix, Directions (`#matrix`, `#directions` open them directly).
- A copy is also published as a private Claude artifact: https://claude.ai/artifact/Ec4CSFyCLJL95jVV2XouvT
- Repo: https://github.com/SandstromPL/BTP (GitHub Pages serves `main`, root).

## Files

| Path | What it is |
|---|---|
| `index.html` | The interactive graph. All data is embedded in it (`const DATA = …`). Loads D3 7.9.0 and Google Fonts from CDNs. |
| `knowledge-graph/knowledge-graph.json` | Merged graph: `papers`, `nodes` (437 concepts with `type`, `description`, `papers`, per-paper `uses` notes), `links` (1106), `citations` (13), `related`, `pairShared`, `paperOverlap`, `threads`, `directions`. |
| `knowledge-graph/per-paper/<id>.json` | Detailed notes per paper: summary, problem, core idea, method steps, definitions, theorems, complexity, experiments with numbers, limitations, open questions, relevance to the project. **Best source for paper details.** |
| `papers/` | The PDFs. Note: `papers/2/1_2312.07420v1.pdf` is misnamed. It is UnCLe (arXiv 2509.17530), not FairSISA. |

## The 11 papers

| Folder | Id | Paper | Unlearning |
|---|---|---|---|
| 1 | bourtoule2021-sisa | **SISA**: Machine Unlearning (IEEE S&P 2021) | exact |
| 1 | kadhe2023-fairsisa | **FairSISA**: fairness of SISA ensembles for LLMs (NeurIPS'23 workshop) | exact + fairness post-processing |
| 1 | wen2026-fedshard | **FedShard**: federated unlearning with efficiency and performance fairness (AAAI-26) | exact |
| 2 | adhikari2025-uncle | **UnCLe**: unlearning for continual learning, hypernetwork (arXiv 2509.17530) | approximate, data-free |
| 2 | rachapudi2026-bidlora | **BID-LoRA**: continual learning + unlearning with three LoRA adapters (arXiv 2604.12686) | approximate |
| 2 | chen2025-meco | **MeCo**: federated continual learning for industrial edge (IEEE TNSE 2025) | none |
| 3 | ginart2019-deletion | **Making AI Forget You**: Q-k-means, DC-k-means (NeurIPS 2019) | exact |
| 3 | pan2023-mufc | **MUFC**: machine unlearning of federated clusters, SCMA (ICLR 2023) | exact |
| 3 | gupta2024-ffc | **p-FClus**: fair federated clustering via personalisation (arXiv 2407.04302) | none |
| 3 | makkar2023-mfc | **MFC / MFC_H**: multishot federated clustering (ECAI 2023) | none |
| 4 | reddy2026-mufsc | **MUFSC**: machine unlearning for federated socially fair k-means (arXiv 2610.13717, Oct 2026; IIT Madras; code github.com/arareddy/mufsc) | exact (same-seed replay) |

Folder themes: 1 = sharding and fairness, 2 = continual learning meets unlearning, 3 = clustering and federated k-means, 4 = fair federated clustering unlearning (joins 1 and 3).
Explicit citations among them: SISA→Ginart; FairSISA→SISA, Ginart; FedShard→SISA; BID-LoRA→UnCLe (baseline); MUFC→Ginart (DC-k-means baseline), SISA; p-FClus→MFC (baseline), MUFC (future work); MUFSC→MUFC (builds on), Ginart, SISA, FedShard.
MFC and p-FClus come from the same IIT Ropar group (Jain, Gupta). MUFSC shares an author (Saurav Prakash) with MUFC. MUFSC's "Gupta et al., 2025" is a different paper (ARCANE), not p-FClus.

## Main interpretation

1. **Two camps.** Exact unlearning (SISA, Ginart, MUFC, FedShard) proves the model after deletion has the same distribution as retraining without the data. Approximate unlearning (UnCLe, BID-LoRA) only measures forgetting (forget/retain accuracy, membership-inference attacks). The "theory" goal fits the exact camp.
2. **Four mechanisms for cheap exact deletion.**
   - Isolation: SISA shards and slices, FedShard's client merge tree, DC-k-means tree. Theory tool: SISA's expected retraining cost formulas.
   - Stability: Q-k-means quantized centroids rarely change. Theory tool: bound on the probability that a deletion changes the output.
   - Lazy resampling: MUFC redoes k-means++ seeding only if a removed point was a chosen seed. Theory tool: conditional-probability exactness proof (Lemma 5.1); expected removal time doesn't grow with n.
   - Same-seed replay: MUFSC reruns training with the same seed, reusing every cached (client, group) summary the deletion did not touch; exact integer statistics make it identical to retraining. Theory tool: replay-exactness induction (Thm 5). Catch: seeded k-means++ choices can shift after one deletion and waste linear work (Lemma 10, Prop 11).
3. **Trade-offs.**
   - Accuracy: SISA's speedup only holds while requests stay below about 3× the number of shards, and it loses about 19 points of top-5 accuracy on ImageNet.
   - Fairness: shards can worsen equalized odds (FairSISA), and unlearning cost falls unevenly on clients (FedShard).
   - Storage: per-slice checkpoints are expensive.
   - Setting: in federated learning you usually delete whole clients; in continual learning you get relapse and spill.
   - Granularity: MUFSC's whole-client deletion is 35.8× faster than retraining with no refinement rounds (17.8× with ordinary replay; 2.4× and 1.7× with one and two rounds), but single-record deletion is only about 1.06×.
4. **The gap.** Folder 3 paired two exact clustering-unlearning papers with two federated clustering papers that have no deletion. Folder 4 (MUFSC) fills part of that gap: exact unlearning for federated fair k-means. Still open: fast single-record deletion (MUFSC ≈ 1.06×), client-level fairness under deletion, and deletion for MFC/p-FClus. Confirm the intended direction with the professor.

## Candidate directions (most → least theory-friendly)

1. **Fast per-record deletion for federated fair k-means (extend MUFSC).** Build a deletion-stable seeded k-means++ sampler (e.g. per-record random keys, the paper's own suggestion) so most deletions leave chosen centres unchanged; combine with DC-k-means partitions or Q-k-means quantization for refinement rounds. Theory: bound the expected per-record unlearning cost (MUFC Thm 5.4 / Ginart amortized style) while keeping same-seed exactness. Laptop; MUFSC's public code is the baseline.
2. **Exact unlearning for multi-round federated clustering (MFC).** Checkpoint the server state per round and quantize uploaded centres; MUFSC's exact replay of refinement rounds is a template. Theory: bound the expected number of rounds a deletion forces you to recompute. Laptop.
3. **Exact unlearning for multi-round federated clustering** (original pre-MUFSC card, kept verbatim at the user's request). One-shot MUFC is cheap to unlearn, but MFC's later rounds tie every client to every other through the global centres. Approach: save the server state after every round and quantize the uploaded centres (as Q-k-means does), so most deletions change nothing. Theory: prove a bound on the expected number of rounds a deletion forces you to recompute. Laptop. Builds on MFC, MUFC, Making AI Forget You.
4. **Client-fair clustering under unlearning.** MUFSC keeps demographic-group fairness exact; nobody checks whether deleting one client hurts the remaining clients unevenly (p-FClus/MFC's fairness, FedShard's Mp/Me). Laptop.
5. **Make MUFC fully incremental.** Its server reruns k-means++ and Lloyd's from scratch; clients skip Lloyd's. Extend the Lemma 5.1 argument.
6. **Cost-fair SISA:** spread unlearning cost evenly across shards/slices. Mostly empirical; needs a GPU.
7. **Federated continual unlearning (MeCo + UnCLe/BID-LoRA).** Little theory available; heavy deep learning.

**Recommendation:** start with 1 (fallback: 2). First steps:
1. Read MUFSC closely, alongside MUFC and Ginart (the three exact clustering-unlearning papers).
2. Run MUFSC's code and reproduce its whole-client (35.8× at T = 0) and single-record (≈1.06×) results.
3. Implement the baselines: retraining, Q-k-means, DC-k-means, MUFC's client-side seeding.
4. Prototype a deletion-stable k-means++ sampler; measure how often its choices change after one deletion.
5. Prove an expected per-record cost bound; compare against MUFSC's direct replay on Adult, Bank, Credit and ACS.

## Inconsistencies found inside the papers

- MUFC: claims ~84× average speedup, but its figure legends average about 48×. The server always retrains fully.
- FedShard: claims 6.2× over retraining, but the main tables show about 3.7–4.3×. It has no convergence or fairness bound. Exactness may break if shard-merging decisions used the leaving client's updates.
- FairSISA: never processes an actual deletion request. Its GPT-2 figures disagree with each other.
- Ginart: ">100×" is the average over both methods; per method it's about 226× for Q-k-means and about 30× for DC-k-means (2.2× on MNIST).
- SISA: the often-quoted (R+1)S/2 speedup is a best case. Under uniform requests, slicing gives at most 1.5× in expectation.
- UnCLe: membership-inference success is about 50% for almost every method, so it barely separates them. Its Theorem 1 conflicts with its own claim that unlearning restores a random initialisation.
- BID-LoRA: switching off the forget adapter before merging brings forget accuracy back to about 70%. Eq. 1 and Algorithm 1 use opposite signs for the forget term. It needs a 10% replay buffer of retained data.
- MFC: clustering cost rises after the first round on heterogeneous data. Its proofs only cover the initial local centres.
- p-FClus: has no theorems. Its "10^5×" claim is about 6–10× in its own table. Personalised clients are compared against baselines that share one set of centres.
- MeCo: its forgetting numbers for FedAvg are inconsistent across two tables. The "meta-learning" step is a first-order alternating update, not true MAML.
- MUFSC: the 35.8× headline comes from a separate compact implementation (ordinary direct replay at T = 0 is 17.8×); timings are in-memory request times for one preselected client per dataset (Bank's 115× reflects only 20 clients). Certificates almost never help (they give up at round zero in 174/175 and 92/95 trials). Groups are SEX only for Adult and ACS (Bank: married vs not; Credit: education). The authors state heavy generative-AI use, including for proofs (some checked in Lean 4).

## Network view (ring layout, implemented 2026-10-24)

The old force-directed hairball (437 concepts, 1106 links at once) was replaced:
- Papers sit on a fixed ring in folder arcs; the order is optimised so papers sharing the most concepts are neighbours.
- Concepts shared by 2+ papers sit inside the ring at the average position of their papers; each paper's own concepts are
  collapsed into a "+n" badge. Datasets, metrics and models are hidden by default (legend chips toggle them).
- No links at rest except dashed citation arrows; hovering shows a node's spokes.
- Clicking a paper fans its own concepts out in an arc with outward labels; clicking a concept lights up its papers and linked concepts.
- Compare mode (paper panel "Compare with…" chips, or the Overlap matrix) shows only two papers and the concepts they share.
- "Show: Everything" fans out every paper's own concepts. Labels are placed by priority and skipped when they would overlap.
