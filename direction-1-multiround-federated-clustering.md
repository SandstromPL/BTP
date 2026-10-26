# Direction 1: exact unlearning for multi-round federated clustering

BTech Project CSQ301 · Efficient algorithms for machine unlearning (theory and implementation)
Written 2026-10-26 from the knowledge graph (`knowledge-graph/`) and the per-paper notes (`knowledge-graph/per-paper/`).

This file has three parts:

- **Part A**: the Directions page as it was *before* the folder 4 paper was added (10 papers; git commit `84b09c7`).
- **Part B**: direction 1 of that page in depth: the problem, what to borrow from the existing papers, a concrete algorithm sketch, the theory to aim for, and an evaluation plan.
- **Part C**: what the new paper in folder 4 (MUFSC, arXiv 2610.13717) adds to that same direction, and how the direction changes because of it.

Everything marked *proposed* or *conjecture* is a research plan, not a result from any paper.

---

## Part A — The Directions page before folder 4

> What the ten papers add up to for a project on efficient unlearning algorithms with both theory and implementation. This is an interpretation across the papers, not something any one of them says.

### Two camps, and the theory lives in one

| Camp | What it means | Papers |
|---|---|---|
| **Exact unlearning** | After a deletion, the model has the same distribution as one retrained without that data. These papers prove it, then show it costs less than retraining. | SISA, Making AI Forget You, MUFC, FedShard |
| **Approximate unlearning** | Forgetting is measured, not proved: forget and retain accuracy, and membership-inference attacks. Both papers expose weak spots. In BID-LoRA, switching off the forget adapter before merging brings the forgotten classes back to about 70% accuracy. | UnCLe, BID-LoRA |

A project with "theory" in its title fits the exact camp best.

### Three ways to make a deletion exact and cheap

| Mechanism | Idea | Papers | Reusable tool |
|---|---|---|---|
| **Isolation** | Each data point influences only one part of the model, so only that part is retrained. | SISA, FedShard, Making AI Forget You (DC-k-means) | Expected retraining cost formulas (SISA) |
| **Stability** | The model rarely changes when one point leaves, so most deletions cost nothing. | Making AI Forget You (Q-k-means) | Bound on the chance that a deletion changes the output |
| **Lazy resampling** | Redo work only if the deleted point was actually used. | MUFC | Conditional-probability exactness proof (MUFC Lemma 5.1) |

A new exact algorithm will almost always be one of these three, or a mix of them.

### Efficiency is never free

- **Accuracy:** SISA's speedup only holds while deletion requests stay below about 3× the number of shards. On ImageNet it loses about 19 points of top-5 accuracy.
- **Fairness:** more shards can make predictions less fair (FairSISA). In federated unlearning, some clients pay much more of the cost than others (FedShard).
- **Storage:** a checkpoint for every slice is expensive. FairSISA dropped slicing for LLMs because of it.
- **Setting:** in federated learning you usually delete a whole client. In continual learning, forgotten tasks come back (relapse) and kept tasks get damaged (spill).

### The gap the papers leave

The latest folder [*note: folder 3 at the time*] pairs two exact clustering-unlearning papers (Ginart, MUFC) with two federated clustering papers that have no deletion at all (MFC, p-FClus), both from the same IIT Ropar group. That reads like a pointer toward adding unlearning to federated clustering. This is a guess; confirm it with your professor.

### Candidate directions (most → least theory-friendly)

| # | Direction | Theory | Runs on |
|---|---|---|---|
| **1** | **Exact unlearning for multi-round federated clustering** | Strong | Laptop |
| 2 | Unlearning for personalised fair clustering | Strong | Laptop |
| 3 | Make MUFC fully incremental | Medium | Laptop |
| 4 | Cost-fair SISA | Light | GPU |
| 5 | Federated continual unlearning | Weak | GPU |

1. **Exact unlearning for multi-round federated clustering.**
   - **Gap:** one-shot MUFC is cheap to unlearn, but MFC's later rounds tie every client to every other through the global centres.
   - **Approach:** save the server state after every round and quantize the uploaded centres (as Q-k-means does), so most deletions change nothing.
   - **Theory:** prove a bound on the expected number of rounds a deletion forces you to recompute.
   - **Builds on:** MFC, MUFC, Making AI Forget You.
2. **Unlearning for personalised fair clustering.**
   - **Gap:** no paper measures whether deleting one client hurts the remaining clients unevenly.
   - **Approach:** remove a client by dropping its centres and re-clustering. Remove a point using stable or quantized local centres. Track the cost spread across clients before and after.
   - **Theory:** bound how far personalised centres move after a deletion, and when re-personalisation can be skipped.
   - **Builds on:** p-FClus, Making AI Forget You, FedShard.
3. **Make MUFC fully incremental.**
   - **Gap:** MUFC's server reruns k-means++ and Lloyd's from scratch after every deletion, and clients skip Lloyd's to keep unlearning cheap.
   - **Approach:** apply the reseeding trick on the server as well, or allow Lloyd's iterations on clients while keeping exactness.
   - **Theory:** extend the Lemma 5.1 exactness argument and the expected removal-time bound.
   - **Builds on:** MUFC, Making AI Forget You.
4. **Cost-fair SISA.**
   - **Gap:** in SISA, what a deletion costs depends on which shard and slice the point landed in.
   - **Approach:** assign shards and slices so the cost is spread evenly, using FedShard's efficiency-fairness metric.
   - **Theory:** mostly empirical, with SISA's cost formulas as the analysis tool.
   - **Builds on:** SISA, FedShard, FairSISA.
5. **Federated continual unlearning.**
   - **Gap:** MeCo learns tasks across clients but cannot forget any of them.
   - **Approach:** add a forgetting step using UnCLe's hypernetwork or BID-LoRA's adapters, and record what each client contributed to the knowledge pool.
   - **Theory:** little to build on; heavy deep-learning work.
   - **Builds on:** MeCo, UnCLe, BID-LoRA.

**Recommendation (then):** start with direction 1 or 2. k-means is simple enough for real theorems and runs on a laptop. It already has:
- exact baselines: retraining, DC-k-means, Q-k-means, MUFC;
- metrics: deletion time, k-means cost, cost spread across clients;
- datasets that several papers use: Covtype, Postures, Adult, Bank Marketing, synthetic Gaussian mixtures.

First steps:
1. Implement k-means++ and Lloyd's algorithm.
2. Implement the exact baselines: Q-k-means, DC-k-means and MUFC's client-side seeding.
3. Implement MFC, and p-FClus if you take direction 2.
4. Reproduce deletion time against retraining on two or three of the shared datasets.
5. Add a deletion step to MFC and compare it with MUFC and with retraining.

---

## Part B — Direction 1 in depth (as of the 10-paper set)

### B.1 The problem in one paragraph

MFC (Makkar et al., ECAI 2023) is a federated k-means method that runs several communication rounds ("shots"). The rounds are why it clusters well in both heterogeneous and homogeneous data regimes without knowing each client's number of clusters k_z. They are also what makes deletion expensive. Every client's upload in round t depends on the global centres from round t−1, which depend on every other client's earlier uploads. So one deleted record ripples through the whole run. Today the only exact way to delete from MFC is to rerun every round from scratch. Direction 1 asks for an MFC-style multi-round federated k-means where most deletions are exact *and* cheap, with a provable bound on the expected cost.

### B.2 How MFC works, and where deletion breaks it

Setup: Z clients; client z holds X^(z) in R^d drawn from k_z ≤ k of the k global clusters (k_z unknown); the target k is known (k = 10 in the paper's experiments).

| Step | What happens | What a deletion touches |
|---|---|---|
| Client init (Alg. 1, Awasthi–Sheffet) | Project local data onto its top-k singular vectors, run approximate k-means, keep only points ≥ 3× closer to their centre than to any other, run Lloyd to convergence → k local centres θ^(z) | A deleted record changes the SVD subspace, the pruning and Lloyd, so θ^(z) can move |
| Init upload | MFC sends all k local centres (k·d floats); MFC_H sends one random centre | The changed θ^(z) enters the server pool |
| Server init | Pool S = all received centres; Lloyd on S → global centres μ; broadcast | μ changes for **everyone** |
| Each shot: client update | Reassign local points to the nearest μ_j; recompute local centres | Every client's assignment now depends on the changed μ |
| Each shot: upload | Send only θ_max^(z), the local centre with the highest cluster cost (d floats) | The arg-max is a discrete choice that can flip after a small change |
| Each shot: server update | S ← S ∪ {θ_max^(1..Z)} (the pool keeps growing); Lloyd on S → new μ; broadcast | Every later round inherits the change |

Without unlearning machinery, exactly removing a client means dropping its centres from S at *every* shot and rerunning all shots, because its uploads changed every later μ and therefore every other client's later θ_max. That costs as much as retraining. A one-shot scheme (k-FED, MUFC) only has to redo the server aggregation.

MFC's guarantees are worth knowing before building on it:
- **Theorem 1:** under well-separability (Assumption A2), enough points per local sub-cluster (A1) and a Gaussian-mixture model, every *initial* local centre is within c·√k·‖X − C‖₂ / √(n_i^(z)) of its optimal global centre, with c = 50/p′ + α.
- **Lemma 6:** MFC_H needs about k² log k clients. This is an expected-value coupon-collector argument.
- **No guarantees for later rounds.** There's no convergence or approximation guarantee for the final centres, and the paper's own plots show cost *rising* after the first shot on heterogeneous Adult and Bank.

### B.3 The exactness target

Use the definition from *Making AI Forget You* (Ginart et al.), which SISA also uses:

> A deletion operation R is exact if R(A(D), D, i) has the same distribution as A(D \ {i}): the model after deleting point i is distributed exactly as a model retrained without it.

The measure of efficiency is the **amortized runtime** over a stream of m deletions. Remark 3.1 gives the lower bound: any sequential learner needs Ω(n/m) amortized time per request, because training alone costs Ω(n).

### B.4 The toolbox from the existing papers

**From MUFC (Pan et al., ICLR 2023): the one-shot baseline and lazy resampling.**
- **How it trains:** clients run only k-means++ *seeding* (no Lloyd), quantize the K seeds to grid bins, and send a sparse count vector through SCMA (secure aggregation via Reed–Solomon coding). The server samples points from the counted bins and runs full k-means++ plus Lloyd.
- **Client deletion (Alg. 3):** keep the seeds chosen before the first deleted seed and resample the rest. **Lemma 5.1** proves this matches reseeding on X \ X_R in distribution. The key identity is P(pick c_i from X | c_i ∉ X_R) = d²(c_i, C′)/φ(X′; C′).
- **Removal time (Theorem 5.4):** expected O(RK²d) for R random removals, against O(nKd) for retraining. It does not depend on n.
- **The weak spot:** the server reruns K-means++ and Lloyd **from scratch** whenever the aggregate changes, at cost O(K²LT d).

**From Making AI Forget You (Ginart et al., NeurIPS 2019): quantization for stability, and trees for modularity.**
- **Q-k-means:**
  - **Training:** after every Lloyd iteration, quantize each centroid to a random-phase ε-lattice, with a γ-balance correction so no cluster drops below γn/k points.
  - **Deletion:** recompute the analog centroids without the point (linearity), re-quantize with the *stored* phases, and retrain only if a quantized centroid changes.
  - **Lemma C.3:** with probability > 1 − 2mTkd^{3/2}/(εγn), the output is unchanged for any m deletions that avoid the initial seeds.
  - **Theorem 4.1:** expected total time O(m² d^{5/2}/ε) for m deletions, independent of n.
  - **Corollary 4.1.1:** E[L_Q] ≤ (8 ln k + 16)L* + ε√(nd(8 ln k + 16)L*) + nd ε²/4.
- **DC-k-means:** points go to random leaves of a tree, and a deletion re-solves one leaf-to-root path. **Proposition 4.2:** O(m · max{n^ρ, n^{1−ρ}} · d) for m deletions with w = n^ρ leaves.

  A depth-1 DC-k-means tree has exactly the shape of one-shot federated clustering, with clients as leaves and the server as the root.

**From SISA (Bourtoule et al., IEEE S&P 2021): checkpointing as a cost model.**
- A shot in MFC plays the role of a slice in SISA: save state before each one, and restart from the last unaffected checkpoint.
- SISA's expected-cost formulas (Eqs. 2–5) are the model for "expected recomputed work". Its warning applies too: under uniform requests, slicing alone gives at most about 1.5× in expectation.

**From FedShard (Wen et al., AAAI-26): one path, not everything.**
- Retraining only one leaf-to-root path of a merge tree gives a speedup of at least P/2 for a single client (Props 2–3). This is a template for analysing "recompute from the first affected stage".

### B.5 Proposed algorithm sketch: checkpointed, quantized MFC

*Proposed, not from any paper.*

**Training (record everything a deletion might need):**
1. **Client init:** replace the Awasthi–Sheffet init with **k-means++ seeding plus MUFC-style lazy reseeding**, which already has an exactness proof (Lemma 5.1), or keep Awasthi–Sheffet and prove a perturbation bound for it. The SVD projection makes that bound harder.
2. **Quantize every message:** use a random-phase ε-lattice (Ginart Q(·)) for the initial upload, each θ_max^(z,t), and each server centroid after *every* server Lloyd iteration. Store the phases.
3. **Checkpoint per shot t:** the pool S_t, the quantized global centres μ_t, and each client's quantized θ_max^(z,t). Each client keeps the analog local statistics (counts, sums) so the effect of a deletion can be computed by linearity.
4. **Guard the arg-max:** θ_max is a discrete choice, so store each client's top-2 cluster costs and the margin between them. A deletion that does not close the margin cannot flip the choice.

**Deleting a record x from client z:**
1. Remove x from its local cluster statistics and recompute client z's analog local centre for every shot (by linearity).
2. Re-quantize with the stored phases and check the stored θ_max margins.
3. If every quantized upload of client z is unchanged at every shot, the server's input never changed, so the whole trajectory is identical. Return the old model. The cost is local: O(kTd).
4. Otherwise find the first shot t* whose upload changed. Restore S_{t*−1}, μ_{t*−1} and replay shots t*..T. Only these shots cost communication with all clients.

**Removing a whole client z:**
1. Drop its uploads from S_t for every t.
2. Check whether the server's quantized μ_t would change. Client z is only 1/Z of each update, so with ε large enough the answer is usually "no" for the early shots.
3. Replay from the first shot that changed.

### B.6 Theory to aim for

*Conjecture, stated so it can be attacked.* Let p_t be the probability that a uniformly random single deletion changes any quantized message in shot t, given that no earlier shot changed. Replay starts at the first changed shot, so

  E[shots recomputed] = Σ_t P(first change at t) · (T − t + 1) ≤ T · Σ_t p_t

and

  E[deletion cost] ≤ O(kTd) (local check) + (Σ_t p_t) · T · C_shot,  with C_shot = O(n·kd) (one full round).

The work is bounding p_t, the same way Ginart et al. do (Lemmas C.2–C.3):
- **(a) Movement.** Deleting one point from a cluster of size ≥ γn/k moves its analog centre by at most k√d/(γn) (data in [0,1]^d). Push that through the server's Lloyd iterations on S_t. Each pool element has weight 1/|S_t|; the hard part is that Lloyd assignments can flip, which is why every server iteration must be quantized.
- **(b) Rounding.** Under a random phase, a move of size Δ changes the rounded point with probability ≤ 2dΔ/ε (Lemma C.2).
- **(c) The arg-max.** Bound the flip probability with a margin assumption, like MUFSC's margin density in Part C or the well-separability MFC already assumes.
- **(d) Quality.** Add the quantization penalty, as in Corollary 4.1.1: smaller ε means better clustering but more recomputation.

A clean final statement would look like Theorem 4.1: expected deletion time independent of n for fixed m, k, T, d and ε, with a clustering-cost guarantee that degrades gracefully with ε.

### B.7 Evaluation plan

- **Datasets:**
  - MFC's own: synthetic 10-Gaussian, UCI Adult, UCI Bank Marketing.
  - MUFC's: Covtype, Postures, and others.
  - Use 50 clients, k = 10, homogeneous and heterogeneous splits, as in MFC.
- **Baselines:**
  - retraining MFC from scratch;
  - MUFC (one-shot, exact);
  - depth-1 DC-k-means (clients as leaves);
  - k-FED.
- **Workloads:** uniformly random record deletions, whole-client withdrawals, and an adversarial stream that targets seeds and high-margin clusters (Ginart App. A.1 shows why this matters).
- **Metrics:**
  - amortized deletion time (algorithm time and end-to-end time);
  - number of shots recomputed;
  - communication rounds triggered;
  - k-means cost relative to centralized k-means;
  - MFC's per-device cost std and IQR (does unlearning hurt some clients more?).

### B.8 Risks to plan for

- MFC's guarantees cover only the first round, and its cost can rise across rounds on heterogeneous data. You may need a stopping rule or a bounded pool S anyway.
- The Awasthi–Sheffet init (SVD plus pruning) is hard to make stable. Swapping in MUFC's lazily reseeded k-means++ is the safer route.
- Quantization costs: Q-k-means scales as d^{5/2} and needs ε ≪ 1/√d to keep quality, so keep d small (Adult and Bank use 5–6 features in MFC).
- The pool S grows by Z centres per shot, which increases the state to checkpoint and edit.

### B.9 What to read first

- **MFC:** Algorithms 1–2 (client init, multishot loop) and Theorem 1.
- **MUFC:** Algorithm 3, Lemma 5.1, Theorem 5.4, and the server-side update.
- **Ginart:** Q-k-means (Sec. 4.1), Lemmas C.1–C.3, Theorem 4.1, Corollary 4.1.1; DC-k-means and Proposition 4.2.
- **SISA:** the expected retraining cost of sharding and slicing (Eqs. 2–5).

---

## Part C — What the new paper (folder 4, MUFSC) adds to direction 1

### C.1 The paper in brief

**Machine Unlearning for Federated Socially Fair k-Means Clustering**, by Reddy, S K, Raza and Prakash (IIT Madras / SVNIT); arXiv 2610.13717, 9 Oct 2026. Code: github.com/arareddy/mufsc. Saurav Prakash is also an author of MUFC, and the paper cites MUFC, Ginart, SISA and FedShard. It does **not** cite MFC or p-FClus.

- **Objective:** federated k-means where the goal is socially fair, i.e. minimise the *worse* of the two demographic groups' average squared distance. Every group-g record carries weight 1/n_g, so a deletion changes a *global* weight.
- **Phase I (one shot):** each (client, group) slice builds its own k-means++ summary without using global group sizes; only the server divides by n_g. A deletion therefore leaves untouched clients' summaries valid.
- **Phase II (T refinement rounds):** clients send exact integer statistics (count, sum, squared-norm sum) per group and cluster, and the server applies a guarded Fair-Lloyd update.
- **Unlearning:** rerun the fully specified training procedure with the **same seed**. Reuse untouched summaries, recompute touched ones, and patch the round statistics with exact signed corrections. The released model equals retraining value for value (Theorem 5).

### C.2 Direction 1, piece by piece: what MUFSC now provides

| Piece of direction 1 | Before MUFSC | What MUFSC adds | Consequence for the project |
|---|---|---|---|
| **Exact unlearning across several rounds** | No paper in the set handled multi-round federated clustering exactly (MUFC is one-shot; MFC has no deletion). | **Same-seed exact replay of T refinement rounds** (Algorithm 1, Theorem 5): cached per-round centres and exact integer statistics, patched round by round. | The "save the server state every round and replay" half of direction 1 **now exists**, for the fair objective. It is a baseline, not a contribution. |
| **What a checkpoint must hold** | Unspecified. | A concrete list: per-slice anchors and multiplicities, global group sizes and server sampler state, per-round centres and statistics, and per-point cached label, d1, d2 and runner-up ID. That is about 64nT bytes in memory; ACS checkpoints average 103 MB (max 504 MB). | Use it as the storage budget to beat. Quantized messages (B.5) should need far less. |
| **What "exact" means in practice** | Ginart's distributional definition. | A stricter **same-seed (pointwise)** target, and the engineering it needs: fixed-point encoding (b = 12 bits), a separate random stream per (seed, role, client, group), an exact rational sampler, and fixed tie and empty-cluster rules (the "declared training map"). | Decide early which definition you prove. Same-seed is easier to test (bit-for-bit) but brings new instability (next row). |
| **Stability: "most deletions change nothing"** | Q-k-means's quantization argument (Ginart), untested in multi-round federated settings. | **Assignment certificates** that keep a point's cached label if centres moved too little to change it (Lemma 4; runner-up test, Prop 8). They almost never help: useful coverage < 1.4%, and they give up at round 0 in 174/175 and 92/95 trials. **Lemma 10:** same-seed sampling is unstable (a tiny change in sampling weights changes the sample with probability > 1/2). **Proposition 11:** one legal deletion can force Ω(n) wasted work. | Strong evidence that **certifying unquantized centres is not enough**. Quantizing the round centres (B.5) is the missing ingredient; MUFSC quantizes only the Phase-I anchors (grid γ), never the refinement-round centres. |
| **Measured speedups** | None for multi-round federated clustering. | **Whole client:** 35.8× at T = 0 (compact path, essentially MUFC's server-only retraining; ordinary replay 17.8×), **2.36× at T = 1, 1.70× at T = 2**. **Single records:** about **1.06×** at T = 15. All are in-memory request times on one machine, for one preselected client per dataset. | These are the numbers direction 1 must beat. Gains collapse as rounds increase because **every replayed round rescans all surviving points**, the exact cost direction 1 tries to avoid. |
| **The theorem direction 1 wanted** | "Bound the expected number of recomputed rounds" was open. | **Eq. 29:** a replay work bound. **Proposition 9:** if replayed centres stay within L_s·r/n of the cached ones (a contraction assumption) and few points sit near decision boundaries, certificate failures scale with the deletion size r, not n. The authors decline to claim an unconditional speed theorem. | The theorem is **still open**. Prop 9 shows which assumptions make it provable (contraction, margin density); quantization could replace the contraction assumption with Ginart's rounding argument. |
| **Whole-client withdrawal** | MUFC: drop the client's counts and rerun the server. | **Proposition 13:** the same idea with group reweighting, plus a raw-data-free compact state (0.28–3.35 MB versus 12.5–55 MB full checkpoints) that survives sequential withdrawals. | For client removal in a one-shot design this is solved. The open question is client removal **with refinement rounds**. |
| **Code and data** | Re-implement everything. | Public code; datasets UCI Adult, UCI Bank Marketing, UCI Credit Default and 2018 ACS PUMS. Adult and Bank are also MFC's datasets. | You can run MUFSC's direct replay as a baseline on the same data MFC uses. |

### C.3 How direction 1 changes

**No longer novel:** "checkpoint every round and replay". MUFSC does it, proves it exact, and shows the gains are modest: 2.4× and 1.7× with one and two rounds for whole clients, 1.06× for single records.

**Still open, and now sharper:**
1. **Make later rounds stable, not just replayable.** Quantize the per-round centres with a random phase (Ginart) and/or use a **deletion-stable seeded sampler**; MUFSC suggests per-record random keys or priority sampling. Most deletions should then leave every round unchanged, so replay stops at the local check instead of rescanning all data. This targets MUFSC's Lemma 10 and Proposition 11.
2. **Prove the bound MUFSC declined to prove.** Show expected recomputed rounds ≤ T·Σ_t p_t with p_t from a Ginart-style rounding argument (B.6), under assumptions no stronger than MUFSC's Proposition 9.
3. **Keep the MFC-specific parts.** MFC's protocol differs from MUFSC's: only the highest-cost centre θ_max is uploaded per round, the server pool S grows, and MFC targets client-level cost fairness rather than demographic fairness. The arg-max upload needs the margin guard in B.5. MUFSC's exact integer statistics give a way to apply deletions as exact patches to MFC's local centres.

**Revised plan:**
1. Run MUFSC's code and reproduce its direct-replay numbers on Adult and Bank: 2.36× and 1.70× for whole clients, about 1.06× for records.
2. Implement MFC with per-round checkpoints and exact integer statistics. This is "MFC with MUFSC-style replay": the new baseline.
3. Add random-phase quantization of round centres plus the θ_max margin guard, and measure how often a deletion changes *nothing*. MUFSC's coverage figure (< 1.4%) is the number to beat by a wide margin.
4. Prove the expected recomputed-rounds bound for the quantized version, and report the clustering-quality cost of ε as in Ginart Corollary 4.1.1.
5. Optionally carry it over to the fair objective. That merges this direction with the current page's direction 1, "Fast per-record deletion for federated fair k-means".

### C.4 Relation to the current Directions page

After MUFSC was added, the Directions page was reordered:
- The new **direction 1** is *Fast per-record deletion for federated fair k-means*: beat MUFSC's 1.06× with a deletion-stable sampler.
- This file's direction became **direction 2**, *Exact unlearning for multi-round federated clustering*, with MUFSC's replay noted as a template.
- The original pre-MUFSC card (Part A, direction 1, word for word) is also shown on the current page as **direction 3**.

Read together, they are two views of one problem: **exact unlearning that stays cheap across refinement rounds in federated k-means.** One starts from MUFSC's fair objective, the other from MFC's protocol. The shared core is making seeding and per-round centres stable under deletion, and proving an expected-cost bound for it.

### C.5 Caveats about MUFSC's evidence

- The 35.8× comes from a separate compact implementation with three timing repeats. The T = 1 and T = 2 figures use the canonical direct replay with four repeats.
- Whole-client results use one preselected client per dataset. Bank's 115× reflects having only 20 clients.
- Timings come from one machine (Apple M3 Max), excluding network, secure aggregation and persistence. Secure aggregation is not implemented.
- Exactness is relative to one fixed preprocessing state. It does not remove a record's influence on the fitted scaler or encoding.
- The groups are SEX only for Adult and ACS (Bank: married vs not; Credit: education level).
- The authors corrected a reporting error in which the worst-group cost Φ and the sum G were swapped. They state they used generative AI heavily, including for proofs (some checked in Lean 4).
