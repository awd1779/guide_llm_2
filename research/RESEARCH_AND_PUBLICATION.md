# Research Strategy and Publication Roadmap

**Status**: SOTA analysis complete, publication plan defined
**Current date**: 2026-03-23
**Target venues**: ASSETS 2026, IEEE RA-L, CoRL 2026

---

## Executive Summary

Guide-LLM v2 combines fine-tuned small LLM + scene graph + tool calling for robot navigation — a combination not yet published. However, individual components are well-established. **Action required: Execute user study and edge benchmarks to substantiate claims before publishing.**

### Key Findings

✓ **Unique integration**: First system combining fine-tuned 4B LLM + scene graph + tool calling + edge deployment + VI focus
✓ **Clear gap**: DRAGON (RA-L 2024) most similar but cloud-based; you're edge-deployable
✓ **Publication path**: Suitable venues include ASSETS, RA-L, CoRL
✗ **Critical gap**: No real user study yet (makes-or-breaks publication vs. DRAGON)

---

## Part 1: Literature Review Summary

### State-of-the-Art Landscape

**LLMs for Robot Navigation** (Cloud-Based):
- SayCan (CoRL 2022): LLM + value functions, no edge deployment
- Inner Monologue (CoRL 2023): Closed-loop planning, cloud GPT-4
- Code as Policies (ICRA 2023): LLM generates robot code, safety concerns

**Fine-Tuned Small LLMs for Tool Calling** (Software APIs):
- Gorilla (ICML 2025): API mastery via SFT
- ToolLLM (ICLR 2024): 16K API mastery on Llama
- Amazon SLM (2024): OPT-350M outperforms ChatGPT on ToolBench (77.55% vs 26%)
  - **Key validation**: Small fine-tuned models beat large general models on specific tasks

**Assistive Technology for VI Navigation** (All Cloud-Based):
- **DRAGON** (RA-L 2024): Dialogue robot for PVI, VLM grounding, **most related work**
- LLM-Glasses (arXiv 2025): Wearable with cloud GPT-4o
- NaviGPT (GROUP 2025): Phone-based with Apple Maps + GPT-4
- **Key observation**: NO existing system uses fine-tuned small LLM for edge VI navigation

**Scene Graphs + LLM Navigation**:
- ConceptGraphs (ICRA 2024): Open-vocab 3D scene graphs
- OrionNav (arXiv 2024): LLM + scene graph on quadruped (no fine-tuning, not assistive)

### Your Unique Position

**What's been done individually**:
- Fine-tuned LLMs for tool calling ✓
- LLM-driven navigation ✓
- Scene graphs for navigation ✓
- Edge deployment on robots ✓
- VI assistive systems ✓

**What's been done in combination**: NOBODY

**Your specific contribution**:
> "The first system to combine a fine-tuned small LLM (4B/9B) with structured function calling over a scene graph for on-device assistive robot navigation, eliminating cloud dependency while achieving tool-calling accuracy competitive with larger models on domain-specific tasks."

---

## Part 2: Competitive Analysis

### Direct Competitors vs. Guide-LLM v2

| Dimension | Your System | DRAGON | FunctionGemma | PRISM | LLM-Glasses |
|-----------|-------------|--------|---------------|-------|------------|
| **Fine-tuned LLM** | ✓ | ✗ VLM | ✓ (Dec 25) | ✓ | ✗ Cloud |
| **Scene graph grounding** | ✓ | ✓ Visual | ✗ | ✗ | ✗ |
| **Tool calling** | ✓ 12 tools | ✓ 4 tools | ✓ 500+ APIs | ✗ | ✗ |
| **Edge deployment** | Planned (not yet benchmarked) | ✗ Cloud API | ✓ (Dec 25) | ✓ | ✗ Cloud |
| **VI/Accessibility** | ✓ | ✓ | ✗ | ✗ | ✓ |
| **Real user study** | ✗ NEEDED | ✓ | ✗ | ✗ | ✗ |
| **Published venue** | - | RA-L 2024 | Blog + ArXiv | ArXiv | ArXiv |

### Your Advantages

1. **Designed for edge deployment** (vs. DRAGON's cloud dependency)
   - Latency and cost advantages expected but **not yet measured on target hardware**
   - Privacy for VI users (no data leaves device)
   - Works offline (no cloud dependency)

2. **Structured knowledge** (scene graphs vs. visual VLMs)
   - Interpretable (no black-box perception)
   - Accessibility-friendly (screen reader compatible)
   - Verifiable and debuggable
   - Deterministic results

3. **Accessibility-first positioning**
   - VI navigation is specialized niche (less competition)
   - Real blind users testing
   - ASSETS 2026 venue alignment

### Your Vulnerabilities

1. **No user study yet** ← MUST FIX BY JUN 2026
   - DRAGON has this
   - Essential for accessibility credibility
   - Differentiator vs. cloud systems

2. **Components becoming mainstream**
   - FunctionGemma released Dec 2025 (Google)
   - PRISM released Jan 2026 (edge distillation)
   - Small models moving to edge rapidly
   - Window closing as components commoditize

3. **Individual components not novel**
   - Fine-tuned LLMs: Yes, normal now
   - Tool calling: Yes, many systems
   - Scene graphs: Yes, becoming standard
   - Innovation is in **integration**, not components

---

## Part 3: Publication Strategy (6-Month Roadmap)

### PHASE 1: Apr-Jun 2026 (EXECUTE NOW)

#### Paper 1A: Real User Study

**Target**: ASSETS 2026 (ACM Conference on Computers and Accessibility)
- **Submission deadline**: ~April 2026
- **Publication**: Oct 2026
- **Why**: Perfect venue for VI robotics; real users valued
- **Fit**: Strong venue match for VI robotics with real user validation

**Study Design**:
- Participants: 5-10 vision-impaired users
- Duration: 2-4 week deployment
- Tasks: Navigation in real environment
- Metrics:
  - Task completion rate (primary)
  - Time to completion
  - Preference vs. alternatives
  - NASA-TLX cognitive load
  - System Usability Scale (SUS)
  - Qualitative feedback

**Paper Focus**:
- Real-world validation of fine-tuned edge LLM
- User experience & accessibility insights
- Lessons learned from actual VI users
- Safety and reliability in practice

**Critical success factors**:
- Recruit diverse VI population
- Real navigation tasks (not simulators)
- Compare against: Manual navigation, verbal directions, DRAGON if possible
- Deep qualitative analysis

#### Paper 1B: Edge Performance Benchmark

**Target**: IEEE RA-L or Robotics & Autonomous Systems (Elsevier)
- **Submission deadline**: May-Jun 2026
- **Publication**: Aug-Oct 2026
- **Why**: Differentiates edge deployment; journal track
- **Fit**: Differentiates edge deployment; journal track (rolling submissions)

**Benchmark Content** (all TBD — must be measured, not projected):
- **Latency**: Cloud vs. edge vs. edge + fallback (measure on actual Jetson hardware)
- **Power consumption**: Jetson Orin Nano power profile (measure with hardware power monitor)
- **Cost analysis**: Calculate from actual API pricing and hardware amortization

- **Comparison matrix**:
  - Your system vs. DRAGON
  - Your system vs. GPT-4 baseline
  - Quantization impact (4-bit vs. 8-bit)
  - Model size trade-offs (4B vs. 9B)

- **Reproducibility**:
  - Docker containers
  - Dataset and code
  - Step-by-step setup guide

**Critical success factors**:
- Honest comparison (don't hide slowdowns)
- Real environment testing (not toy datasets)
- Ablation studies showing design choices

---

### PHASE 2: Jul-Sep 2026

#### Paper 2: Scene Graph Representation Analysis

**Target**: CoRL 2026 (July deadline) or RSS 2027 (Nov deadline)
- **Deadline**: CoRL ~Jul 2026, RSS ~Nov 2026
- **Publication**: Oct 2026 (CoRL) or Jul 2027 (RSS)
- **Why**: Scene graphs are hot topic; accessibility angle unique
- **Acceptance probability**: 40-50% CoRL, 50-60% RSS

**Paper Focus**:
- Text-based vs. visual scene graphs for LLM reasoning
- Accessibility benefits: Screen reader, interpretability
- Ablation: LLM performance with different representations
- User study: Do blind users prefer text-based navigation?

**Why this matters**:
- Scene graphs are trending (HOVSG, 3DGraphLLM accepted ICCV 2025)
- Your text-based approach is contrarian (interesting)
- Accessibility angle not explored in LLM+scene graph papers

---

### PHASE 3: Oct-Dec 2026

#### Paper 3: System + Open-Source

**Target**: Robotics & Autonomous Systems or Journal of Robotics Research
- **Deadline**: Oct-Nov 2026
- **Publication**: Early 2027
- **Why**: Consolidates findings; establishes long-term impact

**Paper + Release**:
- Complete system architecture
- Design trade-offs and rationale
- Implementation lessons learned
- Open-source codebase

**Open-Source Release**:
- GitHub: Full implementation + documentation
- Hugging Face: Model cards + quantized models
- Docker: Pre-built containers
- ROS2 integration: Ready to deploy

---

## Part 4: Venue Analysis

### Tier-1: Best Fit (Highest Priority)

#### ASSETS 2026
- **Perfect match**: Accessibility focus, VI robotics, real users
- **Community**: Disability researchers, accessibility engineers
- **Review process**: Understands VI challenges
- **Impact**: High credibility in accessibility community
- **Timeline**: Submit ~Jun, Publish Oct 2026

**vs. alternatives**: ICRA 2027 is slower (Oct deadline, May publication)

### Tier-1: Journal Track

#### IEEE RA-L
- **Matches**: DRAGON's venue (direct comparison)
- **Timeline**: Rolling submissions, Aug-Oct publication
- **Process**: Rolling submissions (flexible)
- **Bonus**: Can present at ICRA 2027

#### Robotics & Autonomous Systems (Elsevier)
- **Timeline**: Jun submission, Aug-Oct publication
- **Prestige**: Medium (not top-tier)
- **Advantage**: Faster publication

### Tier-2: Conference Track

#### CoRL 2026
- **Deadline**: ~Jul 2026
- **Publication**: Oct 2026
- **Best for**: Scene graph paper or system methodology

#### RSS 2027
- **Deadline**: Nov 2026
- **Publication**: Jul 2027
- **Prestige**: Top-tier (most competitive)
- **Timeline**: Longer but highest impact

#### ICRA 2027
- **Deadline**: Oct 2026
- **Publication**: May 2027
- **Size**: Largest robotics conference
- **Best for**: System paper or reproducibility angle

---

## Part 5: Timeline & Milestones

```
Timeline         What                              Venue             Status
---------        ----                              -----             ------
NOW (Mar 26)     Planning complete                -                 ✓ Done
APR 2026         Recruit VI participants          -                 START
                 Design study protocol            -                 START
                 IRB approval (expect 2-4 weeks)  -                 START

MAY 2026         Conduct user study (5-10)        -                 IN PROGRESS
                 Collect benchmarking data        -                 IN PROGRESS
                 Prepare ablation studies         -                 IN PROGRESS

JUN 2026         Submit user study paper          ASSETS 2026       DUE
                 Submit benchmark paper           RA-L/RAS         DUE
                 Analyze results, start writing   CoRL/RSS          PREPARE

AUG-OCT 2026     Papers under review              All venues        SUBMITTED
                 Prepare scene graph paper        CoRL/RSS          WRITING

OCT 2026         Submit system + open-source      RAS/JRR           TARGET
                 Open-source release              GitHub            LAUNCH

JAN 2027         Follow-up studies                -                 PLAN
                 Community feedback               -                 COLLECT

MAY-JUL 2027     Extended evaluation              ICRA/RSS          POSSIBLE
```

---

## Part 6: Success Metrics

### For User Study Paper (ASSETS 2026)

✓ Task completion: >80% (high bar but required)
✓ User satisfaction: SUS score >70
✓ Cognitive load: NASA-TLX <50%
✓ Preference over baseline: >70% prefer your system
✓ Qualitative insights: Novel accessibility findings

### For Benchmark Paper (RA-L)

✓ Latency: Measure actual end-to-end on target hardware
✓ Cost: Calculate from measured data (not projections)
✓ Reproducibility: Others can replicate setup
✓ Ablation: Show design choice justification
✓ Comparison: Fair evaluation vs. DRAGON and cloud baselines

### For Scene Graph Paper (CoRL/RSS)

✓ Accessibility metrics: Text-based <user study scores>
✓ LLM performance: >90% accuracy on navigation tasks
✓ Ablation: Show text vs. visual trade-offs
✓ Novelty: First accessibility-focused scene graph paper

### For System Paper (Journal)

✓ Reproducibility: Code runs from GitHub
✓ Documentation: Complete setup guide
✓ Community: >50 GitHub stars in first month
✓ Adoption: Citations from other projects

---

## Part 7: Risk Mitigation

### High Risks

**Risk**: DRAGON publishes v2 with edge deployment
- **Mitigation**: Publish user study FIRST (Jun 2026 vs. their later release)
- **Mitigation**: Position on accessibility angle (they're not there)

**Risk**: Google extends FunctionGemma to robotics
- **Mitigation**: Your system already specialized for navigation
- **Mitigation**: Open-source + community as moat

**Risk**: Startups enter VI robotics market
- **Commercial risk, not research**: Still novel academically
- **Mitigation**: Open-source builds goodwill, community

### Medium Risks

**Risk**: Scene graph approach becomes mainstream
- **Mitigation**: Publish Paper 2 by Sep 2026 (early)
- **Mitigation**: Position on accessibility benefits (unique angle)

**Risk**: User study shows poor results (low completion rate)
- **Mitigation**: Publish results honestly (negative results still valuable)
- **Mitigation**: Analyze why and improve

**Risk**: Peer reviewers want more users (want N>10)
- **Mitigation**: Frame 5-10 as rigorous qualitative study
- **Mitigation**: Plan for follow-up with larger N

---

## Part 8: Writing Strategy

### Paper Templates (Ready to Use)

**ASSETS Paper Structure**:
1. Introduction (1 page): VI navigation challenge + your solution
2. Related Work (1 page): DRAGON, LLM-Glasses, RDog comparison
3. System Design (1.5 pages): Architecture, tools, dataset
4. User Study (2 pages): Participants, tasks, evaluation
5. Results (2 pages): Completion rates, satisfaction, qualitative
6. Discussion (1 page): Insights, limitations, future work
7. References (0.5 page)

Total: ~9 pages (ASSETS page limit)

**RA-L Paper Structure**:
1. Introduction (1 page): Edge deployment motivation
2. Related Work (1.5 pages): Cloud vs. edge, fine-tuning, robotics
3. System Design (2 pages): Architecture, perception hierarchy, tools
4. Benchmarking Methodology (1.5 pages): Metrics, baselines, setup
5. Results (2.5 pages): Latency, power, cost, comparison matrices
6. Ablation Studies (1.5 pages): Model size, quantization, tool count
7. Discussion (1 page): Implications, limitations
8. Conclusion (0.5 page)

Total: ~11 pages (RA-L page limit)

---

## Part 9: Next Steps (ACTION ITEMS)

### This Week (Before Mar 30)

- [ ] Contact local VI organizations for participant recruitment
- [ ] Draft IRB protocol for user study
- [ ] Identify external baseline for comparison (DRAGON simulation?)
- [ ] Set up Jetson Orin benchmarking environment

### By Apr 30

- [ ] Recruit 5-10 VI participants
- [ ] Receive IRB approval
- [ ] Complete all benchmarking measurements
- [ ] Draft outline for ASSETS and RA-L papers

### By May 31

- [ ] Conduct 50-100% of user study
- [ ] Analyze preliminary results
- [ ] Write first draft of ASSETS paper
- [ ] Write first draft of RA-L paper

### By Jun 15

- [ ] SUBMIT to ASSETS 2026
- [ ] SUBMIT to RA-L/RAS
- [ ] Begin scene graph paper writing

---

## References

See `/home/ubuntu/guide-llm_2/references/references.bib` for complete bibliography (150+ sources).

Key papers cited:
- Hasan et al., "DRAGON: A Dialogue-Based Robot for Assistive Navigation," RA-L 2024
- Song et al., "Guide-LLM: An Embodied LLM Agent," ArXiv 2024
- Ahn et al., "SayCan: Grounding Language in Robotic Affordances," CoRL 2022
- Amazon, "Small Language Models for Efficient Tool Calling," ArXiv 2024

---

**Created**: 2026-03-23
**Status**: Ready for execution
**Contact**: [Your email]
