# Research: Design, Publication & References

Everything related to system design, research novelty, publication strategy, and references.

---

## Subfolders

### [design/](design/) - System Design & Implementation

Complete architecture and implementation roadmap:
- SYSTEM_DESIGN.md - Architecture, perception, tools
- DATASET_AND_TRAINING.md - Training data strategy
- SAFETY_AND_MONITORING.md - Safety systems
- IMPLEMENTATION_ROADMAP.md - 6-8 week phased plan

→ **Start here if implementing**: [design/README.md](design/README.md)

### [reference/](reference/) - Literature & Citations

- references.bib - 150+ papers (BibTeX format)
- references_summary.md - Human-readable summaries

### [training/](training/) - Fine-Tuning Documentation

- FINDINGS.md - Qwen technical discoveries
- VALIDATION_RESULTS.md - Performance metrics

---

## Main Documents

### [RESEARCH_AND_PUBLICATION.md](RESEARCH_AND_PUBLICATION.md) ⭐ START HERE

Complete research roadmap including:

1. **Literature Review Summary**
   - SOTA landscape across 6 research areas
   - Competitive analysis vs. DRAGON, FunctionGemma, PRISM, LLM-Glasses
   - Your unique advantages and vulnerabilities

2. **Novelty Assessment**
   - Contribution is integration of existing components (fine-tuned LLM + scene graph + tool calling + edge)
   - Unique combination not yet published
   - Components becoming mainstream — timely execution matters

3. **Publication Strategy (3 Phases)**
   - **Phase 1** (Apr-Jun 2026): User study → ASSETS 2026, Benchmarks → RA-L
   - **Phase 2** (Jul-Sep 2026): Scene graph analysis → CoRL 2026 or RSS 2027
   - **Phase 3** (Oct-Dec 2026): System paper + open-source → Journal

4. **Venue Recommendations**
   - Tier-1: ASSETS 2026, IEEE RA-L, CoRL 2026
   - Tier-2: RSS 2027, ICRA 2027
   - Tier-3: Journals (Robotics & Autonomous Systems)

5. **6-Month Timeline**
   - Apr: Recruit VI participants, design IRB protocol
   - May: Conduct user study
   - Jun: Submit papers to ASSETS + RA-L
   - Aug-Oct: Review and publication
   - Nov: Open-source release

6. **Success Metrics**
   - User study: Task completion rate, SUS, NASA-TLX (thresholds TBD after baseline measurement)
   - Benchmarks: Measure actual latency and cost on target hardware
   - Reproducibility: Code, documentation, Docker

---

## Reference Materials

### [reference/references.bib](reference/references.bib)

Complete bibliography in BibTeX format:
- **150+ papers** organized by topic
- Papers in these areas:
  - LLMs for robot navigation (SayCan, Code as Policies, Inner Monologue)
  - Fine-tuned small LLMs (Gorilla, ToolLLM, Octopus, FunctionGemma)
  - Assistive technology for VI (DRAGON, LLM-Glasses, ChitChatGuide)
  - Scene graphs (ConceptGraphs, OrionNav, SayPlan)
  - Edge deployment (QLoRA, Unsloth, EdgeLoRA)
  - LLM + ROS2 integration

### [reference/references_summary.md](reference/references_summary.md)

Human-readable summary of key papers with:
- Paper title and authors
- Venue and year
- Key contribution
- Relevance to Guide-LLM v2

---

## How to Use

**If you're publishing**:
1. Read entire RESEARCH_AND_PUBLICATION.md
2. Choose your target venue (ASSETS, RA-L, or CoRL)
3. Follow the timeline for that phase

**If you're doing a user study**:
1. Section: "PHASE 1: Apr-Jun 2026"
2. Design study (Section 6)
3. IRB approval process
4. Follow ASSETS paper template

**If you need citations**:
1. Open `reference/references.bib` in your citation manager
2. Search by topic or author
3. Export to your paper format (BibTeX, Harvard, etc.)

**If you're evaluating SOTA**:
1. Read "Part 2: Competitive Analysis"
2. Section: "Direct Competitors vs. Guide-LLM v2"
3. Risk assessment and mitigation

---

## Key Findings (TL;DR)

✅ **Novel combination**: First to combine fine-tuned small LLM + scene graph + tool calling for robot navigation (individual components are established)

✅ **Publishable**: Suitable for ASSETS, RA-L, CoRL — pending real benchmarks and user study

⚠️ **Urgent**: User study needed by Jun 2026 (differentiates from DRAGON)

⚠️ **Window closing**: Components becoming mainstream (FunctionGemma Dec 2025, PRISM Jan 2026)

💡 **Strategy**: Publish user study FIRST to establish priority, then benchmarks, then system paper

---

## Next Steps

1. **This week**: Review RESEARCH_AND_PUBLICATION.md
2. **By Apr 1**: Recruit VI participants for user study
3. **By Apr 15**: IRB approval
4. **By May 31**: Conduct user study
5. **By Jun 15**: Submit to ASSETS 2026

---

**Version**: 2026-03-23
**Status**: Strategy complete, ready for execution
