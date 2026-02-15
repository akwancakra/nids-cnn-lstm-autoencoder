---
name: academic-research-agent
description: Perform systematic academic literature research, extract structured data, and synthesize findings for a specific research need.
version: 1.1
author: Akwan Cakra Tajimalela
created: 2026-02-02
---

## When to use
Use this skill when the user asks for academic research, literature review, systematic search, paper comparison, or synthesis of findings.

## Core capabilities
1) Systematic literature search
   - Build keywords and boolean strings
   - Search scholarly sources (Scholar, IEEE Xplore, ScienceDirect, PubMed, ACM)
   - Define inclusion/exclusion criteria
   - Backward and forward citation chasing
   - Log search strategy for reproducibility

2) Source evaluation
   - Check relevance to research question
   - Assess methodological quality
   - Verify venue and author credibility
   - Identify bias and limitations
   - Prioritize by impact and recency

3) Structured data extraction (per paper)
   - Summary
   - Methodology
   - Results
   - Key findings
   - Limitations
   - Important concepts
   - Contributions
   - Implications
   - Further readings

4) Synthesis and analysis
   - Find patterns and trends
   - Identify gaps and contradictions
   - Cluster by themes
   - Compare methods and outcomes
   - Propose hypotheses or future directions

5) Documentation and reporting
   - Produce structured literature review
   - Build comparison tables
   - Format citations (APA/IEEE/Harvard)
   - Create annotated bibliography
   - Provide simple visuals or PRISMA flow

## Workflow

### Stage 1: Clarify research needs
Input: topic, research questions, goals
Output: research protocol (keywords, scope, criteria)

### Stage 2: Systematic search
Input: research protocol
Output: candidate list with metadata

### Stage 3: Screening and selection
Input: candidates
Output: final corpus with selection notes

### Stage 4: Data extraction
Input: final corpus
Output: structured extraction table (CSV/Excel)

### Stage 5: Synthesis and writing
Input: extraction table
Output: literature review draft with citations

## Quality criteria
- Relevance: direct alignment with research questions
- Credibility: peer-reviewed preferred
- Recency: last 5 years unless seminal
- Traceability: clear link from source to claim

## Output templates

### Search report (markdown)
```
## Search Strategy
Database: Google Scholar
Query: ("cnn-lstm" OR "autoencoder") AND ("intrusion detection" OR "nids")
Date: 2026-02-02
Results: 847 hits
After filter: 234 hits
```

### Extraction table (columns)
Author | Year | Methodology | Sample | Key Findings | Limitations

### Synthesis structure (markdown)
```
## Theme 1: [Name]
[Synthesis with citations]

## Theme 2: [Name]
[Synthesis with citations]

## Research Gaps
[Specific gaps]
```

## Limitations
- Cannot access paywalled sources without credentials
- Methodology appraisal still needs human validation

## Usage example
```
research_topic: "Zero-day detection in NIDS using autoencoders"
research_questions:
  - "How effective are hybrid CNN-LSTM autoencoders for zero-day detection?"
  - "What datasets and metrics are used for cross-dataset validation?"
inclusion_criteria:
  - "Empirical studies with quantitative results"
  - "2019-2026"
exclusion_criteria:
  - "Non-technical reviews"
databases:
  - "Google Scholar"
  - "IEEE Xplore"
max_papers: 30
citation_style: "APA 7th"
```
