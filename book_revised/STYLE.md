# How chapters of this book are written

## The formula (Muller and colleagues, Journal of Computer Assisted Learning, 2008)

A clear explainer barely moves readers who think they already know the answer. Stating their wrong
belief and then taking it apart does. Every chapter follows six steps:

1. Name the belief the reader brings to the topic (the misconception).
2. State it fairly, in its strongest form, then give early the first fact that breaks it.
3. Ask the one question the rest of the chapter answers. Ask it once, plainly. Few other questions.
4. Build the answer as a connected case, paragraph linked to paragraph.
5. Run two threads. The A-plot is a human story; the B-plot is the technical material. Every three
   to five dense paragraphs, return briefly (two to five sentences) to the A-plot, then go back.
6. Close the main text by answering the question plainly, then the In practice box, then proofs.

## The A-plot: who the human thread follows

Use only two kinds of people.

- **Documented public cases**, told accurately and cited from the bibliography of
  `paper/bis_ifc_full.tex` (for example Sweeney's linkage of hospital records to a voter list,
  `sweeney2000`; the Netflix Prize de-anonymisation, `narayanan2008`; the census reconstruction,
  `garfinkel2019`, `abowd2022`; mobility traces, `demontjoye2013`). Never invent quotes, dates or
  details about real people.
- **The composite cast at "the Bank"**, a fictional central bank introduced in the preface. They
  are composites, and the preface says so.
  - *Naledi*, the Bank's disclosure control officer, who runs the assessments.
  - *The chair* of the Bank's statistics release committee, who signs decisions.
  - *Tomás*, a visiting researcher who asks for several files from the credit register.
  - *Pieter*, a credit analyst at a reporting lender: the insider.
  - *Grace*, from internal audit, who reads decision records after the fact.

Keep the cast restrained: short scenes about decisions, documents and meetings. No melodrama, no
invented feelings beyond what a scene needs, no dialogue longer than a sentence or two.

## Prose rules

- Academic register; serif look; plain prose. Short words where precision survives; active voice.
- No em dashes (`---` or the character). En dashes only for ranges.
- None of the AI-writing tells: no "not only X but Y", no "X rather than Y" hedging disclaimers, no
  puffery (crucial, pivotal, vital, key role, landscape, testament, underscore, highlight,
  showcase, robust, comprehensive, delve, leverage, matters, notably, additionally to open a
  sentence, serves as, stands as), no rule-of-three flourishes, no summary paragraphs that restate.
- Every table and figure is referred to in the text and explained there.
- Figures carry every distinction twice, by shade and by hatch or line style.
- Prefer a figure to a list or table where a figure works.

## Evidence rules

- Every number comes from the repository: `paper/generated/*.tex`, `paper/generated/numbers.tex`,
  `artifacts-*/*.json`, `docs/*.md`, or the text of `paper/bis_ifc_full.tex`. Put the source in a
  LaTeX comment beside the number, for example `% source: paper/generated/table_credit.tex`.
- Report negative results where the repository has them.
- Synthetic evidence is called synthetic, with its chosen parameters stated where they bite.

## LaTeX conventions

- One file per chapter in `chapters/`, named `chNN-short.tex`, starting with
  `\chapter{Title}\label{ch:short}`.
- Prefix every label with the chapter number: `c04:prop-point`, `c04:eq-cp`, `c04:fig-...`.
- Environments from `preamble.tex`: `proposition`, `corollary`, `definition`, `assumption`,
  `example`, `remark`, `inpractice`, `lstlisting`. Macros `\Prob`, `\E`, `\ind{}`.
- End each chapter with `\begin{inpractice} ... \end{inpractice}` (including an *In the software*
  paragraph naming modules and functions, with at most one short verbatim excerpt of 15 lines or
  fewer in `lstlisting`), then `\section{Proofs}` with full proofs of the chapter's propositions.
- Cite only keys present in the `thebibliography` of `paper/bis_ifc_full.tex`.
- Figures: `\includegraphics` from `../paper/figures`, `../paper/diagrams` or
  `../artifacts-paper/figures`, or new pgfplots figures in the chapter using patterns.
