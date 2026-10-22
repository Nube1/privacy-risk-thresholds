# Submitting to ACM TOPS

`python scripts/make_acmart.py` writes `paper/tops/main.tex` from `paper/main.tex`,
with the tables and figures beside it. Compile from that directory:

```bash
cd paper/tops
pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

Do not edit `paper/tops/main.tex`. It is overwritten on every run of the script. The
manuscript lives in `paper/main.tex`.

## If the journal has a page limit

```bash
python scripts/make_acmart.py --supplement
```

This writes `paper/tops/main.tex` with the appendices removed and
`paper/tops/supplement.tex` holding the proofs, the expanded derivations and the
simulation details, each compiling on its own. Both keep the same class options and
metadata, and the main file keeps the bibliography. Compile each with three
`pdflatex` passes from that directory. Check that no cross-reference in the main
text points into the supplement before you send it, since the two files number
independently.

## Before you submit

1. **Blinding.** The class options are `acmsmall,review,screen`. Add `anonymous` if
   the journal asks for a blinded copy. That also hides the author block and the
   acknowledgement.
2. **ORCID.** The generated file carries a placeholder. Put the real one in
   `scripts/make_acmart.py`.
3. **CCS concepts.** Two are set: social aspects of security and privacy, and
   privacy-preserving protocols. Regenerate them from the ACM CCS tool if you want a
   different emphasis, and paste the XML block into the script.
4. **Copyright.** `\setcopyright{none}` suits a review copy. ACM supplies the right
   command after acceptance.
5. **Reproducibility.** TOPS asks whether artefacts are available. They are: the
   repository builds every number, table and figure in the paper, and
   `docs/CLAIMS_EVIDENCE.md` maps each claim to the evidence for it and to what
   would refute it.

## What the reviewers will ask

The paper is positioned on one claim: per-release thresholds do not govern a
release programme, and the evaluation quantifies the gap on a monthly credit release
system. Sections 7, 10.6 and 13 of `paper/main.tex` carry that argument. The theory sections are the
apparatus it rests on, not the contribution.

The register is synthetic and structurally calibrated. No new mechanism is
proposed. Both are stated in the paper, in Section 13.7 and the discussion.
