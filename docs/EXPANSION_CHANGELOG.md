# Mathematics and code expansion

Version 0.6.0 adds a finite multiscale feature dictionary, exact raw-input neural sensitivities, a model-only global Lipschitz bound, fixed-family TV-shift audit bounds, and a cost-based zero-success trial planner. The corresponding derivations and proofs are integrated into `paper/bis_ifc_full.tex` and `paper/publication_versions/jpc_supplement.tex` through `paper/multiscale_extension.tex`.

The original main journal paper is retained in scope; the expanded derivations belong in its supplement. Earlier empirical tables are unchanged. Newly compiled PDFs supersede the corresponding compiled copies. The `main.pdf` and publication-version full-paper aliases are updated to the expanded manuscript; archived reviewer responses remain historical.

Validation executed locally: 161 privacy tests passed with 93.07 percent total coverage; the multiscale module attained 93 percent coverage. The synthetic development experiment was run. The companion sensor project is a separate package to keep the two supplied research subjects distinct. No external repository, PR or deployment was changed.

Existing lint defects in a late import block and a comprehension variable were corrected without changing algorithms. No claim of a real-world deep-learning improvement, a validated shift radius or a completed journal-readiness review is made by this expansion.

The older real-data check expected exploratory pilot values that round-four had removed from the full manuscript. Those supplied values now have an explicit archived provenance report, and the checker verifies that report separately from the unchanged main manuscript run-value checker. They are not reintroduced as current paper findings.
