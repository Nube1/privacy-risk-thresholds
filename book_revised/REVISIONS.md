# Revisions applied in this copy

`book/` is the untouched original. This copy applies the edits of the revision supplement that fit this source.

Applied: Ch 4 detectability qualifier; Ch 6 log-threshold A_i worked values; Ch 7 legal-element section; Ch 8 value-curve fit note; Ch 9 composition-gap actions; Ch 11 softened claim; Ch 13 synthetic-register box and reading-precision procedure; Ch 16 and Appendix A threshold-elements wording and gaps paragraph.

Not applicable to this source (already present or chapters absent): Ch 15 calibration-band power check, Ch 14 and Ch 17 edits, Part VII notes (Ch 19-23), which exist only in draft 5.

## Rebuild from the 304-page draft (in progress)

Chapter 18 (Continuous adversarial learning) is rebuilt as `chapters/ch18-learning.tex` from the text of the 304-page draft, in a new Part VII. The old `ch18-open.tex` is now in Part VIII and still carries its old chapter number until Chapters 19 to 22 are rebuilt.

Chapter 19 (Releases that change over time) is rebuilt as `chapters/ch19-change.tex`. The draft text had a stray fragment in Section 19.2 ("depends on how the figure is computed."); it is repaired in the source. The draft gives 0.380 for the five-year mass in Example 19.6, a matrix exponential gives 0.379, and the source uses 0.379.

Still to rebuild: Chapters 20 to 22, the updated Chapter 23, and Appendix A.5 and A.6. The module names in Chapter 18's software note (`gate`, `evidence`, `registry`, `betting`) belong to the extension code bundle and are not in `src/privacy_risk`.
