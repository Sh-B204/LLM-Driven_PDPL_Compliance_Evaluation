# Evaluation data

The dataset covers 21 applications. Processed policy files for
19 applications are currently included; the remaining two will be
added after preprocessing.

| Input | Location |
|---|---|
| Processed privacy policies | `data/policies/` |
| Implementing Regulations corpus | `data/pdpl/IRPDPL.docx` |
| Expert labels | `data/ground_truth/PDPL_Results.xlsx`, sheet `Expert Analysis` |

Application names, identifiers and policy paths are defined in
`configs/applications.yaml`. The experiment evaluates the applications
listed in that file.

Policy files must use Word heading styles to identify sections.
The loader extracts paragraph text; tables, images and text boxes
are not extracted.

Expert labels are matched by application name and rubric criterion
code. Each included application must have a binary label for every
criterion.
