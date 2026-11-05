# Evaluation inputs

All 21 configured application DOCX files are included in `policies/`. Exact paths and IDs are in `configs/applications.yaml`.

The retrieval corpus is `pdpl/IRPDPL.docx`. Expert labels are in `ground_truth/PDPL_Results.xlsx`, evaluation uses the individual 15 binary criterion labels rather than the percentage column.

Heading styles define policy sections. The pipeline reads body paragraphs. Input-analysis tables record the extraction audit and language-script shares.
