# Blind reannotation instructions

1. Open each image in `images/` and annotate every visible defect instance.
2. Write one line per box in the matching file under `labels_to_fill/`.
3. Use YOLO format: `class_id center_x center_y width height`, all normalized to 0-1.
4. Do not inspect or open the original dataset labels.
5. Include touching or partially visible instances when their extent is identifiable.
6. Use a tight box around the defect evidence, not a region of uniformly textured background.
7. If an image contains no visible target, leave its label file empty and note it in the review log.

Class ids:

| id | class |
|---:|---|
| 0 | crazing |
| 1 | inclusion |
| 2 | patches |
| 3 | pitted_surface |
| 4 | rolled-in_scale |
| 5 | scratches |

This package is intentionally identity-blind and does not contain original boxes. The
`class_hint` column in `manifest.csv` may be hidden from annotators if a fully blind
class assignment is required.
