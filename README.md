# Pixel Math Recognition Lab

Stage 1 is a desktop experiment for inspecting image pixels and observing how
brightness thresholds change connected dark sections. It does not recognize
characters or alter the source image.

## Requirements

- Python 3.10 or newer
- Pillow (installed with `python -m pip install -r requirements.txt`)
- Tkinter (included with most standard Python installations)

## Run

```powershell
python main.py
```

Open a PNG, JPG/JPEG, BMP, or WEBP image. The original image is retained as RGB;
grayscale and threshold masks are separate images. Zoom and scroll the analysis
canvas to inspect pixels, click a pixel to view its values, adjust the threshold,
and use **Find Connected Sections** or **Analyze Thresholds** to explore 8-neighbor
connectivity.

## Stage 1B: ROI merge experiment

Use **Select ROI** and drag over the analysis canvas to choose a half-open source
pixel rectangle; **Clear ROI** restores full-image analysis. At a fixed threshold,
find the connected sections and choose **Select 2 Merge Seeds**. Click one selected
pixel in each of two different sections, then choose **Track ROI Merge**.
Set **Maximum Merge Threshold** to cap the search range; it must not be below
the current brightness threshold.

The report finds the first threshold that connects the selected seed pixels, sizes
of their starting sections, size of the merged component, and how many other
starting sections it absorbed. The overlay marks every pixel at the exact merge
brightness in magenta and one deterministic shortest connecting path's newly
admitted pixels in yellow; pixels in both categories are cyan. The GUI legend
identifies these colors and the two seed markers. The report separates the three
disjoint categories and shows totals for threshold-added and path pixels. The path
is a reproducible example, not a unique or minimal set of necessary pixels. The
selected-pixel fraction at the merge level helps reveal near-saturated results;
pixel connectivity alone cannot tell whether a bridge is handwriting or
background. **Merge Mask** shows the mask at the merge threshold (or the maximum
threshold if no merge occurs); **Binary Mask** remains the starting-threshold view.

## Stage 1C: template overlay and match experiment

Select an ROI around one photographed character, wait for the threshold mask, choose
a digit in **Template Overlay and Match Lab**, and click **Start / Restart**. The
baseline uses the chosen 5×7 template uniformly scaled to fit and centered in the
ROI. Change the X/Y offsets in source-image pixels or adjust the uniform scale
multiplier (0.25–2.50); **Reset Fit** restores the centered baseline transform.
The **Template Match** view colors shared selected pixels green, image-only pixels
red, and template-only pixels blue. The score is binary-mask intersection over
union (IoU), and the panel records the digit, fixed brightness threshold, half-open
ROI coordinates, offsets, actual scale in image pixels per template pixel, and
pixel-category counts.

Digit templates are manually defined 5×7 binary arrays in
`template_matching.py`; `1` means foreground/selected (black in the mask), and `0`
means background. They are simple starter shapes, not font samples or learned
templates. Scaling uses nearest-neighbor sampling. Offsets are relative to the
centered ROI fit, and template pixels beyond the ROI are clipped. Changing the ROI,
threshold, or digit clears the comparison; restart to establish a new baseline.
If both masks are empty, IoU is reported as N/A because its denominator is zero;
if only one is empty, the score is 0%.

IoU is an overlap measurement, not recognition confidence. A wrong digit can score
high if the ROI is small, the threshold removes useful strokes, the template is
scaled to overlap a large portion of a simple shape, or the image contains
background pixels that happen to match. It does not infer missing strokes, rank
digits, or modify the source photograph.

## Stage 1D: controlled zero-template comparison

The template panel adds three explicit variants: the unchanged digit bank (the
existing 5×7 zero is the baseline), a manually drawn 7×9 thin zero outline with
uniform scaling, and that same outline with independent width/height scale
adjustments. Independent multipliers are each bounded to 0.80–1.20 of the centered
fit dimensions; translation and the brightness threshold remain independently
recorded. The comparison continues to use binary IoU and reports baseline and
adjusted scores, shared/image-only/template-only counts, variant, X/Y offsets,
width/height scales, threshold, and the half-open ROI as
`left, top, right, bottom` (right and bottom exclusive).

For a two-character holdout, tune the first ROI, click **Carry Fit to Next ROI**,
then select the second ROI. The app applies the saved template variant/digit,
threshold, relative X/Y pixel offset, uniform scale multiplier, and width/height
multipliers once. It recomputes the centered baseline for the new ROI, then applies
the saved offsets and scales without retuning. The panel states which values were
transferred. The source photograph remains unchanged.

The **Auto-Align Offset** experiment searches integer X/Y offsets in a square
neighborhood around the current offset (default radius ±5 original-image pixels;
configurable from 0 to 50). The ROI, brightness threshold, template variant, and all
scale values remain fixed. It evaluates binary IoU at every candidate and applies
the highest-scoring offset. Ties prefer the smallest squared displacement from the
starting offset, then the smaller Y offset, then the smaller X offset. The report
shows starting and selected offsets/scores and the improvement in percentage
points. Template pixels extending beyond the ROI are clipped, as in manual fitting.
This is a bounded search of offsets only; it does not classify the character or
modify the source image or template.

## Paste-and-run experiment box

The **Experiment Instructions** panel accepts only the documented `key=value`
fields for `template_auto_align`. Use **Load Example** to fill in a sample without
running it; **Run Experiment** validates the complete configuration and then runs
the current image analysis on a worker thread. Template names are `digit_5x7`,
`thin_zero`, and `thin_zero_independent_xy`. The parser rejects missing, repeated,
unknown, malformed, or out-of-range values and requires an in-bounds, non-empty
half-open ROI. The box is treated strictly as data; it cannot load files or run
code. Invalid configurations leave the prior experiment results and analysis
settings unchanged. Successful results update the ROI, threshold, template view,
and report with initial/final IoU, offsets, scales, overlap counts, and image
dimensions. No files are written.

## Tests

```powershell
python -m unittest discover -v
```

Tests use small generated images and do not require a photograph, network access,
or OCR/AI dependencies.

## Stage 1 limitations

- Connected sections and merge paths are pixel-connectivity results, not character
  recognition or evidence that a bridge is genuine handwriting.
- Threshold analysis uses fixed values 40, 80, 120, 160, 200, and 240 and does not
  recommend a threshold.
- Very large images still require memory for their RGB, grayscale, and mask data;
  analysis runs in a worker thread, but the application is intended for interactive
  experimentation rather than unbounded image sizes.
