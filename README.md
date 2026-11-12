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
connectivity. The zoom toolbar uses stepped magnifications that include 1×; click
**1:1** to return directly to source-pixel scale.

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

The **Experiment Instructions** panel accepts the documented `key=value` formats
for `template_auto_align`, `logical_grouping`, `threshold_growth`, and
`progressive_anchored_fit`. Use **Load
Example** to fill in a sample without running it; **Run Experiment** validates the
complete configuration and then runs the current image analysis on a worker thread.
Template names are `digit_5x7`,
`thin_zero`, and `thin_zero_independent_xy`. The parser rejects missing, repeated,
unknown, malformed, or out-of-range values and requires an in-bounds, non-empty
half-open ROI. The box is treated strictly as data; it cannot load files or run
code. Invalid configurations leave the prior experiment results and analysis
settings unchanged. If validation or a running experiment fails, the panel shows
the exception and traceback in a selectable error-details box; **Copy Error
Details** copies the complete text for sharing. Successful results update the ROI,
threshold, template view, and report with initial/final IoU, offsets, scales,
overlap counts, and image dimensions. No files are written.

## Stage 1E: threshold growth history

Select an ROI, set the desired starting value with the **Brightness Threshold**
slider, click **Find Connected Sections**, and select one section in the section
list. Set a maximum threshold and increment, choose the ROI occupancy limit, then
click **Track Selected Section**. The first pixel of the selected component in
row-major order becomes its stable seed coordinate; later component merges do not
change that identity. Use **Previous** and **Next** to step through the recorded
threshold samples in **Growth History** view.

Each frame reports the seed-connected component's pixel count, newly selected
pixels from the interval since the preceding sample, cumulative other starting
sections absorbed, inclusive-pixel bounding rectangle as `(left, top, width,
height)`, centroid, total selected-pixel percentage in the ROI, and recorded
events. If the increment does not land exactly on the maximum, the exact maximum
is included as the final sample; additions in that final interval are grouped
together. Events note section connections/absorptions, growth, and the first
sample at which ROI occupancy reaches the configured background-dominated limit.
The history uses the original grayscale values and does not change the photograph.
This is a connectivity visualization, not character recognition or a stroke
proposal; a high occupancy warning is a user-configured safeguard, not an
automatically selected threshold.

### Paste-and-run Threshold Growth History

The **Experiment Instructions** box also accepts a `threshold_growth`
configuration. ROI coordinates are half-open original-image pixels and are
independent of canvas zoom or scroll position. Running a valid configuration sets
the ROI and starting threshold, finds and lists the connected sections with their
stable seed coordinates, and centers the analysis view on that ROI. The ROI is
automatically fitted to the Pixel Analysis viewport by default; set
`auto_zoom=false` to keep the current zoom.

```text
experiment=threshold_growth
roi_left=895
roi_top=112
roi_right=926
roi_bottom=151
brightness_threshold=80
maximum_threshold=160
threshold_increment=1
background_limit_percent=50
```

To run growth immediately, add both `seed_x` and `seed_y`, using an actual selected
pixel within the ROI, for example `seed_x=900` and `seed_y=120`. The application
rejects a seed that is outside the ROI or not selected at the starting threshold.
If the seed is omitted, no component is guessed: select the intended entry in
**Connected Sections** (its seed is shown in the row), then click **Track Selected
Section**. The application switches to **Growth History** automatically once
analysis runs. Invalid pasted instructions are rejected before the current analysis
state is changed. Template-matching and logical-grouping configurations retain
their separate existing formats.

To save a recording, add an explicit seed and `auto_record=true`:

```text
seed_x=899
seed_y=137
auto_record=true
snapshot_mode=on_change
experiment_name=zero_01
```

Without both `auto_record=true` and explicit seed coordinates, the run only finds
sections for interactive selection and does not create files in Pictures.

`auto_record=true` processes the complete configured threshold series, saves a
CSV row for every threshold, creates display-independent PNG snapshots when the
tracked overlay or relevant growth state changes, records snapshot decisions in a
manifest, and packages the PNGs, CSVs, and settings as `results.zip`. A threshold
change alone does not create another PNG; an unchanged final threshold is recorded
in the manifest as reusing the latest matching snapshot. Manifest reasons and
columns distinguish tracked-component pixel counts from selected ROI pixels outside
that component, so a snapshot caused only by the rest of the ROI is identified.
The GUI shows progress and offers **Cancel**.
Cancellation or a write/packaging error removes the incomplete run directory; it
does not leave a completed-looking ZIP. A successful run is written under
`<Pictures>/PixelMathExperiments/<experiment_name>/<unique-run-id>/`. Windows'
Pictures known folder is used when available; otherwise the fallback is
`Path.home() / "Pictures"`. Each run receives a new, unique directory. Starting a
new run with the same experiment name removes previous recorder-generated run
directories for that name before creating the new run;
unrelated files in its parent folder are left untouched. If the Pictures or
`PixelMathExperiments` folder was deleted, the recorder recreates it when writing
results. The source photograph is not copied into the output.

PNG snapshots use the original-color ROI as their base, overlay tracked, newly
admitted, and boundary pixels with the existing Stage 1E colors, and include a
measurement label and legend. Their pixel scale is selected from the ROI size and
is independent of the app zoom, operating-system display scaling, and window size.
No missing strokes are generated. With `auto_zoom=true`, the on-screen ROI is
enlarged to the largest continuous magnification that fits inside Pixel Analysis
(up to 64×), rather than rounding down to a whole-number zoom.

## Stage 1F: Logical Shape Grouping Lab

Select an ROI and threshold, then find connected sections. In **Logical Grouping
Lab**, select one or more sections (Ctrl/Shift-click to extend the selection),
enter a piece ID, and choose **Combine Sections into Piece**. A piece can contain
multiple connected sections without changing their source component data. Select
one or more pieces and choose **Create Group** to add a grouping hypothesis;
remove a group to change it. Mark a manual group or a listed alternative as the
intended annotation. In **Logical Grouping** view, dashed boundaries show each
group, with the intended answer highlighted in green. The boundary is a visual
hypothesis, not a feature detected in the photograph. Marking an alternative
records it separately from the **Groups** table, which lists only groups created
manually; the Logical Grouping Lab displays the recorded grouping in a persistent
status label, and the main status line confirms the action.

The measurements report nonnegative horizontal/vertical edge gaps, vertical
overlap, bottom-edge baseline difference, bounding-box dimensions and ratios, and
distances normalized by the median piece height. Alternatives are deterministic
and capped at eight: the all-separate grouping is retained, followed by
spacing/alignment-based one-pair and two-disjoint-pair hypotheses. The geometry
orders candidates but cannot prove they are correct; one-piece groups are allowed.
When two intended groups each contain two pieces, the report compares their
normalized gap, baseline-difference, and height-ratio measurements without
assigning character identities.

The **Experiment Instructions** box accepts a separate `logical_grouping`
configuration. Each piece lists one or more actual selected component seed
coordinates (`x,y`, separated by `|`); the intended groups partition the piece
IDs. Replace the sample ROI and seeds with coordinates from the loaded image:
every seed must be inside the half-open ROI and selected at the configured
threshold, and two seeds in a piece must identify distinct sections.

```text
experiment=logical_grouping
roi_left=0
roi_top=0
roi_right=640
roi_bottom=480
brightness_threshold=120
pieces=piece_1=100,200;piece_2=120,200;piece_3=220,200;piece_4=280,200;piece_5=300,200
intended_groups=ten=piece_1,piece_2;plus=piece_3;twenty=piece_4,piece_5
```

This example annotates the intended `10 + 20` partition. To investigate `10`,
define two pieces in one intended group; for a widely spaced `1 0`, place each
piece in a separate intended group. These labels are annotations only: neither
the parser nor the measurements use them to identify digits or operators.
**Load Example** builds a five-piece `10 + 20` configuration from the five
leftmost currently detected sections, using real seed pixels; review all generated
seeds and group labels before running. Invalid text leaves the current experiment
state unchanged, and the text box never executes code.

## Stage 1G: Progressive Anchored Template Fitting

Paste an `experiment=progressive_anchored_fit` configuration into **Experiment
Instructions** and run it against the loaded photograph. Stage 1G reuses the
manually drawn 5×7 templates for `0`, `6`, `8`, and `9` from
`template_matching.py`; these are illustrative binary shapes, not learned or
photograph-specific templates. The required seed must be selected at the starting
threshold. The supplied zero reference cases are separate runs with their own
explicit ROI and seed:

```text
experiment=progressive_anchored_fit
roi_left=895
roi_top=112
roi_right=926
roi_bottom=151
seed_x=899
seed_y=137
brightness_threshold=80
maximum_threshold=160
threshold_increment=1
background_limit_percent=100
candidate_digits=0,6,8,9
initial_offset_x=0
initial_offset_y=0
initial_rotation_degrees=0
initial_width_scale=1.0
initial_height_scale=1.0
step_translation_pixels=1
cumulative_translation_pixels=3
step_rotation_degrees=5
cumulative_rotation_degrees=15
step_scale=0.05
cumulative_scale=0.15
auto_record=true
experiment_name=zero_reference_1
```

For the second reference, use this complete configuration instead:

```text
experiment=progressive_anchored_fit
roi_left=859
roi_top=111
roi_right=891
roi_bottom=149
seed_x=863
seed_y=140
brightness_threshold=80
maximum_threshold=160
threshold_increment=1
background_limit_percent=100
candidate_digits=0,6,8,9
initial_offset_x=0
initial_offset_y=0
initial_rotation_degrees=0
initial_width_scale=1.0
initial_height_scale=1.0
step_translation_pixels=1
cumulative_translation_pixels=3
step_rotation_degrees=5
cumulative_rotation_degrees=15
step_scale=0.05
cumulative_scale=0.15
auto_record=true
experiment_name=zero_reference_2
```

Both fitting methods for each digit start with the exact same configured offset,
rotation, and independent width/height scales. The fixed method never changes that
transform. The progressive method uses a deterministic, ordered coordinate search
at each later threshold, testing the current value and one positive/negative step
for X, Y, rotation, width, and height, in that order. Its score is binary IoU minus
penalties of 0.002 per source-pixel offset, 0.0005 per degree, and 0.05 per unit
scale deviation from the initial transform. Ties prefer higher IoU, the smaller
single-step change, then canonical Y, X, rotation, width, and height values. Each
step and cumulative bound is enforced independently for both axes/scales relative
to the initial transform; there is no unrestricted jump or stretch. Scale limits
are deviations from each configured initial scale.

At each threshold the tracked component is fitted separately to each template.
Matched pixels are component pixels covered by the template; unmatched photographed
pixels are tracked-component pixels outside it; unsupported template pixels are
foreground template pixels outside the tracked component. Every attachment set is
stored in original-image coordinates. Retained is the intersection of the previous
threshold's current attachments with the current set; gained and lost are the
current-minus-previous and previous-minus-current sets. Thus a pixel that returns
after being lost is gained again, not continuously retained. Newly threshold-selected
pixels are computed across the ROI; newly connected pixels are current tracked
component pixels that were neither in the prior component nor newly selected.
These photographic growth sets are independent of template attachment changes.

With `auto_record=true`, the run is written to
`<Pictures>/PixelMathExperiments/<experiment_name>/<unique-run-id>/`. It saves a
PNG overlay for each digit/method/threshold, `fitting_history.csv` with transforms
and coordinate sets (including cumulative transform deltas from the initial pose),
`growth_measurements.csv`, `fixed_vs_progressive.csv`,
`experiment_settings.txt`, and `results.zip` containing those files. Overlay
colors are green for matched pixels, orange-red for unmatched observed pixels, blue
for unsupported template pixels, and cyan for the ROI boundary. Coordinates and
fitting always use source-image pixels, not GUI zoom; the original photograph and
observed mask are not modified, and no proposed pixels are added.
Without automatic recording, the GUI shows the first configured candidate's final
progressive overlay and final attachment counts for all candidates. This is a fit
comparison only: no winning digit or character recognition is reported. Each
reference configuration must be run independently; the program does not use the
known identity of either photographed zero.

## Stage 1G-B: Attachment-Aware Progressive Fitting

Stage 1G-B is a separately selected `experiment=progressive_anchored_fit_b` mode.
It uses the same threshold growth, 5×7 templates, centered initial placement,
nearest-neighbor rasterization, ordered coordinate search, and transform limits as
Stage 1G. Each candidate/weight pair has an independent transform and attachment
history. Include weight `0` to reproduce the Stage 1G progressive baseline; the
other values are not automatically tuned.

At each later threshold a candidate transform is scored using
`IoU - transform_penalty - attachment_loss_weight * lost_previous_matches / max(1, previous_match_count)`.
Both IoU and the attachment-loss fraction are dimensionless values in `[0, 1]`;
the weight is a configurable non-negative finite multiplier on that same scale.
Losses compare original-image coordinates from the immediately preceding threshold
with coordinates matched by the proposed transform. The first threshold has no
previous attachments and therefore incurs no attachment penalty. Ties use the
existing Stage 1G ordering. This discourages losing matches but does not guarantee
that a fit with no losses is a good fit.

The first proposed controlled comparison can use this complete configuration:

```text
experiment=progressive_anchored_fit_b
roi_left=543
roi_top=34
roi_right=588
roi_bottom=90
seed_x=572
seed_y=44
brightness_threshold=80
maximum_threshold=130
threshold_increment=1
background_limit_percent=100
candidate_digits=0,6,8,9
initial_offset_x=0
initial_offset_y=0
initial_rotation_degrees=0
initial_width_scale=1.0
initial_height_scale=1.0
step_translation_pixels=1
cumulative_translation_pixels=3
step_rotation_degrees=5
cumulative_rotation_degrees=15
step_scale=0.05
cumulative_scale=0.15
attachment_loss_weights=0,0.02,0.05,0.10
auto_record=true
experiment_name=stage_1g_b_attachment_aware
```

With `auto_record=true`, results are written under
`<Pictures>/PixelMathExperiments/stage_1g_b_attachment_aware/<unique-run-id>/`,
separate from Stage 1G recordings. Each run includes `fitting_history.csv` with
per-threshold transforms, score parts, original-coordinate attachment sets,
retention fractions, cumulative gained/lost coordinates and counts, threshold
growth sets, and bound hits; `growth_measurements.csv`; `weight_comparisons.csv`;
an overlay per candidate/weight/threshold; `experiment_settings.txt`; and
`results.zip`. `newly_selected_pixels` and `newly_connected_pixels` are photographic
threshold-growth measurements, distinct from gained template attachments. A
previously lost coordinate that matches again is gained again, not retained.

## Tests

```powershell
python -m unittest discover -v
```

Tests use small generated images and do not require a photograph, network access,
or OCR/AI dependencies.

## Stage 1 limitations

- Connected sections and merge paths are pixel-connectivity results, not character
  recognition or evidence that a bridge is genuine handwriting.
- Logical groups are developer annotations and bounded geometric alternatives;
  results do not classify digits/operators, determine a correct grouping, or
  generalize from the initial examples.
- Stage 1G compares manually defined templates through bounded, deterministic
  transforms and records attachment history; a higher overlap is not recognition,
  and the starter templates may not resemble the handwriting or discriminate
  between digits.
- Stage 1G-B adds an attachment-loss preference to that fitting objective, but a
  lower loss count alone is not evidence of a better digit fit; compare it alongside
  IoU, transform changes, and the complete threshold history.
- Threshold analysis uses fixed values 40, 80, 120, 160, 200, and 240 and does not
  recommend a threshold.
- Very large images still require memory for their RGB, grayscale, and mask data;
  analysis runs in a worker thread, but the application is intended for interactive
  experimentation rather than unbounded image sizes.
