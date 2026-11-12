"""Tkinter application for exploring image pixels and threshold connectivity."""

from __future__ import annotations

import bisect
import queue
import tkinter as tk
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from PIL import Image, ImageTk

from connected_sections import (
    ConnectedSection,
    MergeAnalysis,
    MergeOverlayCategories,
    MERGE_BOTH_COLOR,
    MERGE_PATH_ONLY_COLOR,
    MERGE_THRESHOLD_ONLY_COLOR,
    Roi,
    analyze_component_merge,
    colorize_sections,
    create_merge_overlay,
    find_connected_sections,
)
from experiment_instructions import (
    ExperimentConfig,
    TemplateAutoAlignExperimentResult,
    parse_experiment_config,
    run_template_auto_align_experiment,
)
from growth_experiment_recorder import (
    AutomaticGrowthRecordingResult,
    RecordingCancelled,
    record_threshold_growth_experiment,
)
from logical_grouping import (
    MAX_GROUPING_CANDIDATES,
    GroupingAnalysis,
    LogicalPiece,
    analyze_grouping,
    combine_sections,
    compare_grouping_consistency,
    format_grouping_description,
    mark_candidate_intended,
)
from logical_grouping_instructions import (
    LogicalGroupingConfig,
    LogicalGroupingExperimentResult,
    parse_logical_grouping_config,
    run_logical_grouping_experiment,
)
from threshold_growth_instructions import (
    ThresholdGrowthConfig,
    ThresholdGrowthExperimentResult,
    parse_threshold_growth_config,
    run_threshold_growth_experiment,
)
from pixel_processing import (
    create_binary_mask,
    create_preview,
    get_pixel_info,
    load_image,
)
from template_matching import (
    IMAGE_ONLY_COLOR,
    AUTO_ALIGN_MAX_RADIUS,
    INDEPENDENT_SCALE_MAX,
    INDEPENDENT_SCALE_MIN,
    OVERLAP_COLOR,
    TEMPLATE_VARIANT_DIGIT,
    TEMPLATE_VARIANT_THIN_ZERO,
    TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT,
    TEMPLATE_ONLY_COLOR,
    TemplateComparison,
    TemplateFitSettings,
    TemplateTransform,
    auto_align_offset,
    compare_template,
    create_template_overlay,
    format_roi_coordinates,
    load_template_variant,
    transfer_template_fit,
)
from threshold_growth import (
    GROWTH_BOUNDARY_COLOR,
    GROWTH_COMPONENT_COLOR,
    GROWTH_NEW_BOUNDARY_COLOR,
    GROWTH_NEW_PIXELS_COLOR,
    GrowthAnalysisCancelled,
    ThresholdGrowthHistory,
    analyze_threshold_growth,
    create_growth_overlay,
)

ZOOM_LEVELS = (
    0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0,
    6.0, 8.0, 10.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
)


def next_zoom_level(current: float, direction: int) -> float:
    if direction not in (-1, 1):
        raise ValueError("Zoom direction must be -1 or 1.")
    if direction > 0:
        index = bisect.bisect_right(ZOOM_LEVELS, current)
        return ZOOM_LEVELS[min(index, len(ZOOM_LEVELS) - 1)]
    index = bisect.bisect_left(ZOOM_LEVELS, current) - 1
    return ZOOM_LEVELS[max(index, 0)]


def fit_zoom_to_roi(
    roi: Roi,
    viewport_size: tuple[int, int],
    margin: int = 24,
    maximum_zoom: float = 64.0,
) -> float:
    left, top, right, bottom = roi
    viewport_width, viewport_height = viewport_size
    roi_width, roi_height = right - left, bottom - top
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI must have a positive width and height.")
    if viewport_width <= 0 or viewport_height <= 0:
        raise ValueError("Viewport dimensions must be positive.")
    if margin < 0 or maximum_zoom <= 0:
        raise ValueError("Margin must be nonnegative and maximum zoom positive.")

    available_width = max(1, viewport_width - margin)
    available_height = max(1, viewport_height - margin)
    return min(
        maximum_zoom,
        available_width / roi_width,
        available_height / roi_height,
    )


class PixelMathRecognitionLab:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pixel Math Recognition Lab")
        self.root.geometry("1250x760")
        self.root.minsize(850, 500)

        self.rgb_image: Image.Image | None = None
        self.grayscale_image: Image.Image | None = None
        self.original_preview: Image.Image | None = None
        self.binary_mask: Image.Image | None = None
        self.section_image: Image.Image | None = None
        self.sections: list[ConnectedSection] = []
        self.section_labels: array | None = None
        self.selected_section: ConnectedSection | None = None
        self.roi: Roi | None = None
        self.merge_seeds: list[tuple[int, int]] = []
        self.merge_result: MergeAnalysis | None = None
        self.merge_overlay: Image.Image | None = None
        self.merge_overlay_categories: MergeOverlayCategories | None = None
        self.merge_mask: Image.Image | None = None
        self.growth_history: ThresholdGrowthHistory | None = None
        self.growth_overlay: Image.Image | None = None
        self.growth_seed: tuple[int, int] | None = None
        self.growth_frame_index = 0
        self.growth_max_threshold = tk.IntVar(value=255)
        self.growth_threshold_increment = tk.IntVar(value=1)
        self.growth_occupancy_limit = tk.DoubleVar(value=70.0)
        self.growth_job_running = False
        self._growth_job_version: int | None = None
        self.logical_pieces: dict[str, LogicalPiece] = {}
        self.logical_groups: list[tuple[str, tuple[str, ...]]] = []
        self.logical_intended_group: str | None = None
        self.logical_analysis: GroupingAnalysis | None = None
        self.logical_candidate_index = 0
        self.logical_intended_candidate: int | None = None
        self.logical_groups_are_intended = False
        self.logical_intended_groups: tuple[tuple[str, ...], ...] | None = None
        self.logical_candidate_tree_ids: list[str] = []
        self.template_active = False
        self.template_image: Image.Image | None = None
        self.template_comparison: TemplateComparison | None = None
        self.template_baseline: TemplateComparison | None = None
        self.template_roi: Roi | None = None
        self.template_threshold: int | None = None
        self.template_base_scale = 1.0
        self.template_transform: TemplateTransform | None = None
        self.template_variant = tk.StringVar(value=TEMPLATE_VARIANT_DIGIT)
        self.template_digit = tk.StringVar(value="0")
        self.template_offset_x = tk.IntVar(value=0)
        self.template_offset_y = tk.IntVar(value=0)
        self.template_scale_factor = tk.DoubleVar(value=1.0)
        self.template_width_scale = tk.DoubleVar(value=1.0)
        self.template_height_scale = tk.DoubleVar(value=1.0)
        self.auto_align_radius = tk.IntVar(value=5)
        self.auto_align_report = tk.StringVar(value="No auto-align run yet.")
        self.experiment_job_running = False
        self._experiment_cancel_event: Event | None = None
        self.pending_template_fit: TemplateFitSettings | None = None
        self._selecting_roi = False
        self._roi_drag_start: tuple[int, int] | None = None
        self._roi_rectangle_id: int | None = None
        self._selecting_merge_seeds = False
        self.source_path: Path | None = None
        self.threshold = tk.IntVar(value=128)
        self.merge_limit = tk.IntVar(value=255)
        self.view_mode = tk.StringVar(value="Grayscale")
        self.zoom = 10.0
        self.rendered_image: ImageTk.PhotoImage | None = None
        self.original_photo: ImageTk.PhotoImage | None = None
        self._mask_version = 0
        self._mask_after_id: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pixel-analysis")
        self._worker_results: queue.Queue[tuple[str, int, object, BaseException | None]] = (
            queue.Queue()
        )
        self._experiment_progress: queue.Queue[tuple[int, int, int, str]] = queue.Queue()
        self._closed = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_worker_results)

    def _build_ui(self) -> None:
        panes = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        original_panel = ttk.Frame(panes, padding=6)
        analysis_panel = ttk.Frame(panes, padding=6)
        controls_panel = ttk.Frame(panes, padding=6)
        panes.add(original_panel, weight=1)
        panes.add(analysis_panel, weight=3)
        panes.add(controls_panel, weight=2)

        controls_view = ttk.Frame(controls_panel)
        controls_view.pack(fill=tk.BOTH, expand=True)
        self.controls_canvas = tk.Canvas(controls_view, highlightthickness=0)
        controls_scrollbar = ttk.Scrollbar(
            controls_view, orient=tk.VERTICAL, command=self.controls_canvas.yview
        )
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls_panel = ttk.Frame(self.controls_canvas, padding=2)
        controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls_panel, anchor=tk.NW
        )
        controls_panel.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            ),
        )
        self.controls_canvas.bind(
            "<Configure>",
            lambda event: self.controls_canvas.itemconfigure(
                controls_window, width=event.width
            ),
        )

        ttk.Label(original_panel, text="Original Image", font=("", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        self.dimensions_label = ttk.Label(original_panel, text="No image loaded")
        self.dimensions_label.pack(anchor=tk.W, pady=(0, 6))
        original_frame = ttk.Frame(original_panel)
        original_frame.pack(fill=tk.BOTH, expand=True)
        self.original_canvas = tk.Canvas(
            original_frame, background="#333333", highlightthickness=0
        )
        original_xbar = ttk.Scrollbar(
            original_frame, orient=tk.HORIZONTAL, command=self.original_canvas.xview
        )
        original_ybar = ttk.Scrollbar(
            original_frame, orient=tk.VERTICAL, command=self.original_canvas.yview
        )
        self.original_canvas.configure(
            xscrollcommand=original_xbar.set, yscrollcommand=original_ybar.set
        )
        self.original_canvas.grid(row=0, column=0, sticky="nsew")
        original_ybar.grid(row=0, column=1, sticky="ns")
        original_xbar.grid(row=1, column=0, sticky="ew")
        original_frame.rowconfigure(0, weight=1)
        original_frame.columnconfigure(0, weight=1)
        self.original_canvas.bind("<Configure>", self._render_original)

        ttk.Label(analysis_panel, text="Pixel Analysis", font=("", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )
        toolbar = ttk.Frame(analysis_panel)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(toolbar, text="View:").pack(side=tk.LEFT)
        view_selector = ttk.Combobox(
            toolbar,
            textvariable=self.view_mode,
            values=(
                "Grayscale",
                "Binary Mask",
                "Connected Sections",
                "Merge Mask",
                "Merge Overlay",
                "Template Match",
                "Growth History",
                "Logical Grouping",
            ),
            state="readonly",
            width=19,
        )
        view_selector.pack(side=tk.LEFT, padx=(4, 0))
        view_selector.bind("<<ComboboxSelected>>", self._render_analysis)
        ttk.Button(toolbar, text="−", width=3, command=lambda: self._change_zoom(-1)).pack(
            side=tk.RIGHT
        )
        self.zoom_label = ttk.Label(toolbar, text="10×")
        self.zoom_label.pack(side=tk.RIGHT, padx=5)
        ttk.Button(toolbar, text="+", width=3, command=lambda: self._change_zoom(1)).pack(
            side=tk.RIGHT
        )
        ttk.Button(
            toolbar,
            text="1:1",
            width=4,
            command=lambda: self._set_zoom(1.0),
        ).pack(side=tk.RIGHT, padx=(0, 5))

        analysis_frame = ttk.Frame(analysis_panel)
        analysis_frame.pack(fill=tk.BOTH, expand=True)
        self.analysis_canvas = tk.Canvas(
            analysis_frame, background="#dddddd", highlightthickness=0,
        )
        self.analysis_xbar = ttk.Scrollbar(
            analysis_frame, orient=tk.HORIZONTAL, command=self._scroll_analysis_x
        )
        self.analysis_ybar = ttk.Scrollbar(
            analysis_frame, orient=tk.VERTICAL, command=self._scroll_analysis_y
        )
        self.analysis_canvas.configure(
            xscrollcommand=self.analysis_xbar.set, yscrollcommand=self.analysis_ybar.set
        )
        self.analysis_canvas.grid(row=0, column=0, sticky="nsew")
        self.analysis_ybar.grid(row=0, column=1, sticky="ns")
        self.analysis_xbar.grid(row=1, column=0, sticky="ew")
        analysis_frame.rowconfigure(0, weight=1)
        analysis_frame.columnconfigure(0, weight=1)
        self.analysis_canvas.bind("<Configure>", self._render_analysis)
        self.analysis_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.analysis_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.analysis_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.analysis_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.analysis_canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.analysis_canvas.bind("<Button-4>", lambda event: self.analysis_canvas.yview_scroll(-3, "units"))
        self.analysis_canvas.bind("<Button-5>", lambda event: self.analysis_canvas.yview_scroll(3, "units"))
        self.analysis_canvas.bind("<Shift-Button-4>", lambda event: self.analysis_canvas.xview_scroll(-3, "units"))
        self.analysis_canvas.bind("<Shift-Button-5>", lambda event: self.analysis_canvas.xview_scroll(3, "units"))

        ttk.Button(controls_panel, text="Open Image", command=self._open_image).pack(
            fill=tk.X, pady=(0, 12)
        )
        roi_buttons = ttk.Frame(controls_panel)
        roi_buttons.pack(fill=tk.X, pady=(0, 6))
        self.roi_button = ttk.Button(
            roi_buttons, text="Select ROI", command=self._toggle_roi_selection,
            state=tk.DISABLED,
        )
        self.roi_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.clear_roi_button = ttk.Button(
            roi_buttons, text="Clear ROI", command=self._clear_roi, state=tk.DISABLED
        )
        self.clear_roi_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        self.roi_label = ttk.Label(controls_panel, text="ROI: full image", wraplength=300)
        self.roi_label.pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(controls_panel, text="Brightness Threshold").pack(anchor=tk.W)
        self.threshold_scale = tk.Scale(
            controls_panel, from_=0, to=255, orient=tk.HORIZONTAL,
            variable=self.threshold, command=self._on_threshold_change,
            showvalue=True, resolution=1,
        )
        self.threshold_scale.pack(fill=tk.X, pady=(0, 8))
        merge_limit_row = ttk.Frame(controls_panel)
        merge_limit_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(merge_limit_row, text="Maximum Merge Threshold").pack(side=tk.LEFT)
        self.merge_limit_spinbox = ttk.Spinbox(
            merge_limit_row, from_=0, to=255, textvariable=self.merge_limit,
            width=5, increment=1,
        )
        self.merge_limit_spinbox.pack(side=tk.RIGHT)
        self.find_button = ttk.Button(
            controls_panel, text="Find Connected Sections", command=self._find_sections,
            state=tk.DISABLED,
        )
        self.find_button.pack(fill=tk.X, pady=3)
        self.analyze_button = ttk.Button(
            controls_panel, text="Analyze Thresholds", command=self._analyze_thresholds,
            state=tk.DISABLED,
        )
        self.analyze_button.pack(fill=tk.X, pady=3)
        self.merge_seed_button = ttk.Button(
            controls_panel, text="Select 2 Merge Seeds",
            command=self._toggle_merge_seed_selection, state=tk.DISABLED,
        )
        self.merge_seed_button.pack(fill=tk.X, pady=(8, 3))
        self.merge_track_button = ttk.Button(
            controls_panel, text="Track ROI Merge",
            command=self._track_roi_merge, state=tk.DISABLED,
        )
        self.merge_track_button.pack(fill=tk.X, pady=3)
        self.merge_info_label = ttk.Label(
            controls_panel,
            text="Find connected sections, then select two different sections to track.",
            wraplength=300,
            justify=tk.LEFT,
        )
        self.merge_info_label.pack(anchor=tk.W, pady=(3, 6))
        ttk.Label(controls_panel, text="Merge Overlay Legend", font=("", 9, "bold")).pack(
            anchor=tk.W
        )
        self._build_merge_legend(controls_panel)
        self._build_growth_controls(controls_panel)
        self._build_template_controls(controls_panel)
        self._build_experiment_instructions(controls_panel)
        self.status_label = ttk.Label(controls_panel, text="Open an image to begin.", wraplength=300)
        self.status_label.pack(anchor=tk.W, pady=(6, 8))

        ttk.Separator(controls_panel).pack(fill=tk.X, pady=6)
        ttk.Label(controls_panel, text="Selected Pixel", font=("", 10, "bold")).pack(anchor=tk.W)
        self.pixel_info_label = ttk.Label(controls_panel, text="Click a pixel in the analysis view.", justify=tk.LEFT)
        self.pixel_info_label.pack(anchor=tk.W, pady=(4, 10))
        ttk.Label(controls_panel, text="Connected Sections", font=("", 10, "bold")).pack(anchor=tk.W)
        section_frame = ttk.Frame(controls_panel)
        section_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.section_tree = ttk.Treeview(
            section_frame,
            columns=("pixels", "bounds"),
            show="headings",
            height=7,
            selectmode="extended",
        )
        self.section_tree.heading("pixels", text="Pixels")
        self.section_tree.heading("bounds", text="Bounds (x, y, w, h)")
        self.section_tree.column("pixels", width=60, stretch=False)
        self.section_tree.column("bounds", width=145, stretch=True)
        section_scroll = ttk.Scrollbar(
            section_frame, orient=tk.VERTICAL, command=self.section_tree.yview
        )
        self.section_tree.configure(yscrollcommand=section_scroll.set)
        self.section_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        section_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.section_tree.bind("<<TreeviewSelect>>", self._on_section_selected)
        self._build_logical_grouping_controls(controls_panel)

        ttk.Label(controls_panel, text="Threshold Analysis", font=("", 10, "bold")).pack(
            anchor=tk.W
        )
        self.threshold_tree = ttk.Treeview(
            controls_panel,
            columns=("selected", "sections", "largest"),
            show="headings",
            height=6,
        )
        for column, heading, width in (
                ("selected", "Threshold / selected pixels", 150),
                ("sections", "Sections", 65),
                ("largest", "Largest", 65),
        ):
            self.threshold_tree.heading(column, text=heading)
            self.threshold_tree.column(column, width=width, stretch=True)
        self.threshold_tree.pack(fill=tk.X, pady=(4, 0))

    def _build_growth_controls(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Threshold Growth History", padding=5)
        panel.pack(fill=tk.X, pady=(3, 5))
        ttk.Label(
            panel,
            text="Starting threshold uses the Brightness Threshold slider. "
                 "Select a section below, then track its seed pixel.",
            wraplength=300,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        controls = ttk.Frame(panel)
        controls.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(controls, text="Maximum").grid(row=0, column=0, sticky=tk.W)
        self.growth_max_spinbox = ttk.Spinbox(
            controls, from_=0, to=255, textvariable=self.growth_max_threshold,
            width=5, state=tk.DISABLED,
        )
        self.growth_max_spinbox.grid(row=0, column=1, padx=(3, 8))
        ttk.Label(controls, text="Increment").grid(row=0, column=2, sticky=tk.W)
        self.growth_increment_spinbox = ttk.Spinbox(
            controls, from_=1, to=255, textvariable=self.growth_threshold_increment,
            width=5, state=tk.DISABLED,
        )
        self.growth_increment_spinbox.grid(row=0, column=3, padx=(3, 0))
        ttk.Label(controls, text="Background limit (%)").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(3, 0)
        )
        self.growth_occupancy_spinbox = ttk.Spinbox(
            controls, from_=1, to=100, increment=1,
            textvariable=self.growth_occupancy_limit, width=5, state=tk.DISABLED,
        )
        self.growth_occupancy_spinbox.grid(row=1, column=3, sticky=tk.E, pady=(3, 0))
        self.growth_track_button = ttk.Button(
            panel, text="Track Selected Section", command=self._track_section_growth,
            state=tk.DISABLED,
        )
        self.growth_track_button.pack(fill=tk.X, pady=(3, 2))
        navigation = ttk.Frame(panel)
        navigation.pack(fill=tk.X)
        self.growth_previous_button = ttk.Button(
            navigation, text="Previous", command=lambda: self._step_growth(-1),
            state=tk.DISABLED,
        )
        self.growth_previous_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.growth_next_button = ttk.Button(
            navigation, text="Next", command=lambda: self._step_growth(1),
            state=tk.DISABLED,
        )
        self.growth_next_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.growth_step_label = ttk.Label(panel, text="No growth history.")
        self.growth_step_label.pack(anchor=tk.W, pady=(3, 0))
        self.growth_info_label = ttk.Label(
            panel, text="Find connected sections inside an ROI to begin.",
            wraplength=300, justify=tk.LEFT,
        )
        self.growth_info_label.pack(anchor=tk.W, pady=(2, 0))
        legend = ttk.Frame(panel)
        legend.pack(fill=tk.X, pady=(3, 0))
        for color, label in (
            (GROWTH_COMPONENT_COLOR, "Tracked"),
            (GROWTH_NEW_PIXELS_COLOR, "New"),
            (GROWTH_BOUNDARY_COLOR, "Boundary"),
            (GROWTH_NEW_BOUNDARY_COLOR, "New boundary"),
        ):
            tk.Label(
                legend, text="■",
                fg=f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}",
                font=("", 10, "bold"),
            ).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Label(legend, text=label).pack(side=tk.LEFT, padx=(0, 5))

    def _build_logical_grouping_controls(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Logical Grouping Lab", padding=5)
        panel.pack(fill=tk.X, pady=(3, 5))
        ttk.Label(
            panel,
            text="Select sections above to combine into a piece. Then select pieces "
                 "to form a group. Dashed boxes are hypotheses, not detected boundaries.",
            wraplength=300,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        piece_row = ttk.Frame(panel)
        piece_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(piece_row, text="Piece ID").pack(side=tk.LEFT)
        self.logical_piece_name = tk.StringVar(value="piece_1")
        self.logical_piece_name_entry = ttk.Entry(
            piece_row, textvariable=self.logical_piece_name, width=12
        )
        self.logical_piece_name_entry.pack(side=tk.LEFT, padx=(4, 4))
        self.logical_add_piece_button = ttk.Button(
            piece_row, text="Combine sections", command=self._add_logical_piece,
            state=tk.DISABLED,
        )
        self.logical_add_piece_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.logical_piece_tree = ttk.Treeview(
            panel, columns=("sections", "bounds"), show="tree headings",
            height=4, selectmode="extended",
        )
        self.logical_piece_tree.heading("#0", text="Piece")
        self.logical_piece_tree.heading("sections", text="Section seeds")
        self.logical_piece_tree.heading("bounds", text="Bounds")
        self.logical_piece_tree.column("#0", width=75, stretch=False)
        self.logical_piece_tree.column("sections", width=100, stretch=True)
        self.logical_piece_tree.column("bounds", width=100, stretch=True)
        self.logical_piece_tree.pack(fill=tk.X, pady=(2, 2))
        group_controls = ttk.Frame(panel)
        group_controls.pack(fill=tk.X)
        self.logical_create_group_button = ttk.Button(
            group_controls, text="Create group from selected pieces",
            command=self._create_logical_group, state=tk.DISABLED,
        )
        self.logical_create_group_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.logical_remove_group_button = ttk.Button(
            group_controls, text="Remove group", command=self._remove_logical_group,
            state=tk.DISABLED,
        )
        self.logical_remove_group_button.pack(side=tk.LEFT, padx=(4, 0))
        self.logical_group_tree = ttk.Treeview(
            panel, columns=("pieces", "intended"), show="tree headings",
            height=3, selectmode="browse",
        )
        self.logical_group_tree.heading("#0", text="Group")
        self.logical_group_tree.heading("pieces", text="Pieces")
        self.logical_group_tree.heading("intended", text="Annotation")
        self.logical_group_tree.column("#0", width=70, stretch=False)
        self.logical_group_tree.column("pieces", width=150, stretch=True)
        self.logical_group_tree.column("intended", width=75, stretch=False)
        self.logical_group_tree.pack(fill=tk.X, pady=(2, 2))
        self.logical_group_tree.bind(
            "<<TreeviewSelect>>", self._on_logical_group_selected
        )
        self.logical_mark_intended_button = ttk.Button(
            panel, text="Mark selected group intended",
            command=self._mark_logical_group_intended, state=tk.DISABLED,
        )
        self.logical_mark_intended_button.pack(fill=tk.X, pady=(2, 2))
        self.logical_candidate_tree = ttk.Treeview(
            panel, columns=("pair", "score"), show="tree headings",
            height=4, selectmode="browse",
        )
        self.logical_candidate_tree.heading("#0", text="Alternative")
        self.logical_candidate_tree.heading("pair", text="Proposed pair")
        self.logical_candidate_tree.heading("score", text="Geometry score")
        self.logical_candidate_tree.column("#0", width=100, stretch=False)
        self.logical_candidate_tree.column("pair", width=110, stretch=True)
        self.logical_candidate_tree.column("score", width=90, stretch=False)
        self.logical_candidate_tree.pack(fill=tk.X, pady=(2, 2))
        self.logical_candidate_tree.bind(
            "<<TreeviewSelect>>", self._on_logical_candidate_selected
        )
        self.logical_mark_candidate_button = ttk.Button(
            panel, text="Mark selected alternative intended",
            command=self._mark_logical_candidate_intended, state=tk.DISABLED,
        )
        self.logical_mark_candidate_button.pack(fill=tk.X, pady=(2, 2))
        self.logical_intended_label = ttk.Label(
            panel, text="No intended grouping recorded.",
            wraplength=300, justify=tk.LEFT,
        )
        self.logical_intended_label.pack(anchor=tk.W, pady=(2, 0))
        self.logical_measurements_label = ttk.Label(
            panel, text="Measurements and alternatives will appear here.",
            wraplength=300, justify=tk.LEFT,
        )
        self.logical_measurements_label.pack(anchor=tk.W, pady=(2, 0))

    def _build_template_controls(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Template Overlay and Match Lab", padding=5)
        panel.pack(fill=tk.X, pady=(3, 5))
        template_row = ttk.Frame(panel)
        template_row.pack(fill=tk.X)
        ttk.Label(template_row, text="Digit:").pack(side=tk.LEFT)
        self.template_selector = ttk.Combobox(
            template_row, textvariable=self.template_digit,
            values=tuple(str(digit) for digit in range(10)),
            state="readonly", width=4,
        )
        self.template_selector.pack(side=tk.LEFT, padx=(4, 5))
        self.template_selector.bind("<<ComboboxSelected>>", self._on_template_selected)
        self.template_start_button = ttk.Button(
            template_row, text="Start / Restart", command=self._start_template_match,
            state=tk.DISABLED,
        )
        self.template_start_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.template_reset_button = ttk.Button(
            template_row, text="Reset Fit", command=self._reset_template_transform,
            state=tk.DISABLED,
        )
        self.template_reset_button.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(panel, text="Template variant").pack(anchor=tk.W, pady=(3, 0))
        self.template_variant_selector = ttk.Combobox(
            panel,
            textvariable=self.template_variant,
            values=(
                TEMPLATE_VARIANT_DIGIT,
                TEMPLATE_VARIANT_THIN_ZERO,
                TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT,
            ),
            state="readonly",
        )
        self.template_variant_selector.pack(fill=tk.X)
        self.template_variant_selector.bind(
            "<<ComboboxSelected>>", self._on_template_variant_selected
        )

        ttk.Label(panel, text="Offset X (pixels from centered baseline)").pack(anchor=tk.W)
        self.template_x_spinbox = ttk.Spinbox(
            panel, from_=-1, to=1, textvariable=self.template_offset_x,
            width=7, state=tk.DISABLED,
        )
        self.template_x_spinbox.pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(panel, text="Offset Y (pixels from centered baseline)").pack(anchor=tk.W)
        self.template_y_spinbox = ttk.Spinbox(
            panel, from_=-1, to=1, textvariable=self.template_offset_y,
            width=7, state=tk.DISABLED,
        )
        self.template_y_spinbox.pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(panel, text="Uniform scale multiplier (baseline = 1.00)").pack(anchor=tk.W)
        self.template_scale_control = tk.Scale(
            panel, from_=0.25, to=2.5, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.template_scale_factor, showvalue=True,
            command=self._on_template_transform, state=tk.DISABLED,
        )
        self.template_scale_control.pack(fill=tk.X)
        auto_align_row = ttk.Frame(panel)
        auto_align_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(auto_align_row, text="Search radius (pixels):").pack(side=tk.LEFT)
        self.auto_align_radius_spinbox = ttk.Spinbox(
            auto_align_row, from_=0, to=AUTO_ALIGN_MAX_RADIUS,
            textvariable=self.auto_align_radius, width=5, state=tk.DISABLED,
        )
        self.auto_align_radius_spinbox.pack(side=tk.LEFT, padx=(4, 6))
        self.auto_align_button = ttk.Button(
            auto_align_row, text="Auto-Align Offset",
            command=self._auto_align_template_offset, state=tk.DISABLED,
        )
        self.auto_align_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.auto_align_report_label = ttk.Label(
            panel, textvariable=self.auto_align_report, wraplength=300, justify=tk.LEFT
        )
        self.auto_align_report_label.pack(anchor=tk.W, pady=(2, 0))
        self.template_width_label = ttk.Label(panel, text="Width scale (0.80–1.20×)")
        self.template_width_label.pack(anchor=tk.W)
        self.template_width_spinbox = ttk.Spinbox(
            panel, from_=INDEPENDENT_SCALE_MIN, to=INDEPENDENT_SCALE_MAX,
            increment=0.05, textvariable=self.template_width_scale,
            width=7, state=tk.DISABLED,
        )
        self.template_width_spinbox.pack(anchor=tk.W, pady=(0, 2))
        self.template_height_label = ttk.Label(panel, text="Height scale (0.80–1.20×)")
        self.template_height_label.pack(anchor=tk.W)
        self.template_height_spinbox = ttk.Spinbox(
            panel, from_=INDEPENDENT_SCALE_MIN, to=INDEPENDENT_SCALE_MAX,
            increment=0.05, textvariable=self.template_height_scale,
            width=7, state=tk.DISABLED,
        )
        self.template_height_spinbox.pack(anchor=tk.W, pady=(0, 2))
        self.template_offset_x.trace_add("write", self._on_template_transform)
        self.template_offset_y.trace_add("write", self._on_template_transform)
        self.template_width_scale.trace_add("write", self._on_template_transform)
        self.template_height_scale.trace_add("write", self._on_template_transform)
        self.template_baseline_label = ttk.Label(panel, text="Baseline IoU: —")
        self.template_baseline_label.pack(anchor=tk.W)
        self.template_score_label = ttk.Label(panel, text="Current IoU: —")
        self.template_score_label.pack(anchor=tk.W)
        self.template_parameters_label = ttk.Label(
            panel, text="Start a comparison with a selected ROI.",
            wraplength=300, justify=tk.LEFT,
        )
        self.template_parameters_label.pack(anchor=tk.W, pady=(2, 0))
        self.template_carry_button = ttk.Button(
            panel, text="Carry Fit to Next ROI",
            command=self._carry_template_fit, state=tk.DISABLED,
        )
        self.template_carry_button.pack(fill=tk.X, pady=(3, 0))
        self.template_carry_label = ttk.Label(
            panel, text="No fit transfer armed.", wraplength=300, justify=tk.LEFT
        )
        self.template_carry_label.pack(anchor=tk.W, pady=(2, 0))
        self._build_template_legend(panel)

    @staticmethod
    def _build_template_legend(parent: ttk.LabelFrame) -> None:
        legend = ttk.Frame(parent)
        legend.pack(fill=tk.X, pady=(2, 0))
        for color, label in (
            (OVERLAP_COLOR, "Both"),
            (IMAGE_ONLY_COLOR, "Image only"),
            (TEMPLATE_ONLY_COLOR, "Template only"),
        ):
            tk.Label(
                legend, text="■", fg=f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}",
                font=("", 10, "bold"),
            ).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Label(legend, text=label).pack(side=tk.LEFT, padx=(0, 6))

    def _build_experiment_instructions(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Experiment Instructions", padding=5)
        panel.pack(fill=tk.X, pady=(3, 5))
        text_frame = ttk.Frame(panel)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.experiment_text = tk.Text(text_frame, height=12, width=34, wrap=tk.NONE)
        text_scroll = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=self.experiment_text.yview
        )
        self.experiment_text.configure(yscrollcommand=text_scroll.set)
        self.experiment_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        buttons = ttk.Frame(panel)
        buttons.pack(fill=tk.X, pady=(4, 0))
        self.experiment_run_button = ttk.Button(
            buttons, text="Run Experiment", command=self._run_pasted_experiment
        )
        self.experiment_run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            buttons, text="Load Example", command=self._load_experiment_example
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(
            buttons, text="Load Grouping Example",
            command=self._load_grouping_example,
        ).pack(side=tk.LEFT, padx=(4, 0))
        self.experiment_status_label = ttk.Label(
            panel, text="Paste or load a configuration; it runs only when requested.",
            wraplength=300, justify=tk.LEFT,
        )
        self.experiment_status_label.pack(anchor=tk.W, pady=(4, 2))
        progress_row = ttk.Frame(panel)
        progress_row.pack(fill=tk.X, pady=(2, 2))
        self.experiment_progress = ttk.Progressbar(
            progress_row, orient=tk.HORIZONTAL, mode="determinate", maximum=100
        )
        self.experiment_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.experiment_cancel_button = ttk.Button(
            progress_row, text="Cancel", command=self._cancel_experiment,
            state=tk.DISABLED,
        )
        self.experiment_cancel_button.pack(side=tk.LEFT, padx=(5, 0))
        self.experiment_results_label = ttk.Label(
            panel, text="No experiment results.", wraplength=300, justify=tk.LEFT
        )
        self.experiment_results_label.pack(anchor=tk.W)

    def _load_experiment_example(self) -> None:
        example = (
            "experiment=template_auto_align\n"
            "roi_left=859\n"
            "roi_top=111\n"
            "roi_right=891\n"
            "roi_bottom=149\n"
            "brightness_threshold=160\n"
            "digit=0\n"
            "template=thin_zero_independent_xy\n"
            "uniform_scale=1.00\n"
            "width_scale=0.90\n"
            "height_scale=0.85\n"
            "offset_x=0\n"
            "offset_y=0\n"
            "auto_align_radius=5\n"
        )
        self.experiment_text.delete("1.0", tk.END)
        self.experiment_text.insert("1.0", example)
        self.experiment_status_label.configure(
            text="Example loaded; click Run Experiment to execute it."
        )

    def _load_grouping_example(self) -> None:
        if self.rgb_image is None or self.roi is None or self.section_labels is None:
            self.experiment_status_label.configure(
                text="Select an ROI and find connected sections before loading a grouping example."
            )
            return
        if len(self.sections) < 5:
            self.experiment_status_label.configure(
                text="The 10 + 20 demonstration needs five selected sections. "
                     "Use a suitable ROI/threshold, or enter your own grouping configuration."
            )
            return
        width, _height = self.rgb_image.size
        seeds = []
        for section in sorted(
            self.sections, key=lambda item: (item.min_x, item.min_y, item.number)
        )[:5]:
            seed = next(
                (
                    (x, y)
                    for y in range(section.min_y, section.max_y + 1)
                    for x in range(section.min_x, section.max_x + 1)
                    if self.section_labels[y * width + x] == section.number
                ),
                None,
            )
            if seed is None:
                self.experiment_status_label.configure(
                    text="Could not resolve component seed coordinates; find sections again."
                )
                return
            seeds.append(seed)
        left, top, right, bottom = self.roi
        piece_lines = [
            f"piece_{index + 1}={x},{y}"
            for index, (x, y) in enumerate(seeds)
        ]
        example = (
            "experiment=logical_grouping\n"
            f"roi_left={left}\nroi_top={top}\nroi_right={right}\nroi_bottom={bottom}\n"
            f"brightness_threshold={self.threshold.get()}\n"
            f"pieces={';'.join(piece_lines)}\n"
            "intended_groups=group_left=piece_1,piece_2;"
            "group_middle=piece_3;group_right=piece_4,piece_5\n"
        )
        self.experiment_text.delete("1.0", tk.END)
        self.experiment_text.insert("1.0", example)
        self.experiment_status_label.configure(
            text="Grouping example loaded. Review/adjust the section seeds and intended "
                 "annotations, then click Run Experiment."
        )

    def _run_pasted_experiment(self) -> None:
        if self.experiment_job_running:
            return
        if self.rgb_image is None or self.grayscale_image is None:
            self.experiment_status_label.configure(
                text="Open an image before running an experiment."
            )
            return
        try:
            instruction_text = self.experiment_text.get("1.0", tk.END)
            experiment_name = next(
                (
                    line.partition("=")[2].strip()
                    for line in instruction_text.splitlines()
                    if line.strip().startswith("experiment=")
                ),
                "",
            )
            if experiment_name == "logical_grouping":
                config = parse_logical_grouping_config(
                    instruction_text, self.rgb_image.size
                )
            elif experiment_name == "threshold_growth":
                config = parse_threshold_growth_config(
                    instruction_text, self.rgb_image.size
                )
            else:
                config = parse_experiment_config(
                    instruction_text, self.rgb_image.size
                )
        except ValueError as error:
            self.experiment_status_label.configure(text=f"Invalid configuration: {error}")
            return

        rgb_image = self.rgb_image
        grayscale_image = self.grayscale_image
        version = self._mask_version
        cancellation = Event() if isinstance(config, ThresholdGrowthConfig) and config.auto_record else None
        self._experiment_cancel_event = cancellation
        self.experiment_job_running = True
        self.experiment_run_button.configure(state=tk.DISABLED)
        self.experiment_progress.configure(value=0)
        self.experiment_cancel_button.configure(
            state=tk.NORMAL if cancellation is not None else tk.DISABLED
        )
        self.experiment_status_label.configure(
            text=(
                "Recording growth experiment…"
                if cancellation is not None
                else "Running experiment in background…"
            )
        )
        def progress(completed: int, total: int, message: str) -> None:
            self._experiment_progress.put(
                (version, completed, max(total, 1), message)
            )

        self._submit(
            "experiment",
            version,
            lambda: self._run_experiment_config(
                experiment_name,
                config,
                rgb_image,
                grayscale_image,
                cancellation,
                progress if cancellation is not None else None,
            ),
        )

    def _cancel_experiment(self) -> None:
        if self._experiment_cancel_event is None:
            return
        self._experiment_cancel_event.set()
        self.experiment_cancel_button.configure(state=tk.DISABLED)
        self.experiment_status_label.configure(
            text="Cancel requested; stopping after the current analysis step…"
        )

    @staticmethod
    def _run_experiment_config(
        experiment_name: str,
        config: ExperimentConfig | LogicalGroupingConfig | ThresholdGrowthConfig,
        rgb_image: Image.Image,
        grayscale_image: Image.Image,
        cancel_event: Event | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> (
        TemplateAutoAlignExperimentResult
        | LogicalGroupingExperimentResult
        | ThresholdGrowthExperimentResult
        | AutomaticGrowthRecordingResult
    ):
        if experiment_name == "logical_grouping":
            if not isinstance(config, LogicalGroupingConfig):
                raise TypeError("Logical grouping configuration was not parsed.")
            return run_logical_grouping_experiment(
                config, rgb_image, grayscale_image
            )
        if experiment_name == "threshold_growth":
            if not isinstance(config, ThresholdGrowthConfig):
                raise TypeError("Threshold growth configuration was not parsed.")
            if config.auto_record:
                return record_threshold_growth_experiment(
                    config,
                    rgb_image,
                    grayscale_image,
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                )
            return run_threshold_growth_experiment(
                config, rgb_image, grayscale_image
            )
        if not isinstance(config, ExperimentConfig):
            raise TypeError("Template matching configuration was not parsed.")
        return run_template_auto_align_experiment(
            config, rgb_image, grayscale_image
        )

    def _reset_template_match(self, message: str) -> None:
        self.template_active = False
        self.template_image = None
        self.template_comparison = None
        self.template_baseline = None
        self.template_roi = None
        self.template_threshold = None
        self.template_base_scale = 1.0
        self.template_transform = None
        self.template_baseline_label.configure(text="Baseline IoU: —")
        self.template_score_label.configure(text="Current IoU: —")
        self.template_parameters_label.configure(text=message)
        self.template_start_button.configure(
            state=tk.NORMAL
            if self.binary_mask is not None and self.roi is not None
            else tk.DISABLED
        )
        self.template_reset_button.configure(state=tk.DISABLED)
        self.template_x_spinbox.configure(state=tk.DISABLED)
        self.template_y_spinbox.configure(state=tk.DISABLED)
        self.template_scale_control.configure(state=tk.DISABLED)
        self.template_width_spinbox.configure(state=tk.DISABLED)
        self.template_height_spinbox.configure(state=tk.DISABLED)
        self.template_carry_button.configure(state=tk.DISABLED)
        self.auto_align_radius_spinbox.configure(state=tk.DISABLED)
        self.auto_align_button.configure(state=tk.DISABLED)
        self.auto_align_report.set("No auto-align run yet.")
        if self.view_mode.get() == "Template Match":
            self.view_mode.set("Binary Mask")
        self._render_analysis()

    def _on_template_selected(self, _event: tk.Event | None = None) -> None:
        if (
            self.template_variant.get() != TEMPLATE_VARIANT_DIGIT
            and self.template_digit.get() != "0"
        ):
            self.template_variant.set(TEMPLATE_VARIANT_DIGIT)
        if self.template_active:
            self._reset_template_match(
                "Template changed; start again to record its baseline score."
            )

    def _on_template_variant_selected(self, _event: tk.Event | None = None) -> None:
        if self.template_variant.get() != TEMPLATE_VARIANT_DIGIT:
            self.template_digit.set("0")
        if self.template_active:
            self._reset_template_match(
                "Template variant changed; restart to record its baseline score."
            )

    def _start_template_match(self) -> None:
        if self.rgb_image is None or self.binary_mask is None or self.roi is None:
            self.status_label.configure(
                text="Load an image, select a character ROI, and wait for its threshold mask."
            )
            return
        roi_mask = self.binary_mask.crop(self.roi)
        template = load_template_variant(
            self.template_variant.get(), self.template_digit.get()
        )
        self.template_roi = self.roi
        self.template_threshold = self.threshold.get()
        self.template_base_scale = min(
            roi_mask.width / template.width, roi_mask.height / template.height
        )
        self.template_transform = TemplateTransform(self.template_base_scale)
        self.template_offset_x.set(0)
        self.template_offset_y.set(0)
        self.template_scale_factor.set(1.0)
        self.template_width_scale.set(1.0)
        self.template_height_scale.set(1.0)
        self.template_x_spinbox.configure(
            from_=-roi_mask.width, to=roi_mask.width, state=tk.NORMAL
        )
        self.template_y_spinbox.configure(
            from_=-roi_mask.height, to=roi_mask.height, state=tk.NORMAL
        )
        independent = (
            self.template_variant.get() == TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT
        )
        self.template_scale_control.configure(
            state=tk.DISABLED if independent else tk.NORMAL
        )
        self.template_width_spinbox.configure(
            state=tk.NORMAL if independent else tk.DISABLED
        )
        self.template_height_spinbox.configure(
            state=tk.NORMAL if independent else tk.DISABLED
        )
        self.template_reset_button.configure(state=tk.NORMAL)
        self.template_start_button.configure(state=tk.NORMAL)
        self.template_carry_button.configure(state=tk.NORMAL)
        self.auto_align_radius_spinbox.configure(state=tk.NORMAL)
        self.auto_align_button.configure(state=tk.NORMAL)
        self.auto_align_report.set("No auto-align run yet.")
        self.template_active = True
        baseline = self._calculate_template_comparison(0, 0, 1.0)
        self.template_baseline = baseline
        self.template_comparison = baseline
        self.template_baseline_label.configure(
            text=f"Baseline IoU: {self._format_iou(baseline.iou)}"
        )
        self.view_mode.set("Template Match")
        self._update_template_display()
        self.status_label.configure(
            text="Template baseline recorded; adjust position or scale."
        )

    @staticmethod
    def _format_iou(iou: float | None) -> str:
        return "N/A (both masks empty)" if iou is None else f"{100 * iou:.2f}%"

    def _calculate_template_comparison(
        self, offset_x: int, offset_y: int, scale_factor: float
    ) -> TemplateComparison:
        if self.binary_mask is None or self.template_roi is None:
            raise RuntimeError("A threshold mask and selected ROI are required.")
        if self.template_transform is None:
            raise RuntimeError("Template transform has not been initialized.")
        self.template_transform.offset_x = offset_x
        self.template_transform.offset_y = offset_y
        self.template_transform.scale_multiplier = scale_factor
        self.template_transform.set_independent_scales(
            self.template_width_scale.get(), self.template_height_scale.get()
        )
        roi_mask = self.binary_mask.crop(self.template_roi)
        return compare_template(
            roi_mask,
            load_template_variant(
                self.template_variant.get(), self.template_digit.get()
            ),
            offset_x=offset_x,
            offset_y=offset_y,
            scale=self.template_transform.scale,
            scale_y=self.template_transform.scale_y,
        )

    def _on_template_transform(self, *_args) -> None:
        if not self.template_active:
            return
        try:
            self.template_comparison = self._calculate_template_comparison(
                self.template_offset_x.get(),
                self.template_offset_y.get(),
                self.template_scale_factor.get(),
            )
        except (ValueError, tk.TclError) as error:
            self.status_label.configure(text=f"Invalid template adjustment: {error}")
            return
        self._update_template_display()

    def _update_template_display(self, overlay: Image.Image | None = None) -> None:
        if (
            self.rgb_image is None
            or self.template_roi is None
            or self.template_comparison is None
        ):
            return
        left, top, right, bottom = self.template_roi
        self.template_image = (
            overlay
            if overlay is not None
            else create_template_overlay(
                self.rgb_image, self.template_roi, self.template_comparison
            )
        )
        comparison = self.template_comparison
        baseline = self.template_baseline
        self.template_score_label.configure(
            text=f"Current IoU: {self._format_iou(comparison.iou)}"
        )
        baseline_text = self._format_iou(baseline.iou) if baseline else "—"
        variant = self.template_variant.get()
        width_multiplier = (
            self.template_transform.width_scale_multiplier
            if self.template_transform is not None else 1.0
        )
        height_multiplier = (
            self.template_transform.height_scale_multiplier
            if self.template_transform is not None else 1.0
        )
        self.template_parameters_label.configure(
            text=(
                f"Variant: {variant}; digit: {self.template_digit.get()}\n"
                f"Brightness threshold: {self.template_threshold}\n"
                f"ROI (left, top, right, bottom): "
                f"{format_roi_coordinates((left, top, right, bottom))}\n"
                f"Offset: x={comparison.offset_x}, y={comparison.offset_y} px\n"
                f"Width scale: {width_multiplier:.2f}× "
                f"(actual {comparison.scale:g} px/template px)\n"
                f"Height scale: {height_multiplier:.2f}× "
                f"(actual {comparison.scale_y:g} px/template px)\n"
                f"Baseline IoU: {baseline_text}; adjusted IoU: "
                f"{self._format_iou(comparison.iou)}\n"
                f"Shared/image-only/template-only pixels: {comparison.intersection_pixels}/"
                f"{comparison.image_only_pixels}/{comparison.template_only_pixels}\n"
            )
        )
        self._render_analysis()

    def _reset_template_transform(self) -> None:
        if not self.template_active:
            return
        if self.template_transform is not None:
            self.template_transform.reset()
        self.template_offset_x.set(0)
        self.template_offset_y.set(0)
        self.template_scale_factor.set(1.0)
        self.template_width_scale.set(1.0)
        self.template_height_scale.set(1.0)
        self._on_template_transform()

    def _auto_align_template_offset(self) -> None:
        if (
            not self.template_active
            or self.binary_mask is None
            or self.template_roi is None
            or self.template_transform is None
        ):
            return
        try:
            radius = self.auto_align_radius.get()
            comparison = auto_align_offset(
                self.binary_mask.crop(self.template_roi),
                load_template_variant(
                    self.template_variant.get(), self.template_digit.get()
                ),
                (self.template_offset_x.get(), self.template_offset_y.get()),
                radius,
                scale=self.template_transform.scale,
                scale_y=self.template_transform.scale_y,
            )
        except (ValueError, tk.TclError) as error:
            self.status_label.configure(text=f"Auto-align failed: {error}")
            return

        self.auto_align_report.set(
            f"Auto-align (±{comparison.search_radius}px): start "
            f"({comparison.starting_offset[0]}, {comparison.starting_offset[1]}) "
            f"IoU {self._format_iou(comparison.starting_iou)}; selected "
            f"({comparison.selected_offset[0]}, {comparison.selected_offset[1]}) "
            f"IoU {self._format_iou(comparison.selected_iou)}; improvement "
            f"{'N/A' if comparison.improvement_percentage_points is None else f'{comparison.improvement_percentage_points:+.2f} pp'}."
        )
        self.template_offset_x.set(comparison.selected_offset[0])
        self.template_offset_y.set(comparison.selected_offset[1])
        self.template_comparison = comparison.comparison
        self._update_template_display()

    def _carry_template_fit(self) -> None:
        if not self.template_active or self.template_transform is None:
            return
        settings = TemplateFitSettings(
            variant=self.template_variant.get(),
            digit=self.template_digit.get(),
            threshold=self.template_threshold if self.template_threshold is not None
            else self.threshold.get(),
            offset_x=self.template_transform.offset_x,
            offset_y=self.template_transform.offset_y,
            scale_multiplier=self.template_transform.scale_multiplier,
            width_scale_multiplier=self.template_transform.width_scale_multiplier,
            height_scale_multiplier=self.template_transform.height_scale_multiplier,
        )
        self.pending_template_fit = settings
        self.template_carry_label.configure(
            text=(
                "Next ROI will reuse variant/digit, threshold, relative pixel offset "
                f"({settings.offset_x}, {settings.offset_y}), uniform multiplier "
                f"{settings.scale_multiplier:.2f}×, width {settings.width_scale_multiplier:.2f}×, "
                f"and height {settings.height_scale_multiplier:.2f}×. Template will "
                "be recentered in the new ROI; transfer is one-shot."
            )
        )

    def _apply_pending_template_fit(self) -> None:
        settings = self.pending_template_fit
        if settings is None or self.roi is None or self.binary_mask is None:
            return
        self.pending_template_fit = None
        self.template_variant.set(settings.variant)
        self.template_digit.set(settings.digit)
        self.template_carry_label.configure(
            text=(
                "Transferred variant/digit, threshold, relative offset "
                f"({settings.offset_x}, {settings.offset_y}) px, uniform multiplier "
                f"{settings.scale_multiplier:.2f}×, width {settings.width_scale_multiplier:.2f}×, "
                f"and height {settings.height_scale_multiplier:.2f}×; template was "
                "recentered on this ROI. Adjustments were not retuned."
            )
        )
        self._start_template_match()
        self.template_offset_x.set(settings.offset_x)
        self.template_offset_y.set(settings.offset_y)
        self.template_scale_factor.set(settings.scale_multiplier)
        self.template_width_scale.set(settings.width_scale_multiplier)
        self.template_height_scale.set(settings.height_scale_multiplier)
        if self.template_transform is not None:
            roi_mask = self.binary_mask.crop(self.roi)
            template = load_template_variant(settings.variant, settings.digit)
            new_base_scale = min(
                roi_mask.width / template.width, roi_mask.height / template.height
            )
            self.template_transform = transfer_template_fit(settings, new_base_scale)
        self._on_template_transform()

    @staticmethod
    def _build_merge_legend(parent: ttk.Frame) -> None:
        legend = ttk.Frame(parent)
        legend.pack(fill=tk.X, anchor=tk.W, pady=(2, 5))
        items = (
            ("■", MERGE_THRESHOLD_ONLY_COLOR, "New at threshold only"),
            ("■", MERGE_PATH_ONLY_COLOR, "New on path only"),
            ("■", MERGE_BOTH_COLOR, "Both categories"),
            ("○", (0, 85, 255), "Seed 1"),
            ("○", (255, 136, 0), "Seed 2"),
        )
        for item_index, (marker, color, label) in enumerate(items):
            row, pair = divmod(item_index, 2)
            marker_column = pair * 2
            tk.Label(
                legend,
                text=marker,
                fg=f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}",
                font=("", 10, "bold"),
            ).grid(row=row, column=marker_column, sticky=tk.W, padx=(0, 3))
            ttk.Label(legend, text=label).grid(
                row=row, column=marker_column + 1, sticky=tk.W, padx=(0, 8)
            )

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Supported images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._mask_version += 1
        self._selecting_roi = False
        self._roi_drag_start = None
        self._selecting_merge_seeds = False
        self.roi_button.configure(text="Select ROI")
        self.merge_seed_button.configure(text="Select 2 Merge Seeds")
        self.analysis_canvas.delete("roi-preview")
        version = self._mask_version
        preview_size = (
            max(1, self.original_canvas.winfo_width() - 8),
            max(1, self.original_canvas.winfo_height() - 8),
        )
        self.threshold_scale.configure(state=tk.DISABLED)
        self.find_button.configure(state=tk.DISABLED)
        self.analyze_button.configure(state=tk.DISABLED)
        self.growth_track_button.configure(state=tk.DISABLED)
        self.roi_button.configure(state=tk.DISABLED)
        self.clear_roi_button.configure(state=tk.DISABLED)
        self.merge_seed_button.configure(state=tk.DISABLED)
        self.merge_track_button.configure(state=tk.DISABLED)
        self.template_start_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Loading image in background…")
        self._submit(
            "load",
            version,
            lambda: self._load_image_result(path, preview_size),
        )

    @staticmethod
    def _load_image_result(
            path: str, preview_size: tuple[int, int]
    ) -> tuple[str, Image.Image, Image.Image, Image.Image]:
        rgb_image, grayscale_image = load_image(path)
        preview = create_preview(rgb_image, preview_size)
        return path, rgb_image, grayscale_image, preview

    def _install_image(
            self,
            path: str,
            rgb_image: Image.Image,
            grayscale_image: Image.Image,
            preview: Image.Image,
    ) -> None:
        self.rgb_image = rgb_image
        self.grayscale_image = grayscale_image
        self.original_preview = preview
        source_path = Path(path)
        self.source_path = source_path
        self.threshold_scale.configure(state=tk.NORMAL)
        self.binary_mask = None
        self.section_image = None
        self.sections.clear()
        self.section_labels = None
        self.selected_section = None
        self._clear_logical_grouping()
        self.roi = None
        self.growth_job_running = False
        self._clear_growth_history()
        self.merge_seeds.clear()
        self.merge_result = None
        self.merge_overlay = None
        self.merge_overlay_categories = None
        self.merge_mask = None
        self.pending_template_fit = None
        self.template_carry_label.configure(text="No fit transfer armed.")
        self._reset_template_match("Select an ROI and start a template comparison.")
        self._selecting_roi = False
        self._selecting_merge_seeds = False
        self.view_mode.set("Grayscale")
        self.dimensions_label.configure(
            text=f"{rgb_image.width} × {rgb_image.height} pixels"
        )
        self.status_label.configure(text=f"Loaded: {source_path.name}")
        self._clear_tree(self.section_tree)
        self._clear_tree(self.threshold_tree)
        self._clear_logical_grouping()
        self.find_button.configure(state=tk.DISABLED)
        self.analyze_button.configure(state=tk.DISABLED)
        self.roi_button.configure(state=tk.NORMAL)
        self.clear_roi_button.configure(state=tk.DISABLED)
        self.merge_seed_button.configure(state=tk.DISABLED)
        self.merge_track_button.configure(state=tk.DISABLED)
        self.growth_max_spinbox.configure(state=tk.NORMAL)
        self.growth_increment_spinbox.configure(state=tk.NORMAL)
        self.growth_occupancy_spinbox.configure(state=tk.NORMAL)
        self.template_start_button.configure(
            state=tk.NORMAL if self.binary_mask is not None and self.roi is not None else tk.DISABLED
        )
        self.merge_info_label.configure(
            text="Find connected sections, then select two different sections to track."
        )
        self.roi_label.configure(text="ROI: full image")
        self._render_original()
        self._render_analysis()
        self._schedule_mask()

    def _render_original(self, _event: tk.Event | None = None) -> None:
        if self.original_preview is None:
            return
        available_width = max(1, self.original_canvas.winfo_width() - 8)
        available_height = max(1, self.original_canvas.winfo_height() - 8)
        scale = min(
            available_width / self.original_preview.width,
            available_height / self.original_preview.height,
            1.0,
        )
        display_size = (
            max(1, round(self.original_preview.width * scale)),
            max(1, round(self.original_preview.height * scale)),
        )
        display = self.original_preview.resize(display_size, Image.Resampling.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(display)
        self.original_canvas.delete("all")
        self.original_canvas.create_image(4, 4, anchor=tk.NW, image=self.original_photo)
        self.original_canvas.configure(
            scrollregion=(0, 0, display_size[0] + 8, display_size[1] + 8)
        )

    def _render_analysis(self, _event: tk.Event | None = None) -> None:
        self.analysis_canvas.delete("all")
        if self.rgb_image is None or self.grayscale_image is None:
            self.analysis_canvas.configure(scrollregion=(0, 0, 1, 1))
            return

        mode = self.view_mode.get()
        if mode == "Grayscale":
            image = self.grayscale_image
        elif mode == "Binary Mask":
            image = self._mask_within_roi()
        elif mode == "Merge Overlay":
            image = self.merge_overlay
        elif mode == "Merge Mask":
            image = self.merge_mask
        elif mode == "Template Match":
            image = self.template_image
        elif mode == "Growth History":
            image = self.growth_overlay
        elif mode == "Logical Grouping":
            image = self.rgb_image
        else:
            image = self.section_image
        if image is None:
            self.analysis_canvas.configure(scrollregion=(0, 0, 1, 1))
            return

        width, height = image.size
        cell = self.zoom
        self.analysis_canvas.configure(scrollregion=(0, 0, width * cell, height * cell))
        canvas_width = max(1, self.analysis_canvas.winfo_width())
        canvas_height = max(1, self.analysis_canvas.winfo_height())
        left = max(0, int(self.analysis_canvas.canvasx(0) // cell))
        top = max(0, int(self.analysis_canvas.canvasy(0) // cell))
        right = min(width, int(self.analysis_canvas.canvasx(canvas_width) // cell) + 2)
        bottom = min(height, int(self.analysis_canvas.canvasy(canvas_height) // cell) + 2)
        if right <= left or bottom <= top:
            return

        crop = image.crop((left, top, right, bottom))
        if (
            mode == "Connected Sections"
            and self.selected_section is not None
            and self.section_labels is not None
        ):
            crop_pixels = crop.load()
            label_width = self.rgb_image.width
            for crop_y, image_y in enumerate(range(top, bottom)):
                for crop_x, image_x in enumerate(range(left, right)):
                    label_index = image_y * label_width + image_x
                    if self.section_labels[label_index] == self.selected_section.number:
                        crop_pixels[crop_x, crop_y] = (0, 0, 0)
        display_width = max(1, round((right - left) * cell))
        display_height = max(1, round((bottom - top) * cell))
        crop = crop.resize(
            (display_width, display_height), Image.Resampling.NEAREST
        )
        self.rendered_image = ImageTk.PhotoImage(crop)
        self.analysis_canvas.create_image(
            left * cell, top * cell, anchor=tk.NW, image=self.rendered_image
        )

        if cell >= 6:
            x0, y0 = left * cell, top * cell
            x1, y1 = right * cell, bottom * cell
            step = max(1, round(cell))
            for x in range(round(x0), round(x1) + 1, step):
                self.analysis_canvas.create_line(x, y0, x, y1, fill="#777777")
            for y in range(round(y0), round(y1) + 1, step):
                self.analysis_canvas.create_line(x0, y, x1, y, fill="#777777")

        if mode == "Connected Sections" and self.selected_section is not None:
            section = self.selected_section
            x, y, box_width, box_height = section.bounding_rectangle
            self.analysis_canvas.create_rectangle(
                x * cell + 1,
                y * cell + 1,
                (x + box_width) * cell - 1,
                (y + box_height) * cell - 1,
                outline="#FFFF00",
                width=max(2, round(cell / 5)),
                dash=(5, 2),
            )
        if mode == "Logical Grouping":
            palette = ("#00FFFF", "#FF55FF", "#FFFF00", "#00FF88", "#FF8800")
            selected_alternative = self.logical_candidate_tree.selection()
            displaying_intended_alternative = (
                self.logical_intended_candidate is not None
                and (
                    not selected_alternative
                    or int(selected_alternative[0]) == self.logical_intended_candidate
                )
            )
            for index, group in enumerate(self._logical_display_groups()):
                bounds = self._logical_group_bounds(group)
                if bounds is None:
                    continue
                x, y, box_width, box_height = bounds
                is_intended = displaying_intended_alternative
                if self.logical_groups_are_intended and not selected_alternative:
                    is_intended = index < len(self.logical_groups)
                if self.logical_intended_group is not None and not selected_alternative:
                    is_intended = index < len(self.logical_groups) and (
                        self.logical_groups[index][0] == self.logical_intended_group
                    )
                self.analysis_canvas.create_rectangle(
                    x * cell + 2,
                    y * cell + 2,
                    (x + box_width) * cell - 2,
                    (y + box_height) * cell - 2,
                    outline="#00FF00" if is_intended else palette[index % len(palette)],
                    width=max(2, cell // 5),
                    dash=(6, 3),
                )
                self.analysis_canvas.create_text(
                    x * cell + 4,
                    y * cell + 4,
                    text="intended" if is_intended else f"group {index + 1}",
                    anchor=tk.NW,
                    fill="#00FF00" if is_intended else palette[index % len(palette)],
                )
        if mode == "Growth History" and self.growth_seed is not None:
            seed_x, seed_y = self.growth_seed
            inset = max(1, cell // 4)
            self.analysis_canvas.create_oval(
                seed_x * cell + inset,
                seed_y * cell + inset,
                (seed_x + 1) * cell - inset,
                (seed_y + 1) * cell - inset,
                outline="#FFFFFF",
                width=2,
            )
        if self.roi is not None:
            left, top, right, bottom = self.roi
            self.analysis_canvas.create_rectangle(
                left * cell,
                top * cell,
                right * cell,
                bottom * cell,
                outline="#FF00FF",
                width=max(2, cell // 5),
                dash=(5, 3),
            )
        for index, (seed_x, seed_y) in enumerate(self.merge_seeds):
            color = "#0055FF" if index == 0 else "#FF8800"
            inset = max(1, cell // 4)
            self.analysis_canvas.create_oval(
                seed_x * cell + inset,
                seed_y * cell + inset,
                (seed_x + 1) * cell - inset,
                (seed_y + 1) * cell - inset,
                outline=color,
                fill="",
                width=2,
            )

    def _mask_within_roi(self) -> Image.Image | None:
        if self.binary_mask is None:
            return None
        roi = self.roi
        if roi is None:
            return self.binary_mask
        result = Image.new("L", self.binary_mask.size, 255)
        left, top, _right, _bottom = roi
        result.paste(self.binary_mask.crop(roi), (left, top))
        return result

    def _set_zoom(self, zoom: float) -> None:
        self.zoom = zoom
        self.zoom_label.configure(text=f"{self.zoom:g}×")
        self._render_analysis()

    def _change_zoom(self, direction: int) -> None:
        self._set_zoom(next_zoom_level(self.zoom, direction))

    def _scroll_analysis_x(self, *args) -> None:
        self.analysis_canvas.xview(*args)
        self._render_analysis()

    def _scroll_analysis_y(self, *args) -> None:
        self.analysis_canvas.yview(*args)
        self._render_analysis()

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.analysis_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        self._render_analysis()

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        self.analysis_canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        self._render_analysis()

    def _canvas_pixel(self, event: tk.Event) -> tuple[int, int]:
        return (
            int(self.analysis_canvas.canvasx(event.x) // self.zoom),
            int(self.analysis_canvas.canvasy(event.y) // self.zoom),
        )

    def _on_canvas_press(self, event: tk.Event) -> None:
        x, y = self._canvas_pixel(event)
        if self._selecting_roi:
            if self.rgb_image is not None and 0 <= x < self.rgb_image.width and 0 <= y < self.rgb_image.height:
                self._roi_drag_start = (x, y)
                self._draw_roi_preview(x, y, x + 1, y + 1)
            return
        if self._selecting_merge_seeds:
            self._select_merge_seed((x, y))
            return
        if self.view_mode.get() == "Connected Sections":
            self._select_section_at_pixel(x, y)
        self._on_pixel_click(event)

    def _select_section_at_pixel(self, x: int, y: int) -> None:
        if (
            self.section_labels is None
            or self.rgb_image is None
            or not (0 <= x < self.rgb_image.width and 0 <= y < self.rgb_image.height)
        ):
            return
        label = self.section_labels[y * self.rgb_image.width + x]
        if label:
            self.section_tree.selection_set(str(label))
            self._on_section_selected(None)
        else:
            self.section_tree.selection_remove(*self.section_tree.selection())
            self._on_section_selected(None)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if not self._selecting_roi or self._roi_drag_start is None:
            return
        if self.rgb_image is None:
            return
        x, y = self._canvas_pixel(event)
        width, height = self.rgb_image.size
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        start_x, start_y = self._roi_drag_start
        self._draw_roi_preview(
            min(start_x, x), min(start_y, y),
            max(start_x, x) + 1, max(start_y, y) + 1,
        )

    def _on_canvas_release(self, event: tk.Event) -> None:
        if not self._selecting_roi or self._roi_drag_start is None:
            return
        if self.rgb_image is None:
            return
        x, y = self._canvas_pixel(event)
        width, height = self.rgb_image.size
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        start_x, start_y = self._roi_drag_start
        self._roi_drag_start = None
        self._selecting_roi = False
        self.roi_button.configure(text="Select ROI")
        self.analysis_canvas.delete("roi-preview")
        roi = (
            min(start_x, x),
            min(start_y, y),
            max(start_x, x) + 1,
            max(start_y, y) + 1,
        )
        self.roi = roi
        self._reset_template_match("ROI changed; start a new template comparison.")
        self.roi_label.configure(
            text=f"ROI: x={roi[0]}..{roi[2] - 1}, "
            f"y={roi[1]}..{roi[3] - 1}"
        )
        self.clear_roi_button.configure(state=tk.NORMAL)
        self.find_button.configure(state=tk.NORMAL)
        self.analyze_button.configure(state=tk.NORMAL)
        self._reset_merge_tracking()
        self._invalidate_region_analysis()
        if self.pending_template_fit is not None:
            if self.threshold.get() != self.pending_template_fit.threshold:
                self.threshold.set(self.pending_template_fit.threshold)
                self._on_threshold_change(str(self.pending_template_fit.threshold))
            elif self.binary_mask is not None:
                self._apply_pending_template_fit()
        self._render_analysis()

    def _draw_roi_preview(self, left: int, top: int, right: int, bottom: int) -> None:
        if self._roi_rectangle_id is not None:
            self.analysis_canvas.delete(self._roi_rectangle_id)
        self._roi_rectangle_id = self.analysis_canvas.create_rectangle(
            left * self.zoom,
            top * self.zoom,
            right * self.zoom,
            bottom * self.zoom,
            outline="#FF00FF",
            width=max(2, self.zoom // 5),
            dash=(5, 3),
            tags="roi-preview",
        )

    def _toggle_roi_selection(self) -> None:
        if self.rgb_image is None:
            return
        self._selecting_roi = not self._selecting_roi
        self._selecting_merge_seeds = False
        self.merge_seed_button.configure(text="Select 2 Merge Seeds")
        self.roi_button.configure(text="Cancel ROI" if self._selecting_roi else "Select ROI")
        if self._selecting_roi:
            self.status_label.configure(text="Drag a rectangle around the image region to analyze.")
        else:
            self.analysis_canvas.delete("roi-preview")
            self.status_label.configure(text="ROI selection canceled.")

    def _clear_roi(self) -> None:
        if self.roi is None:
            return
        self.roi = None
        self._reset_template_match("ROI cleared; select a character ROI to compare.")
        self.roi_label.configure(text="ROI: full image")
        self.clear_roi_button.configure(state=tk.DISABLED)
        self._reset_merge_tracking()
        self._invalidate_region_analysis()
        self._render_analysis()

    def _reset_merge_tracking(self) -> None:
        self.merge_seeds.clear()
        self.merge_result = None
        self.merge_overlay = None
        self.merge_overlay_categories = None
        self.merge_mask = None
        if self.view_mode.get() in ("Merge Overlay", "Merge Mask"):
            self.view_mode.set("Binary Mask")
        self._selecting_merge_seeds = False
        self.merge_seed_button.configure(
            text="Select 2 Merge Seeds",
            state=tk.NORMAL if self.section_labels is not None else tk.DISABLED,
        )
        self.merge_track_button.configure(state=tk.DISABLED)
        self.merge_info_label.configure(
            text="Find connected sections, then select two different sections to track."
        )

    def _invalidate_region_analysis(self) -> None:
        self._mask_version += 1
        self.sections.clear()
        self.view_mode.set("Binary Mask")
        self.section_labels = None
        self.section_image = None
        self.selected_section = None
        self._clear_growth_history()
        self._clear_logical_grouping()
        self._clear_tree(self.section_tree)
        self.find_button.configure(state=tk.DISABLED)
        self.analyze_button.configure(state=tk.DISABLED)
        self.merge_seed_button.configure(state=tk.DISABLED)
        if self.binary_mask is None and self.grayscale_image is not None:
            self._schedule_mask()
        elif self.binary_mask is not None:
            self.find_button.configure(state=tk.NORMAL)
            self.analyze_button.configure(state=tk.NORMAL)
            self.template_start_button.configure(
                state=tk.NORMAL if self.roi is not None else tk.DISABLED
            )
        self.status_label.configure(text="Region changed; find sections again to select seeds.")

    def _clear_growth_history(self) -> None:
        self.growth_job_running = False
        self._growth_job_version = None
        self.growth_history = None
        self.growth_overlay = None
        self.growth_seed = None
        self.growth_frame_index = 0
        self.growth_step_label.configure(text="No growth history.")
        self.growth_info_label.configure(
            text="Find connected sections inside an ROI to begin."
        )
        self.growth_previous_button.configure(state=tk.DISABLED)
        self.growth_next_button.configure(state=tk.DISABLED)
        if self.view_mode.get() == "Growth History":
            self.view_mode.set("Binary Mask")
        self._update_growth_track_button()

    def _clear_logical_grouping(self) -> None:
        self.logical_pieces.clear()
        self.logical_groups.clear()
        self.logical_intended_group = None
        self.logical_analysis = None
        self.logical_candidate_index = 0
        self.logical_intended_candidate = None
        self.logical_groups_are_intended = False
        self.logical_intended_groups = None
        self.logical_intended_label.configure(text="No intended grouping recorded.")
        self.logical_candidate_tree_ids.clear()
        self._clear_tree(self.logical_piece_tree)
        self._clear_tree(self.logical_group_tree)
        self._clear_tree(self.logical_candidate_tree)
        self.logical_add_piece_button.configure(state=tk.DISABLED)
        self.logical_create_group_button.configure(state=tk.DISABLED)
        self.logical_remove_group_button.configure(state=tk.DISABLED)
        self.logical_mark_intended_button.configure(state=tk.DISABLED)
        self.logical_mark_candidate_button.configure(state=tk.DISABLED)
        self.logical_measurements_label.configure(
            text="Combine sections into pieces to calculate measurements."
        )
        if self.view_mode.get() == "Logical Grouping":
            self.view_mode.set("Binary Mask")

    def _update_growth_track_button(self) -> None:
        enabled = (
            self.roi is not None
            and self.section_labels is not None
            and self.selected_section is not None
            and not self.growth_job_running
        )
        self.growth_track_button.configure(
            state=tk.NORMAL if enabled else tk.DISABLED
        )

    def _toggle_merge_seed_selection(self) -> None:
        if self.section_labels is None:
            return
        if self.roi is None:
            self.status_label.configure(text="Select an ROI before choosing merge seeds.")
            return
        self._selecting_merge_seeds = not self._selecting_merge_seeds
        self._selecting_roi = False
        self.roi_button.configure(text="Select ROI")
        if not self._selecting_merge_seeds:
            self.merge_seed_button.configure(text="Select 2 Merge Seeds")
            self.status_label.configure(text="Merge seed selection canceled.")
            return
        self.merge_seeds.clear()
        self.merge_result = None
        self.merge_overlay = None
        self.merge_overlay_categories = None
        self.merge_track_button.configure(state=tk.DISABLED)
        self.view_mode.set("Binary Mask")
        self.merge_seed_button.configure(text="Cancel Seed Selection")
        self.merge_info_label.configure(
            text="Click two selected pixels in different sections at the current threshold."
        )
        self.status_label.configure(text="Select the first section, then a different second section.")
        self._render_analysis()

    def _select_merge_seed(self, seed: tuple[int, int]) -> None:
        if self.rgb_image is None or self.section_labels is None:
            return
        x, y = seed
        width, height = self.rgb_image.size
        if not (0 <= x < width and 0 <= y < height):
            return
        if self.roi is not None:
            left, top, right, bottom = self.roi
            if not (left <= x < right and top <= y < bottom):
                self.status_label.configure(text="Choose seed pixels inside the selected ROI.")
                return
        label = self.section_labels[y * width + x]
        if not label:
            self.status_label.configure(text="Choose a black, selected pixel in the current mask.")
            return
        if self.merge_seeds:
            first_x, first_y = self.merge_seeds[0]
            first_label = self.section_labels[first_y * width + first_x]
            if label == first_label:
                self.status_label.configure(text="Choose a pixel in a different connected section.")
                return
        self.merge_seeds.append(seed)
        self._render_analysis()
        if len(self.merge_seeds) == 2:
            self._selecting_merge_seeds = False
            self.merge_seed_button.configure(text="Select 2 Merge Seeds")
            self.merge_track_button.configure(state=tk.NORMAL)
            self.status_label.configure(text="Two distinct starting sections selected.")
            self.merge_info_label.configure(
                text=f"Seeds: ({self.merge_seeds[0][0]}, {self.merge_seeds[0][1]}) and "
                     f"({self.merge_seeds[1][0]}, {self.merge_seeds[1][1]}). "
                     "Track their first merge."
            )
        else:
            self.status_label.configure(text="First section selected; choose a different second section.")

    def _on_pixel_click(self, event: tk.Event) -> None:
        if self.rgb_image is None or self.grayscale_image is None:
            return
        x, y = self._canvas_pixel(event)
        if not (0 <= x < self.rgb_image.width and 0 <= y < self.rgb_image.height):
            return
        info = get_pixel_info(self.rgb_image, self.grayscale_image, x, y)
        self.pixel_info_label.configure(
            text=(
                f"Pixel ({info['x']}, {info['y']})\n"
                f"RGB: ({info['red']}, {info['green']}, {info['blue']})\n"
                f"Hex: {info['hex']}\n"
                f"Brightness: {info['brightness']}"
            )
        )

    def _on_threshold_change(self, _value: str) -> None:
        if self.grayscale_image is None:
            return
        self.view_mode.set("Binary Mask")
        self._clear_growth_history()
        self._clear_logical_grouping()
        self._reset_template_match("Threshold changed; restart comparison at this threshold.")
        self.binary_mask = None
        self.section_image = None
        self.sections.clear()
        self.section_labels = None
        self.selected_section = None
        self._update_growth_track_button()
        self._reset_merge_tracking()
        self._clear_tree(self.section_tree)
        self.find_button.configure(state=tk.DISABLED)
        self.analyze_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Updating threshold mask…")
        self._render_analysis()
        self._schedule_mask()

    def _schedule_mask(self) -> None:
        self._mask_version += 1
        version = self._mask_version
        if self._mask_after_id is not None:
            self.root.after_cancel(self._mask_after_id)
        self._mask_after_id = self.root.after(120, self._start_mask_job, version)

    def _start_mask_job(self, version: int) -> None:
        self._mask_after_id = None
        if self.grayscale_image is None:
            return
        grayscale_image = self.grayscale_image
        threshold = self.threshold.get()
        self._submit(
            "mask",
            version,
            lambda: create_binary_mask(grayscale_image, threshold),
        )

    def _find_sections(self) -> None:
        mask = self._mask_within_roi()
        if mask is None:
            return
        version = self._mask_version
        self.status_label.configure(text="Finding connected sections…")
        self.find_button.configure(state=tk.DISABLED)
        self._submit("sections", version, lambda: self._section_results(mask))

    def _track_section_growth(self) -> None:
        grayscale_image = self.grayscale_image
        rgb_image = self.rgb_image
        roi = self.roi
        section_labels = self.section_labels
        section = self.selected_section
        if (
            grayscale_image is None
            or rgb_image is None
            or roi is None
            or section_labels is None
            or section is None
        ):
            self.status_label.configure(
                text="Select an ROI, find connected sections, and select one section."
            )
            return
        if grayscale_image is None or roi is None:
            return
        try:
            maximum = int(self.growth_max_spinbox.get())
            increment = int(self.growth_increment_spinbox.get())
            occupancy_limit = float(self.growth_occupancy_spinbox.get())
        except ValueError:
            self.status_label.configure(
                text="Enter whole-number thresholds and a valid background limit."
            )
            return
        start = self.threshold.get()
        if not start <= maximum <= 255:
            self.status_label.configure(
                text="Maximum growth threshold must be at least the starting threshold."
            )
            return
        if not 1 <= increment <= 255:
            self.status_label.configure(text="Threshold increment must be from 1 to 255.")
            return
        if not 0 < occupancy_limit <= 100:
            self.status_label.configure(text="Background limit must be greater than 0 and at most 100%.")
            return

        width, _height = rgb_image.size
        seed: tuple[int, int] | None = None
        for y in range(section.min_y, section.max_y + 1):
            for x in range(section.min_x, section.max_x + 1):
                if section_labels[y * width + x] == section.number:
                    seed = (x, y)
                    break
            if seed is not None:
                break
        if seed is None:
            self.status_label.configure(
                text="Could not find a seed pixel for the selected section; find sections again."
            )
            return

        stable_seed = seed
        version = self._mask_version
        self.growth_job_running = True
        self._growth_job_version = version
        self.growth_track_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Recording threshold growth in background…")
        self._submit(
            "growth",
            version,
            lambda: analyze_threshold_growth(
                grayscale_image,
                roi,
                stable_seed,
                start,
                maximum,
                increment,
                occupancy_limit,
            ),
        )

    def _install_growth_history(self, history: ThresholdGrowthHistory) -> None:
        self.growth_job_running = False
        self._growth_job_version = None
        self.growth_history = history
        self.growth_seed = history.frames[0].seed
        self.growth_track_button.configure(
            state=tk.NORMAL if self.selected_section is not None else tk.DISABLED
        )
        self._show_growth_frame(0)
        self.status_label.configure(text="Threshold growth history complete.")

    def _install_threshold_growth_result(
        self, result: ThresholdGrowthExperimentResult
    ) -> None:
        config = result.config
        self._mask_version += 1
        self.roi = config.roi
        self.threshold.set(config.brightness_threshold)
        if self.grayscale_image is None:
            raise RuntimeError("The grayscale source image is no longer available.")
        self.binary_mask = create_binary_mask(
            self.grayscale_image, config.brightness_threshold
        )
        self.sections = list(result.sections)
        self.section_labels = result.section_labels
        self.section_image = result.section_image
        self.selected_section = None
        self.merge_seeds.clear()
        self.merge_analysis = None
        self.merge_overlay = None
        self.merge_mask = None
        self.merge_seed_button.configure(state=tk.DISABLED)
        self.merge_track_button.configure(state=tk.DISABLED)
        self._clear_growth_history()
        self._clear_logical_grouping()
        self._reset_template_match(
            "Threshold Growth History uses the pasted ROI and threshold."
        )
        self.growth_max_threshold.set(config.maximum_threshold)
        self.growth_threshold_increment.set(config.threshold_increment)
        self.growth_occupancy_limit.set(config.background_limit_percentage)
        self.roi_label.configure(text=f"ROI: {format_roi_coordinates(config.roi)}")
        self._populate_sections()
        self.find_button.configure(state=tk.NORMAL)
        self.analyze_button.configure(state=tk.NORMAL)
        self.template_start_button.configure(state=tk.NORMAL)
        self.merge_seed_button.configure(state=tk.NORMAL)
        self._render_analysis()
        self._center_analysis_on_roi(config.roi, config.auto_zoom)

        if config.seed is not None:
            if result.history is None:
                raise RuntimeError("Seeded growth experiment returned no history.")
            seed_x, seed_y = config.seed
            label = result.section_labels[seed_y * result.image_size[0] + seed_x]
            self.selected_section = next(
                section for section in self.sections if section.number == label
            )
            self.section_tree.selection_set(str(label))
            self.section_tree.see(str(label))
            self._update_growth_track_button()
            self._install_growth_history(result.history)
            self._center_analysis_on_roi(config.roi, config.auto_zoom)
            if config.auto_record and self.growth_history is not None:
                self._show_growth_frame(len(self.growth_history.frames) - 1)
            self.experiment_status_label.configure(
                text=(
                    f"Threshold growth complete for seed {config.seed}; "
                    f"{len(self.sections)} connected sections found."
                )
            )
            return

        self.view_mode.set("Connected Sections")
        self._render_analysis()
        self._update_growth_track_button()
        section_count = len(self.sections)
        if section_count:
            status = (
                f"Found {section_count} sections at threshold "
                f"{config.brightness_threshold}. Select the intended section in "
                "Connected Sections, then click Track Selected Section to run growth."
            )
        else:
            status = (
                f"No connected sections were found in the ROI at threshold "
                f"{config.brightness_threshold}."
            )
        self.status_label.configure(text=status)
        self.experiment_status_label.configure(text=status)
        self._center_analysis_on_roi(config.roi, config.auto_zoom)

    def _center_analysis_on_roi(self, roi: Roi, auto_zoom: bool = False) -> None:
        if self.rgb_image is None:
            return
        self.root.update_idletasks()
        canvas_width = max(1, self.analysis_canvas.winfo_width())
        canvas_height = max(1, self.analysis_canvas.winfo_height())
        image_width, image_height = self.rgb_image.size
        left, top, right, bottom = roi
        if auto_zoom:
            self.zoom = fit_zoom_to_roi(
                roi,
                (canvas_width, canvas_height),
            )
            self.zoom_label.configure(text=f"{self.zoom:g}×")
            self._render_analysis()
        total_width = max(1, image_width * self.zoom)
        total_height = max(1, image_height * self.zoom)
        center_x = (left + right) * self.zoom / 2
        center_y = (top + bottom) * self.zoom / 2
        self.analysis_canvas.xview_moveto(
            max(0.0, min(1.0, (center_x - canvas_width / 2) / total_width))
        )
        self.analysis_canvas.yview_moveto(
            max(0.0, min(1.0, (center_y - canvas_height / 2) / total_height))
        )
        self._render_analysis()

    def _show_growth_frame(self, index: int) -> None:
        if self.growth_history is None or self.rgb_image is None:
            return
        self.growth_frame_index = min(
            max(index, 0), len(self.growth_history.frames) - 1
        )
        frame = self.growth_history.frames[self.growth_frame_index]
        self.growth_overlay = create_growth_overlay(self.rgb_image, frame)
        self.view_mode.set("Growth History")
        self.growth_step_label.configure(
            text=(
                f"Threshold {frame.threshold} "
                f"({self.growth_frame_index + 1}/{len(self.growth_history.frames)})"
            )
        )
        bounds = (
            "none" if frame.bounding_rectangle is None
            else f"(left, top, width, height)={frame.bounding_rectangle}"
        )
        centroid = (
            "none" if frame.centroid is None
            else f"({frame.centroid[0]:.2f}, {frame.centroid[1]:.2f})"
        )
        event_text = " ".join(frame.events) if frame.events else "No significant event."
        background_note = (
            " BACKGROUND-DOMINATED." if frame.background_dominated else ""
        )
        self.growth_info_label.configure(
            text=(
                f"Stable seed pixel: {frame.seed}.\n"
                f"Tracked pixels: {frame.pixel_count}; newly admitted this step: "
                f"{frame.newly_admitted_count}; other starting sections absorbed: "
                f"{frame.absorbed_start_sections}.\n"
                f"Bounding rectangle {bounds}; centroid {centroid}.\n"
                f"ROI selected: {frame.selected_pixel_count}/{frame.roi_pixel_count} "
                f"({frame.occupancy_percentage:.1f}%).{background_note}\n{event_text}"
            )
        )
        self.growth_previous_button.configure(
            state=tk.NORMAL if self.growth_frame_index > 0 else tk.DISABLED
        )
        self.growth_next_button.configure(
            state=(
                tk.NORMAL
                if self.growth_frame_index + 1 < len(self.growth_history.frames)
                else tk.DISABLED
            )
        )
        self._render_analysis()

    def _step_growth(self, amount: int) -> None:
        if self.growth_history is not None:
            self._show_growth_frame(self.growth_frame_index + amount)

    @staticmethod
    def _section_results(
            mask: Image.Image,
    ) -> tuple[list[ConnectedSection], array, Image.Image]:
        sections, labels = find_connected_sections(mask)
        section_image = colorize_sections(labels, mask.size)
        return sections, labels, section_image

    def _analyze_thresholds(self) -> None:
        if self.grayscale_image is None:
            return
        grayscale_image = (
            self.grayscale_image.crop(self.roi) if self.roi is not None
            else self.grayscale_image
        )
        version = self._mask_version
        self.status_label.configure(text="Analyzing thresholds in background…")
        self.analyze_button.configure(state=tk.DISABLED)
        self._submit("thresholds", version, lambda: self._threshold_results(grayscale_image))

    def _track_roi_merge(self) -> None:
        if self.grayscale_image is None or self.rgb_image is None or len(self.merge_seeds) != 2:
            return
        try:
            max_threshold = int(self.merge_limit_spinbox.get())
        except ValueError:
            self.status_label.configure(text="Enter a whole-number merge limit from 0 to 255.")
            return
        if not self.threshold.get() <= max_threshold <= 255:
            self.status_label.configure(
                text="Maximum merge threshold must be at least the starting threshold."
            )
            return
        grayscale_image = self.grayscale_image
        rgb_image = self.rgb_image
        seeds = tuple(self.merge_seeds)
        roi = self.roi
        start_threshold = self.threshold.get()
        version = self._mask_version
        self.merge_track_button.configure(state=tk.DISABLED)
        self.merge_seed_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Tracking selected sections through thresholds…")
        self._submit(
            "merge",
            version,
            lambda: self._merge_results(
                rgb_image, grayscale_image, start_threshold, max_threshold, seeds, roi
            ),
        )

    @staticmethod
    def _merge_results(
            rgb_image: Image.Image,
            grayscale_image: Image.Image,
            start_threshold: int,
            max_threshold: int,
            seeds: tuple[tuple[int, int], ...],
            roi: Roi | None,
    ) -> tuple[MergeAnalysis, Image.Image, Image.Image, MergeOverlayCategories]:
        result = analyze_component_merge(
            grayscale_image, start_threshold, seeds[0], seeds[1],
            max_threshold=max_threshold, roi=roi,
        )
        overlay, categories = create_merge_overlay(rgb_image, result)
        display_threshold = (
            result.merge_threshold
            if result.merge_threshold is not None
            else result.max_threshold
        )
        merge_mask = create_binary_mask(grayscale_image, display_threshold)
        if roi is not None:
            roi_mask = Image.new("L", merge_mask.size, 255)
            left, top, _right, _bottom = roi
            roi_mask.paste(merge_mask.crop(roi), (left, top))
            merge_mask = roi_mask
        return result, overlay, merge_mask, categories

    @staticmethod
    def _threshold_results(grayscale_image: Image.Image) -> list[tuple[int, int, int, int]]:
        results = []
        for threshold in (40, 80, 120, 160, 200, 240):
            mask = create_binary_mask(grayscale_image, threshold)
            sections, _labels = find_connected_sections(mask)
            selected = sum(section.pixel_count for section in sections)
            largest = max((section.pixel_count for section in sections), default=0)
            results.append((threshold, selected, len(sections), largest))
        return results

    def _submit(self, kind: str, version: int, operation) -> None:
        future: Future = self._executor.submit(operation)

        def completed(done: Future) -> None:
            try:
                result, error = done.result(), None
            except Exception as exception:
                result, error = None, exception
            self._worker_results.put((kind, version, result, error))

        future.add_done_callback(completed)

    def _install_experiment_result(
        self, result: TemplateAutoAlignExperimentResult
    ) -> None:
        config = result.config
        self._mask_version += 1
        if self._mask_after_id is not None:
            self.root.after_cancel(self._mask_after_id)
            self._mask_after_id = None
        self.template_active = False
        self._clear_growth_history()
        self._clear_logical_grouping()
        self.roi = config.roi
        self.threshold.set(config.brightness_threshold)
        self.binary_mask = result.binary_mask
        self.sections.clear()
        self.section_labels = None
        self.section_image = None
        self.selected_section = None
        self._update_growth_track_button()
        self._clear_tree(self.section_tree)
        self._clear_tree(self.threshold_tree)
        self._reset_merge_tracking()
        self.merge_seed_button.configure(state=tk.DISABLED)
        self.merge_track_button.configure(state=tk.DISABLED)
        self._selecting_roi = False
        self._selecting_merge_seeds = False
        self.roi_button.configure(text="Select ROI")
        self.clear_roi_button.configure(state=tk.NORMAL)
        self.roi_label.configure(
            text=f"ROI: {format_roi_coordinates(config.roi)}"
        )

        self.template_variant.set(config.template_variant)
        self.template_digit.set(config.digit)
        self.template_roi = config.roi
        self.template_threshold = config.brightness_threshold
        template = load_template_variant(config.template_variant, config.digit)
        roi_width = config.roi[2] - config.roi[0]
        roi_height = config.roi[3] - config.roi[1]
        self.template_base_scale = min(
            roi_width / template.width, roi_height / template.height
        )
        self.template_transform = TemplateTransform(
            self.template_base_scale,
            offset_x=config.offset_x,
            offset_y=config.offset_y,
            scale_multiplier=config.uniform_scale,
            width_scale_multiplier=config.width_scale,
            height_scale_multiplier=config.height_scale,
        )
        self.template_offset_x.set(config.offset_x)
        self.template_offset_y.set(config.offset_y)
        self.template_scale_factor.set(config.uniform_scale)
        self.template_width_scale.set(config.width_scale)
        self.template_height_scale.set(config.height_scale)
        self.auto_align_radius.set(config.auto_align_radius)
        self.template_baseline = result.initial_comparison
        self.template_comparison = result.alignment.comparison
        self.template_image = result.overlay
        self.view_mode.set("Template Match")
        self.template_active = True
        self.template_start_button.configure(state=tk.NORMAL)
        self.template_reset_button.configure(state=tk.NORMAL)
        self.template_carry_button.configure(state=tk.NORMAL)
        self.auto_align_radius_spinbox.configure(state=tk.NORMAL)
        self.auto_align_button.configure(state=tk.NORMAL)
        independent = config.template_variant == TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT
        self.template_scale_control.configure(
            state=tk.DISABLED if independent else tk.NORMAL
        )
        self.template_width_spinbox.configure(
            state=tk.NORMAL if independent else tk.DISABLED
        )
        self.template_height_spinbox.configure(
            state=tk.NORMAL if independent else tk.DISABLED
        )
        self.template_baseline_label.configure(
            text=f"Baseline IoU: {self._format_iou(result.initial_comparison.iou)}"
        )
        self.auto_align_report.set(
            f"Start ({config.offset_x}, {config.offset_y}) IoU "
            f"{self._format_iou(result.alignment.starting_iou)}; selected "
            f"({result.alignment.selected_offset[0]}, "
            f"{result.alignment.selected_offset[1]}) IoU "
            f"{self._format_iou(result.alignment.selected_iou)}; improvement "
            f"{'N/A' if result.alignment.improvement_percentage_points is None else f'{result.alignment.improvement_percentage_points:+.2f} pp'}."
        )
        self.template_carry_label.configure(text="No fit transfer armed.")
        self.pending_template_fit = None
        self._update_template_display(result.overlay)
        self.experiment_results_label.configure(
            text=self._format_experiment_result(result)
        )
        self.experiment_status_label.configure(text="Experiment completed.")
        self.status_label.configure(text="Paste-and-run template experiment complete.")

    def _install_logical_grouping_result(
        self, result: LogicalGroupingExperimentResult
    ) -> None:
        config = result.config
        self._mask_version += 1
        if self._mask_after_id is not None:
            self.root.after_cancel(self._mask_after_id)
            self._mask_after_id = None
        self._clear_growth_history()
        self._reset_template_match("Logical grouping experiment selected a new ROI/threshold.")
        self._reset_merge_tracking()
        self.roi = config.roi
        self.threshold.set(config.brightness_threshold)
        self.binary_mask = create_binary_mask(self.grayscale_image, config.brightness_threshold)
        self.sections = list(result.sections)
        self.section_labels = result.section_labels
        self.section_image = result.section_image
        self.selected_section = None
        self._clear_tree(self.section_tree)
        self._populate_sections()
        self.roi_label.configure(text=f"ROI: {format_roi_coordinates(config.roi)}")
        self.clear_roi_button.configure(state=tk.NORMAL)
        self.find_button.configure(state=tk.NORMAL)
        self.analyze_button.configure(state=tk.NORMAL)
        self._clear_logical_grouping()
        self.logical_pieces = {
            piece.identifier: piece for piece in result.analysis.pieces
        }
        self.logical_groups = list(config.intended_groups)
        self.logical_groups_are_intended = True
        self.logical_intended_groups = tuple(
            piece_ids for _group_id, piece_ids in config.intended_groups
        )
        self.logical_analysis = result.analysis
        self._refresh_logical_grouping()
        self.view_mode.set("Logical Grouping")
        self.logical_mark_intended_button.configure(
            state=tk.NORMAL if self.logical_groups else tk.DISABLED
        )
        self.logical_mark_candidate_button.configure(
            state=tk.NORMAL if result.analysis.candidates else tk.DISABLED
        )
        self.experiment_results_label.configure(
            text=self._format_logical_grouping_result(result)
        )
        self.experiment_status_label.configure(
            text="Logical grouping experiment completed; annotations remain user-provided."
        )
        self.status_label.configure(text="Paste-and-run logical grouping experiment complete.")
        self._render_analysis()

    @staticmethod
    def _format_logical_grouping_result(
        result: LogicalGroupingExperimentResult,
    ) -> str:
        config = result.config
        intended = "; ".join(
            f"{name}={','.join(piece_ids)}"
            for name, piece_ids in config.intended_groups
        )
        measurements = [
            f"{item.first_piece}/{item.second_piece}: gap "
            f"{item.horizontal_gap}px horizontal, {item.vertical_gap}px vertical; "
            f"overlap {item.vertical_overlap}px; baseline difference "
            f"{item.baseline_difference}px; size ratios "
            f"{item.width_ratio:.2f}×/{item.height_ratio:.2f}×; "
            f"normalized horizontal gap {item.normalized_horizontal_gap:.2f}× height."
            for item in result.analysis.pair_measurements
        ]
        paired_groups = tuple(
            piece_ids for _group_id, piece_ids in config.intended_groups
            if len(piece_ids) == 2
        )
        consistency = compare_grouping_consistency(
            result.analysis.pair_measurements, paired_groups
        )
        alternatives = []
        for index, candidate in enumerate(result.analysis.candidates, start=1):
            pairs = (
                "all separate"
                if not candidate.proposed_pairs
                else " / ".join(
                    f"{first}+{second}" for first, second in candidate.proposed_pairs
                )
            )
            alternatives.append(f"{index}: {pairs}")
        return (
            "Experiment: logical_grouping\n"
            f"Image: {result.image_size[0]}×{result.image_size[1]}\n"
            f"ROI: {format_roi_coordinates(config.roi)}\n"
            f"Threshold: {config.brightness_threshold}\n"
            f"Pieces: {len(result.analysis.pieces)}; intended grouping annotations: {intended}\n"
            f"Typical piece height: {result.analysis.typical_piece_height:.2f}px\n"
            f"Measurements: {'; '.join(measurements) if measurements else 'No piece pairs.'}\n"
            f"Grouping consistency: {'; '.join(consistency) if consistency else 'Need at least two annotated two-piece groups.'}\n"
            f"Alternatives: {'; '.join(alternatives)}"
        )

    @classmethod
    def _format_experiment_result(
        cls, result: TemplateAutoAlignExperimentResult
    ) -> str:
        config = result.config
        final = result.alignment.comparison
        improvement = result.alignment.improvement_percentage_points
        return (
            f"Experiment: {config.experiment}\n"
            f"Image: {result.image_size[0]}×{result.image_size[1]} pixels\n"
            f"ROI: {format_roi_coordinates(config.roi)}\n"
            f"Brightness threshold: {config.brightness_threshold}\n"
            f"Digit/template: {config.digit} / {config.template_variant}\n"
            f"Scales (uniform/width/height): {config.uniform_scale:.2f} / "
            f"{config.width_scale:.2f} / {config.height_scale:.2f}\n"
            f"Initial offset: ({config.offset_x}, {config.offset_y}); "
            f"initial IoU: {cls._format_iou(result.initial_comparison.iou)}\n"
            f"Selected offset: {result.alignment.selected_offset}; "
            f"final IoU: {cls._format_iou(result.alignment.selected_iou)}\n"
            f"Improvement: {'N/A' if improvement is None else f'{improvement:+.2f} percentage points'}\n"
            f"Shared pixels: {final.intersection_pixels}; "
            f"image-only: {final.image_only_pixels}; "
            f"template-only: {final.template_only_pixels}"
        )

    def _poll_worker_results(self) -> None:
        if self._closed:
            return
        while True:
            try:
                version, completed, total, message = self._experiment_progress.get_nowait()
            except queue.Empty:
                break
            if version == self._mask_version and self.experiment_job_running:
                self.experiment_progress.configure(
                    value=min(100, 100 * completed / max(total, 1))
                )
                self.experiment_status_label.configure(text=message)
        while True:
            try:
                kind, version, result, error = self._worker_results.get_nowait()
            except queue.Empty:
                break
            if version != self._mask_version:
                if kind == "experiment":
                    self.experiment_job_running = False
                    self.experiment_run_button.configure(state=tk.NORMAL)
                    self._experiment_cancel_event = None
                    self.experiment_cancel_button.configure(state=tk.DISABLED)
                    if isinstance(result, AutomaticGrowthRecordingResult):
                        self.experiment_progress.configure(value=100)
                        self.experiment_status_label.configure(
                            text=(
                                f"Recording complete, but the analysis view changed. "
                                f"Output: {result.output_directory}"
                            )
                        )
                        continue
                    self.experiment_status_label.configure(
                        text="Experiment result discarded because the image or analysis region changed."
                    )
                elif kind == "growth":
                    if self._growth_job_version == version:
                        self.growth_job_running = False
                        self._growth_job_version = None
                continue
            if error is not None:
                if kind == "experiment":
                    self.experiment_job_running = False
                    self.experiment_run_button.configure(state=tk.NORMAL)
                    self._experiment_cancel_event = None
                    self.experiment_cancel_button.configure(state=tk.DISABLED)
                    if isinstance(error, (RecordingCancelled, GrowthAnalysisCancelled)):
                        self.experiment_progress.configure(value=0)
                        self.experiment_status_label.configure(
                            text="Recording canceled; incomplete output was removed."
                        )
                    else:
                        self.experiment_progress.configure(value=0)
                        self.experiment_status_label.configure(
                            text=f"Experiment failed; previous results preserved: {error}"
                        )
                    continue
                if kind == "growth":
                    self.growth_job_running = False
                    self._growth_job_version = None
                    self._update_growth_track_button()
                    self.status_label.configure(
                        text=f"Growth analysis failed; previous history preserved: {error}"
                    )
                    continue
                if kind == "load":
                    self.status_label.configure(text=f"Could not open image: {error}")
                    self.threshold_scale.configure(state=tk.NORMAL)
                    if self.binary_mask is not None:
                        self.find_button.configure(state=tk.NORMAL)
                        self.analyze_button.configure(state=tk.NORMAL)
                    if self.rgb_image is not None:
                        self.roi_button.configure(state=tk.NORMAL)
                        self.clear_roi_button.configure(
                            state=tk.NORMAL if self.roi is not None else tk.DISABLED
                        )
                        self.template_start_button.configure(
                            state=tk.NORMAL
                            if self.binary_mask is not None and self.roi is not None
                            else tk.DISABLED
                        )
                        self.merge_seed_button.configure(
                            state=tk.NORMAL
                            if self.roi is not None and self.section_labels is not None
                            else tk.DISABLED
                        )
                        self.merge_track_button.configure(
                            state=tk.NORMAL if len(self.merge_seeds) == 2 else tk.DISABLED
                        )
                    self._update_growth_track_button()
                    messagebox.showerror(
                        "Could not open image", str(error), parent=self.root
                    )
                    continue
                self.status_label.configure(text=f"Processing failed: {error}")
                if kind == "mask":
                    self.find_button.configure(state=tk.DISABLED)
                    self.analyze_button.configure(state=tk.DISABLED)
                elif kind == "sections":
                    self.find_button.configure(state=tk.NORMAL)
                elif kind == "thresholds":
                    self.analyze_button.configure(state=tk.NORMAL)
                elif kind == "merge":
                    self.merge_track_button.configure(state=tk.NORMAL)
                    self.merge_seed_button.configure(state=tk.NORMAL)
                continue

            if kind == "load":
                self._install_image(*result)
            elif kind == "mask":
                self.binary_mask = result
                self.find_button.configure(state=tk.NORMAL)
                self.analyze_button.configure(state=tk.NORMAL)
                self.template_start_button.configure(
                    state=tk.NORMAL if self.roi is not None else tk.DISABLED
                )
                self.status_label.configure(text=f"Threshold mask ready (≤ {self.threshold.get()}).")
                if (
                    self.pending_template_fit is not None
                    and self.pending_template_fit.threshold == self.threshold.get()
                    and self.roi is not None
                ):
                    self._apply_pending_template_fit()
                self._render_analysis()
            elif kind == "sections":
                self.sections, self.section_labels, self.section_image = result
                self.selected_section = None
                self._populate_sections()
                self.view_mode.set("Connected Sections")
                self.status_label.configure(
                    text=(
                        f"Found {len(self.sections)} connected sections "
                        f"{'inside the ROI' if self.roi is not None else 'in the full image'}."
                    )
                )
                self.find_button.configure(state=tk.NORMAL)
                self.merge_seed_button.configure(
                    state=tk.NORMAL if self.roi is not None else tk.DISABLED
                )
                self._update_growth_track_button()
                self._render_analysis()
            elif kind == "thresholds":
                self._populate_threshold_results(result)
                self.status_label.configure(text="Threshold analysis complete.")
                self.analyze_button.configure(state=tk.NORMAL)
            elif kind == "merge":
                (
                    self.merge_result,
                    self.merge_overlay,
                    self.merge_mask,
                    self.merge_overlay_categories,
                ) = result
                self.merge_info_label.configure(
                    text=self._format_merge_result(
                        self.merge_result, self.merge_overlay_categories
                    )
                )
                self.status_label.configure(text="Merge tracking complete.")
                self.merge_track_button.configure(state=tk.NORMAL)
                self.merge_seed_button.configure(state=tk.NORMAL)
                self.view_mode.set("Merge Overlay")
                self._render_analysis()
            elif kind == "experiment":
                self.experiment_job_running = False
                self.experiment_run_button.configure(state=tk.NORMAL)
                self._experiment_cancel_event = None
                self.experiment_cancel_button.configure(state=tk.DISABLED)
                if isinstance(result, LogicalGroupingExperimentResult):
                    self._install_logical_grouping_result(result)
                elif isinstance(result, AutomaticGrowthRecordingResult):
                    self._install_threshold_growth_result(result.experiment)
                    self.experiment_status_label.configure(
                        text=(
                            f"Recording complete: {result.output_directory} "
                            f"({len(result.snapshot_files)} snapshots)."
                        )
                    )
                elif isinstance(result, ThresholdGrowthExperimentResult):
                    self._install_threshold_growth_result(result)
                else:
                    self._install_experiment_result(result)
            elif kind == "growth":
                self._install_growth_history(result)
        self.root.after(100, self._poll_worker_results)

    @staticmethod
    def _format_merge_result(
        result: MergeAnalysis, categories: MergeOverlayCategories
    ) -> str:
        sections = (
            f"Starting section sizes: {result.seed_section_sizes[0]} and "
            f"{result.seed_section_sizes[1]} pixels."
        )
        category_report = (
            f"New at merge threshold only: {len(categories.new_at_threshold_only)}.\n"
            f"New on path only: {len(categories.new_on_path_only)}.\n"
            f"Both categories: {len(categories.both)}.\n"
            f"Total newly admitted at merge threshold: "
            f"{categories.newly_selected_total}.\n"
            f"Total newly admitted on representative path: "
            f"{categories.path_pixels_total}."
        )
        if result.merge_threshold is None:
            return (
                f"{sections}\nNo connection found through threshold "
                f"{result.max_threshold} within the ROI.\nSelected at limit: "
                f"{result.selected_pixel_count}/{result.roi_pixel_count} ROI pixels.\n"
                f"{category_report}"
            )
        return (
            f"{sections}\nFirst merge threshold: {result.merge_threshold}.\n"
            f"{category_report}\n"
            f"Merged component: {result.merged_component_size} pixels; "
            f"other starting sections absorbed: {result.absorbed_start_sections}.\n"
            f"ROI selected at merge: {result.selected_pixel_count}/"
            f"{result.roi_pixel_count} pixels "
            f"({100 * result.selected_pixel_count / result.roi_pixel_count:.1f}%)."
        )

    def _populate_sections(self) -> None:
        self._clear_tree(self.section_tree)
        for section in self.sections:
            bounds = ", ".join(str(value) for value in section.bounding_rectangle)
            seed_x, seed_y = self._section_seed(section)
            self.section_tree.insert(
                "", tk.END, iid=str(section.number),
                values=(section.pixel_count, bounds),
                text=f"Section {section.number} @ ({seed_x},{seed_y})",
            )
        self.section_tree.configure(show="tree headings")
        self.section_tree.heading("#0", text="Section")

    def _on_section_selected(self, _event: tk.Event) -> None:
        selected = self.section_tree.selection()
        self.logical_add_piece_button.configure(
            state=tk.NORMAL
            if selected and self.roi is not None
            else tk.DISABLED
        )
        if not selected:
            self.selected_section = None
            self._update_growth_track_button()
            self._render_analysis()
            return
        number = int(selected[0])
        section = next(
            section for section in self.sections if section.number == number
        )
        self.selected_section = section
        self._update_growth_track_button()
        self.status_label.configure(
            text=(
                f"Section {section.number}: {section.pixel_count} pixels; "
                f"x={section.min_x}..{section.max_x}, y={section.min_y}..{section.max_y}; "
                f"box {section.bounding_rectangle}."
            )
        )
        self._render_analysis()

    def _add_logical_piece(self) -> None:
        selected_ids = self.section_tree.selection()
        if not selected_ids:
            self.status_label.configure(text="Select one or more connected sections first.")
            return
        identifier = self.logical_piece_name.get().strip()
        if not identifier or identifier in self.logical_pieces:
            self.status_label.configure(
                text="Enter a non-empty, unique piece ID before combining sections."
            )
            return
        sections = tuple(
            section for section in self.sections
            if str(section.number) in selected_ids
        )
        try:
            piece = combine_sections(identifier, sections)
        except ValueError as error:
            self.status_label.configure(text=f"Could not create piece: {error}")
            return
        self.logical_pieces[identifier] = piece
        self.logical_intended_candidate = None
        self.logical_intended_group = None
        self.logical_intended_groups = None
        self._refresh_logical_grouping()
        self.logical_piece_name.set(f"piece_{len(self.logical_pieces) + 1}")
        self.status_label.configure(
            text=f"Created logical piece '{identifier}' from {len(sections)} section(s)."
        )

    def _create_logical_group(self) -> None:
        selected = self.logical_piece_tree.selection()
        if not selected:
            self.status_label.configure(text="Select one or more logical pieces first.")
            return
        assigned = {
            piece_id
            for _group_id, piece_ids in self.logical_groups
            for piece_id in piece_ids
        }
        if assigned.intersection(selected):
            self.status_label.configure(
                text="A piece is already in a group. Remove or change that group first."
            )
            return
        group_number = 1
        existing_group_ids = {group_id for group_id, _pieces in self.logical_groups}
        while f"group_{group_number}" in existing_group_ids:
            group_number += 1
        group_id = f"group_{group_number}"
        self.logical_groups.append((group_id, tuple(selected)))
        self.logical_groups_are_intended = False
        self._refresh_logical_grouping()
        self.logical_group_tree.selection_set(group_id)
        self._on_logical_group_selected(None)
        self.status_label.configure(
            text=f"Created grouping hypothesis '{group_id}'."
        )

    def _remove_logical_group(self) -> None:
        selected = self.logical_group_tree.selection()
        if not selected:
            return
        group_id = selected[0]
        self.logical_groups = [
            group for group in self.logical_groups if group[0] != group_id
        ]
        self.logical_groups_are_intended = False
        if self.logical_intended_group == group_id:
            self.logical_intended_group = None
            self.logical_intended_groups = None
        self._refresh_logical_grouping()

    def _mark_logical_group_intended(self) -> None:
        selected = self.logical_group_tree.selection()
        if not selected:
            return
        self.logical_intended_group = selected[0]
        self.logical_intended_candidate = None
        self.logical_groups_are_intended = False
        assigned = set()
        groups = []
        for _group_id, piece_ids in self.logical_groups:
            groups.append(piece_ids)
            assigned.update(piece_ids)
        groups.extend(
            (identifier,) for identifier in self.logical_pieces
            if identifier not in assigned
        )
        self.logical_intended_groups = tuple(groups)
        self._refresh_logical_grouping()
        self.logical_group_tree.selection_set(selected[0])
        self.status_label.configure(
            text=f"Intended grouping recorded: {format_grouping_description(self.logical_intended_groups)}"
        )

    def _mark_logical_candidate_intended(self) -> None:
        selected = self.logical_candidate_tree.selection()
        if not selected:
            return
        candidate_index = int(selected[0])
        if self.logical_analysis is None:
            return
        try:
            intended = mark_candidate_intended(
                self.logical_analysis, candidate_index
            )
        except ValueError as error:
            self.status_label.configure(text=str(error))
            return
        self.logical_intended_candidate = intended.alternative_index
        self.logical_intended_groups = intended.groups
        self.logical_intended_group = None
        self.logical_groups_are_intended = False
        self._refresh_logical_grouping()
        self.logical_candidate_tree.selection_set(selected[0])
        self._on_logical_candidate_selected(None)
        self.status_label.configure(
            text=f"Intended grouping recorded: {intended.description}"
        )

    def _on_logical_group_selected(self, _event: tk.Event | None) -> None:
        selected = self.logical_group_tree.selection()
        self.logical_remove_group_button.configure(
            state=tk.NORMAL if selected else tk.DISABLED
        )
        self.logical_mark_intended_button.configure(
            state=tk.NORMAL if selected else tk.DISABLED
        )
        if selected:
            self.view_mode.set("Logical Grouping")
            self._render_analysis()

    def _on_logical_candidate_selected(self, _event: tk.Event | None) -> None:
        selected = self.logical_candidate_tree.selection()
        if not selected:
            return
        self.logical_candidate_index = int(selected[0])
        self.logical_mark_candidate_button.configure(state=tk.NORMAL)
        self.view_mode.set("Logical Grouping")
        self._render_analysis()

    def _refresh_logical_grouping(self) -> None:
        self._clear_tree(self.logical_piece_tree)
        for identifier, piece in self.logical_pieces.items():
            seeds = [self._section_seed(section) for section in piece.sections]
            section_seeds = ", ".join(f"({x},{y})" for x, y in seeds)
            self.logical_piece_tree.insert(
                "", tk.END, iid=identifier,
                values=(section_seeds, piece.bounding_rectangle),
                text=identifier,
            )
        self.logical_add_piece_button.configure(
            state=(
                tk.NORMAL
                if self.section_tree.selection() and self.roi is not None
                else tk.DISABLED
            )
        )
        self.logical_create_group_button.configure(
            state=tk.NORMAL if self.logical_pieces else tk.DISABLED
        )
        self._clear_tree(self.logical_group_tree)
        for group_id, piece_ids in self.logical_groups:
            annotation = "Intended" if group_id == self.logical_intended_group else ""
            self.logical_group_tree.insert(
                "", tk.END, iid=group_id, text=group_id,
                values=(", ".join(piece_ids), annotation),
            )

        pieces = tuple(self.logical_pieces.values())
        self.logical_analysis = analyze_grouping(
            pieces, candidate_limit=MAX_GROUPING_CANDIDATES
        ) if pieces else None
        self._clear_tree(self.logical_candidate_tree)
        self.logical_candidate_tree_ids = []
        if self.logical_analysis is not None:
            for index, candidate in enumerate(self.logical_analysis.candidates):
                iid = str(index)
                self.logical_candidate_tree_ids.append(iid)
                pair = "all separate" if not candidate.proposed_pairs else " / ".join(
                    f"{first} + {second}"
                    for first, second in candidate.proposed_pairs
                )
                score = (
                    "—" if candidate.heuristic_score is None
                    else f"{candidate.heuristic_score:.2f}"
                )
                annotation = " — Intended" if index == self.logical_intended_candidate else ""
                self.logical_candidate_tree.insert(
                    "", tk.END, iid=iid, text=f"Alternative {index + 1}{annotation}",
                    values=(pair, score),
                )
        self.logical_mark_candidate_button.configure(
            state=(
                tk.NORMAL
                if self.logical_analysis is not None and self.logical_analysis.candidates
                else tk.DISABLED
            )
        )
        self.logical_intended_label.configure(
            text=(
                "No intended grouping recorded."
                if self.logical_intended_groups is None
                else f"Intended grouping recorded: "
                     f"{format_grouping_description(self.logical_intended_groups)}"
            )
        )
        self.logical_candidate_index = min(
            self.logical_candidate_index,
            max(0, len(self.logical_analysis.candidates) - 1)
            if self.logical_analysis is not None else 0,
        )
        self._update_logical_measurement_report()
        if self.view_mode.get() == "Logical Grouping":
            self._render_analysis()

    def _section_seed(self, section: ConnectedSection) -> tuple[int, int]:
        if self.section_labels is None or self.rgb_image is None:
            raise ValueError("Find connected sections before using section seeds.")
        image_width = self.rgb_image.width
        for x in range(section.min_x, section.max_x + 1):
            if self.section_labels[section.min_y * image_width + x] == section.number:
                return x, section.min_y
        raise ValueError(f"Could not find a seed pixel for section {section.number}.")

    def _update_logical_measurement_report(self) -> None:
        analysis = self.logical_analysis
        if analysis is None:
            self.logical_measurements_label.configure(
                text="Combine sections into pieces to calculate measurements."
            )
            return
        report = [
            f"Typical piece height: {analysis.typical_piece_height:.1f}px."
        ]
        for measure in analysis.pair_measurements:
            report.append(
                f"{measure.first_piece} ↔ {measure.second_piece}: "
                f"horizontal gap {measure.horizontal_gap}px "
                f"({measure.normalized_horizontal_gap:.2f}× height); "
                f"vertical gap {measure.vertical_gap}px "
                f"({measure.normalized_vertical_gap:.2f}×); "
                f"vertical overlap {measure.vertical_overlap}px; "
                f"baseline difference {measure.baseline_difference}px "
                f"({measure.normalized_baseline_difference:.2f}×). "
                f"Bounds {measure.first_width}×{measure.first_height} and "
                f"{measure.second_width}×{measure.second_height}; relative width/height "
                f"ratios {measure.width_ratio:.2f}/{measure.height_ratio:.2f}."
            )
        paired_groups = tuple(
            piece_ids for _group_id, piece_ids in self.logical_groups
            if len(piece_ids) == 2
        )
        report.extend(
            "Pair consistency comparison: " + comparison
            for comparison in compare_grouping_consistency(
                analysis.pair_measurements, paired_groups
            )
        )
        if len(analysis.pieces) == 1:
            report.append("One-piece groups are retained as valid hypotheses.")
        if self.logical_intended_groups is not None:
            report.append(
                f"Recorded intended grouping: "
                f"{format_grouping_description(self.logical_intended_groups)}"
            )
        report.append("Geometry scores order alternatives; they do not establish correctness.")
        self.logical_measurements_label.configure(text="\n".join(report))

    def _logical_display_groups(self) -> tuple[tuple[str, ...], ...]:
        selected = self.logical_candidate_tree.selection()
        if selected and self.logical_analysis:
            candidate_index = int(selected[0])
            return self.logical_analysis.candidates[candidate_index].groups
        if self.logical_intended_groups is not None:
            return self.logical_intended_groups
        if self.logical_groups:
            assigned = set()
            display_groups = []
            for _group_id, piece_ids in self.logical_groups:
                display_groups.append(piece_ids)
                assigned.update(piece_ids)
            display_groups.extend(
                (identifier,) for identifier in self.logical_pieces
                if identifier not in assigned
            )
            return tuple(display_groups)
        if self.logical_analysis and self.logical_analysis.candidates:
            return self.logical_analysis.candidates[
                self.logical_candidate_index
            ].groups
        return ()

    def _logical_group_bounds(
        self, group: tuple[str, ...]
    ) -> tuple[int, int, int, int] | None:
        pieces = [self.logical_pieces[name] for name in group if name in self.logical_pieces]
        if not pieces:
            return None
        bounds = [piece.bounding_rectangle for piece in pieces]
        left = min(box[0] for box in bounds)
        top = min(box[1] for box in bounds)
        right = max(box[0] + box[2] for box in bounds)
        bottom = max(box[1] + box[3] for box in bounds)
        return left, top, right - left, bottom - top

    def _populate_threshold_results(self, results: list[tuple[int, int, int, int]]) -> None:
        self._clear_tree(self.threshold_tree)
        for threshold, selected, section_count, largest in results:
            self.threshold_tree.insert(
                "", tk.END,
                values=(f"{threshold} / {selected}", section_count, largest),
            )

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        tree.delete(*tree.get_children())

    def _on_close(self) -> None:
        self._closed = True
        if self._experiment_cancel_event is not None:
            self._experiment_cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PixelMathRecognitionLab(root)
    root.mainloop()


if __name__ == "__main__":
    main()
