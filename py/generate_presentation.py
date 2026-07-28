"""
Generates a comprehensive, professional PPTX presentation for the
AdaptiveStabilization Final Progress Report, embedding all data tables,
findings, and plot images from the project.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Colour Palette ──────────────────────────────────────────────────
DARK      = RGBColor(0x0F, 0x17, 0x2A)   # titles
CHARCOAL  = RGBColor(0x1E, 0x29, 0x3B)   # body text
SLATE     = RGBColor(0x47, 0x55, 0x69)   # subtitles
MUTED     = RGBColor(0x64, 0x74, 0x8B)   # captions
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT    = RGBColor(0x38, 0x6B, 0xF6)   # blue highlights
RED       = RGBColor(0xDC, 0x26, 0x26)   # warnings
GREEN     = RGBColor(0x16, 0xA3, 0x4A)   # success
BG_LIGHT  = RGBColor(0xF8, 0xFA, 0xFC)   # light slide bg
TBL_HDR   = RGBColor(0x1E, 0x40, 0xAF)   # table header blue
TBL_ALT   = RGBColor(0xEF, 0xF6, 0xFF)   # table alt row

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


# ── Helper functions ────────────────────────────────────────────────
def _set_slide_bg(slide, color):
    """Set a solid background colour for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_textbox(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=CHARCOAL, alignment=PP_ALIGN.LEFT,
                 font_name="Calibri"):
    """Add a simple textbox and return the paragraph for chaining."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def _add_bullet_frame(slide, left, top, width, height, items,
                      font_size=16, color=CHARCOAL, bold_first=False):
    """Add a text frame with bullet points."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = Pt(6)
        if bold_first and i == 0:
            p.font.bold = True
    return tf


def _add_image(slide, filename, left, top, width, height=None):
    """Embed an image from plots/ if it exists, scaled to given width."""
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        if height:
            slide.shapes.add_picture(path, Inches(left), Inches(top),
                                     Inches(width), Inches(height))
        else:
            slide.shapes.add_picture(path, Inches(left), Inches(top),
                                     Inches(width))
        return True
    return False


def _add_table(slide, rows, cols, data, left, top, width, height,
               col_widths=None):
    """Add a styled table to the slide."""
    table_shape = slide.shapes.add_table(rows, cols,
                                         Inches(left), Inches(top),
                                         Inches(width), Inches(height))
    table = table_shape.table

    # Set column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(11)
                paragraph.font.name = "Calibri"
                paragraph.alignment = PP_ALIGN.CENTER
                if r == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = WHITE
                else:
                    paragraph.font.color.rgb = CHARCOAL

            # Header row
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TBL_HDR
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TBL_ALT

    return table


def _add_section_number(slide, number):
    """Add a subtle section number badge in the top-left corner."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.4), Inches(0.3), Inches(0.5), Inches(0.4))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.text = str(number)
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER


# ── Build Presentation ─────────────────────────────────────────────
def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    BLANK = prs.slide_layouts[6]  # blank layout

    # ================================================================
    # SLIDE 1 — Title
    # ================================================================
    s1 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s1, DARK)
    _add_textbox(s1, 1.5, 1.5, 10, 1.5,
                 "Adaptive Stabilization Techniques\nfor Ill-Conditioned Linear Regression",
                 font_size=36, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
    _add_textbox(s1, 1.5, 3.3, 10, 0.6,
                 "Final Project Progress Report  •  AIMA 203 – Numerical Methods",
                 font_size=18, color=MUTED, alignment=PP_ALIGN.CENTER)
    _add_textbox(s1, 1.5, 4.3, 10, 0.5,
                 "Utsab Baral  •  Aman Sapkota  •  Kushagra Poudel",
                 font_size=16, color=SLATE, alignment=PP_ALIGN.CENTER)
    _add_textbox(s1, 1.5, 5.0, 10, 0.5,
                 "Kathmandu University, Nepal  |  May 2026",
                 font_size=14, color=MUTED, alignment=PP_ALIGN.CENTER)

    # ================================================================
    # SLIDE 2 — Research Question & Motivation
    # ================================================================
    s2 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s2, BG_LIGHT)
    _add_section_number(s2, 1)
    _add_textbox(s2, 1.0, 0.3, 11, 0.6,
                 "Research Question & Motivation", font_size=28, bold=True, color=DARK)
    _add_textbox(s2, 1.0, 1.1, 11, 0.7,
                 "\"Does minimizing training loss guarantee recovering the true model parameters under ill-conditioning?\"",
                 font_size=20, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)
    _add_bullet_frame(s2, 1.0, 2.0, 5.5, 4.5, [
        "• Real-world datasets contain redundant / collinear features",
        "• This creates an ill-conditioned data matrix X",
        "• Condition number κ = λ_max / λ_min of the Hessian H = XᵀX",
        "• High κ → steep ravines in some directions, infinitely flat valleys in others",
        "• Same learning rate must serve both — an inherent conflict",
        "• Central Thesis: Modern optimizers can achieve near-perfect loss (~10⁻⁶)",
        "  while parameters diverge catastrophically — the \"Optimization Mirage\""
    ], font_size=15)
    # Illustration box for the concept
    _add_textbox(s2, 7.0, 2.2, 5.5, 4.0,
                 "The Condition Number κ\n\n"
                 "κ = 10      →  Nearly circular loss bowl (easy)\n"
                 "κ = 10⁴    →  Elongated valley (tricky)\n"
                 "κ = 10⁷    →  Ultra-flat ravine (catastrophic)\n\n"
                 "At κ = 10⁷, moving thousands of units\n"
                 "along the flat direction changes the loss\n"
                 "by less than floating-point precision.",
                 font_size=14, color=CHARCOAL)

    # ================================================================
    # SLIDE 3 — Code Architecture
    # ================================================================
    s3 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s3, BG_LIGHT)
    _add_section_number(s3, 2)
    _add_textbox(s3, 1.0, 0.3, 11, 0.6,
                 "Code Architecture & Pipeline", font_size=28, bold=True, color=DARK)

    arch_data = [
        ["File", "Lines", "Purpose"],
        ["config.py", "82", "All hyperparameters, dataset specs, LR formulas"],
        ["data_generator.py", "78", "SVD-based synthetic data with exact κ control"],
        ["models.py", "28", "LinearModel and LinearWithBatchNorm"],
        ["optimizers.py", "157", "6 training loops with metric tracking"],
        ["train.py", "126", "Main orchestrator: data → train → save"],
        ["plot_results.py", "148", "4 core publication-quality plots"],
        ["plot_advanced.py", "245", "Multi-level drill-down plots"],
    ]
    _add_table(s3, 8, 3, arch_data, 0.8, 1.2, 6.5, 3.5,
               col_widths=[1.8, 0.8, 3.9])

    _add_bullet_frame(s3, 7.8, 1.2, 5.0, 5.0, [
        "Pipeline Flow:",
        "  config.py  →  data_generator.py  →  train.py",
        "  train.py  →  results/results.pkl (77.6 MB)",
        "  results.pkl  →  plot_results.py  →  plots/",
        "",
        "Total: 90 training runs",
        "  (3 datasets × 6 methods × 5 seeds)",
        "",
        "Key Design: Linear Regression chosen to isolate",
        "optimizer dynamics from neural-net complexity",
        "(no local minima, constant Hessian, quadratic loss)."
    ], font_size=14)

    # ================================================================
    # SLIDE 4 — Experimental Design: Datasets
    # ================================================================
    s4 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s4, BG_LIGHT)
    _add_section_number(s4, 3)
    _add_textbox(s4, 1.0, 0.3, 11, 0.6,
                 "Experimental Design — Datasets", font_size=28, bold=True, color=DARK)

    ds_data = [
        ["Dataset", "κ", "Epochs", "Description"],
        ["Well-Conditioned", "10", "1,005", "Isotropic loss bowl – all methods should converge"],
        ["Moderate", "10⁴", "6,000", "Elongated valley – methods begin to diverge"],
        ["Severe", "10⁷", "50,000", "Ultra-flat valley – tests numerical limits"],
    ]
    _add_table(s4, 4, 4, ds_data, 0.8, 1.2, 11.5, 2.0,
               col_widths=[2.2, 1.0, 1.2, 7.1])

    _add_bullet_frame(s4, 0.8, 3.5, 5.5, 3.5, [
        "Data Generation (SVD-Constructed):",
        "  • X = U Σ Vᵀ with exact κ control",
        "  • n = 1,000 train, 200 test samples",
        "  • d = 50 features",
        "  • σ log-spaced from σ_max=1.0 to σ_min = σ_max/√κ",
        "  • Noise: ε ~ N(0, 10⁻³)",
        "  • Precision: float64 (avoids float32 breakdown)"
    ], font_size=14)

    _add_bullet_frame(s4, 7.0, 3.5, 5.5, 3.5, [
        "Metrics Tracked Per Epoch:",
        "  • Train Loss — ½‖Xθ − y‖²/n",
        "  • Test Loss — same on held-out data",
        "  • Parameter Error — ‖θ − θ*‖₂",
        "  • Gradient Norm — ‖∇L‖₂",
        "  • Eigen-Direction Errors (top & bottom eigenvectors)",
        "  • Wall clock time, CPU %, RAM MB",
        "  • Divergence detection (NaN/Inf)"
    ], font_size=14)

    # ================================================================
    # SLIDE 5 — Experimental Design: 6 Optimizers
    # ================================================================
    s5 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s5, BG_LIGHT)
    _add_section_number(s5, 4)
    _add_textbox(s5, 1.0, 0.3, 11, 0.6,
                 "Experimental Design — 6 Optimization Methods", font_size=28, bold=True, color=DARK)

    opt_data = [
        ["Method", "Key Mechanism", "Learning Rate", "Special Parameters"],
        ["Vanilla SGD", "Raw gradient descent", "1800", "None"],
        ["Momentum SGD", "Velocity accumulation", "1900", "β* = ((√κ−1)/(√κ+1))²"],
        ["Adam", "Per-coord adaptive scaling", "0.1", "β₁=0.9, β₂=0.999, ε=10⁻⁸"],
        ["Ridge L2", "SGD + L2 penalty", "1800", "λ from κ (reduce eff. conditioning)"],
        ["BN-on-Linear", "BatchNorm before linear", "0.1 (Adam)", "affine=False"],
        ["AdamW", "Adam + decoupled decay", "0.1", "weight_decay = Ridge λ"],
    ]
    _add_table(s5, 7, 4, opt_data, 0.8, 1.1, 11.5, 3.5,
               col_widths=[2.0, 3.0, 1.5, 5.0])

    _add_bullet_frame(s5, 0.8, 5.0, 11.5, 2.0, [
        "Statistical Design:  5 random initialization seeds per method-dataset pair  →  90 total training runs",
        "Results include mean ± std bands across seeds  •  Bias² and Variance computed from final θ across seeds"
    ], font_size=14)

    # ================================================================
    # SLIDE 6 — Numerical Stability Baselines
    # ================================================================
    s6 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s6, BG_LIGHT)
    _add_section_number(s6, 5)
    _add_textbox(s6, 1.0, 0.3, 11, 0.6,
                 "Result 1 — Numerical Stability Baselines", font_size=28, bold=True, color=DARK)

    ns_data = [
        ["Dataset", "κ", "Naive Solve Error", "Stable (Pseudoinverse)"],
        ["Well-Conditioned", "10", "1.273 × 10⁻²", "1.273 × 10⁻²"],
        ["Moderate", "10⁴", "2.793 × 10⁻¹", "2.793 × 10⁻¹"],
        ["Severe", "10⁷", "5.764", "5.764"],
    ]
    _add_table(s6, 4, 4, ns_data, 0.6, 1.1, 5.5, 2.0,
               col_widths=[1.8, 0.6, 1.5, 1.6])

    _add_image(s6, "2_numerical_stability.png", 6.5, 1.0, 6.2, 4.5)

    _add_bullet_frame(s6, 0.6, 3.5, 5.5, 3.5, [
        "Key Insight:",
        "  • Both solvers produce identical errors (float64 precision)",
        "  • Error growth is NOT numerical breakdown",
        "  • It is inherent noise amplification:",
        "    ‖δθ‖/‖θ‖ ≈ κ × ‖δb‖/‖b‖",
        "  • noise_std=10⁻³ × κ=10⁷ → relative error ≈ O(1–10)",
        "  • Confirms the theoretical perturbation bound"
    ], font_size=14)

    # ================================================================
    # SLIDE 7 — Convergence Results (Main Plot)
    # ================================================================
    s7 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s7, BG_LIGHT)
    _add_section_number(s7, 6)
    _add_textbox(s7, 1.0, 0.3, 11, 0.6,
                 "Result 2 — Global Convergence Trajectories", font_size=28, bold=True, color=DARK)

    _add_image(s7, "1_convergence.png", 0.5, 1.0, 12.3, 6.0)

    # ================================================================
    # SLIDE 8 — The Optimization Mirage (Key Finding)
    # ================================================================
    s8 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s8, DARK)
    _add_textbox(s8, 1.0, 0.4, 11, 0.6,
                 "⚠️  THE OPTIMIZATION MIRAGE", font_size=32, bold=True, color=RED,
                 alignment=PP_ALIGN.CENTER)
    _add_textbox(s8, 1.0, 1.2, 11, 0.6,
                 "Perfect loss does NOT mean correct parameters.",
                 font_size=22, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    mirage_data = [
        ["Method", "Train Loss (κ=10⁷)", "Param Error (κ=10⁷)", "Verdict"],
        ["Vanilla SGD", "4.93 × 10⁻⁷", "0.701 (still falling)", "✅ Robust"],
        ["Momentum SGD", "4.89 × 10⁻⁷", "5.764", "❌ Catastrophic"],
        ["Adam", "4.97 × 10⁻⁷", "5.764", "❌ Catastrophic"],
        ["AdamW", "5.06 × 10⁻⁷", "5.621", "❌ Catastrophic"],
        ["BN-on-Linear", "2.58 × 10⁻⁵", "0.991", "⚠️ Broken by design"],
        ["Ridge L2", "∞ (DIVERGED)", "∞", "💥 LR bug"],
    ]
    _add_table(s8, 7, 4, mirage_data, 1.0, 2.2, 11.3, 3.5,
               col_widths=[2.2, 2.5, 3.5, 3.1])

    _add_textbox(s8, 1.0, 6.0, 11.3, 1.0,
                 "At κ=10⁷, Adam/Momentum/AdamW losses drop to ~5×10⁻⁷ (near-perfect!) "
                 "but parameters are 5.76× further from θ* than random initialization.",
                 font_size=15, color=MUTED, alignment=PP_ALIGN.CENTER)

    # ================================================================
    # SLIDE 9 — Per-Dataset Results: Well-Conditioned (κ=10)
    # ================================================================
    s9 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s9, BG_LIGHT)
    _add_section_number(s9, 7)
    _add_textbox(s9, 1.0, 0.3, 11, 0.6,
                 "Per-Dataset Results — κ = 10  (Well-Conditioned)", font_size=28, bold=True, color=DARK)

    kappa10_data = [
        ["Method", "Train Loss", "Test Loss", "Param Error", "Diverged", "Time"],
        ["Vanilla SGD", "4.507×10⁻⁷", "5.019×10⁻⁷", "0.01273", "0/5", "0.6s"],
        ["Momentum SGD", "4.507×10⁻⁷", "5.019×10⁻⁷", "0.01273", "0/5", "0.2s"],
        ["Adam", "4.507×10⁻⁷", "5.019×10⁻⁷", "0.01273", "0/5", "0.4s"],
        ["Ridge L2", "4.524×10⁻⁷", "4.943×10⁻⁷", "0.01393", "0/5", "0.2s"],
        ["BN-on-Linear", "4.933×10⁻⁷", "5.562×10⁻⁷", "0.979 ⚠️", "0/5", "0.5s"],
        ["AdamW", "4.507×10⁻⁷", "5.019×10⁻⁷", "0.01273", "0/5", "0.3s"],
    ]
    _add_table(s9, 7, 6, kappa10_data, 0.5, 1.2, 12.3, 3.5,
               col_widths=[2.0, 2.0, 2.0, 2.0, 1.3, 1.0])

    _add_bullet_frame(s9, 0.5, 5.0, 12.3, 2.0, [
        "Insight:  All methods converge to noise-floor loss (~10⁻⁷).  Param error ≈ 0.013 for all EXCEPT BN-on-Linear (stuck at 0.979).",
        "BatchNorm's first failure signal — even in the easiest scenario, it cannot recover parameter magnitude due to scale invariance."
    ], font_size=15)

    # ================================================================
    # SLIDE 10 — Per-Dataset Results: Moderate (κ=10⁴)
    # ================================================================
    s10 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s10, BG_LIGHT)
    _add_section_number(s10, 8)
    _add_textbox(s10, 1.0, 0.3, 11, 0.6,
                 "Per-Dataset Results — κ = 10⁴  (Moderate)", font_size=28, bold=True, color=DARK)

    kappa4_data = [
        ["Method", "Train Loss", "Test Loss", "Param Error", "Diverged", "Time"],
        ["Vanilla SGD", "4.881×10⁻⁷", "6.673×10⁻⁷", "0.217 ± 0.016", "0/5", "1.3s"],
        ["Momentum SGD", "4.875×10⁻⁷", "6.810×10⁻⁷", "0.279", "0/5", "1.3s"],
        ["Adam", "5.014×10⁻⁷", "7.594×10⁻⁷", "0.280", "0/5", "1.8s"],
        ["Ridge L2", "∞ (DIVERGED)", "∞", "∞", "5/5 ❌", "0.0s"],
        ["BN-on-Linear", "8.406×10⁻⁵", "4.179×10⁻⁴", "0.991", "0/5", "2.9s"],
        ["AdamW", "5.155×10⁻⁷", "8.441×10⁻⁷", "0.280", "0/5", "1.8s"],
    ]
    _add_table(s10, 7, 6, kappa4_data, 0.5, 1.2, 12.3, 3.5,
               col_widths=[2.0, 2.0, 2.0, 2.3, 1.3, 1.0])

    _add_bullet_frame(s10, 0.5, 5.0, 12.3, 2.0, [
        "The Mirage Begins:  Losses are all ~10⁻⁷ (excellent!) but param errors stabilize around 0.28 (28% error).  Perfect loss ≠ correct parameters.",
        "Ridge L2 diverges (all 5 seeds) — LR=1800 + large λ penalty overshoots stability bound η < 2/L_eff.  This is a hyperparameter bug."
    ], font_size=15)

    # ================================================================
    # SLIDE 11 — Per-Dataset Results: Severe (κ=10⁷)
    # ================================================================
    s11 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s11, BG_LIGHT)
    _add_section_number(s11, 9)
    _add_textbox(s11, 1.0, 0.3, 11, 0.6,
                 "Per-Dataset Results — κ = 10⁷  (Severe)", font_size=28, bold=True, color=DARK)

    kappa7_data = [
        ["Method", "Train Loss", "Test Loss", "Param Error", "Diverged", "Time"],
        ["Vanilla SGD", "4.930×10⁻⁷", "5.657×10⁻⁷", "0.701 ± 0.059", "0/5", "10.7s"],
        ["Momentum SGD", "4.886×10⁻⁷", "5.751×10⁻⁷", "5.764 ❌", "0/5", "10.8s"],
        ["Adam", "4.965×10⁻⁷", "6.123×10⁻⁷", "5.764 ❌", "0/5", "14.9s"],
        ["Ridge L2", "∞ (DIVERGED)", "∞", "∞", "5/5 💥", "0.3s"],
        ["BN-on-Linear", "2.578×10⁻⁵", "1.256×10⁻⁴", "0.991 ⚠️", "0/5", "24.1s"],
        ["AdamW", "5.061×10⁻⁷", "6.416×10⁻⁷", "5.621 ❌", "0/5", "15.1s"],
    ]
    _add_table(s11, 7, 6, kappa7_data, 0.5, 1.2, 12.3, 3.5,
               col_widths=[2.0, 2.0, 2.0, 2.3, 1.3, 1.0])

    _add_bullet_frame(s11, 0.5, 5.0, 12.3, 2.0, [
        "Catastrophic Parameter Explosion:  Adam, Momentum, AdamW all achieve train loss ~5×10⁻⁷ while param error explodes to 5.76 — further from θ* than random init.",
        "Vanilla SGD is the most robust unregularized method — param error 0.701 and still falling at epoch 50,000.  Its raw gradient steps naturally ignore flat directions."
    ], font_size=15)

    # ================================================================
    # SLIDE 12 — Bias-Variance Decomposition
    # ================================================================
    s12 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s12, BG_LIGHT)
    _add_section_number(s12, 10)
    _add_textbox(s12, 1.0, 0.3, 11, 0.6,
                 "Result 3 — Bias-Variance Decomposition", font_size=28, bold=True, color=DARK)

    _add_image(s12, "3_bias_variance.png", 0.5, 1.0, 6.5, 4.5)

    bv_data = [
        ["Dataset / Method", "Bias²", "Variance"],
        ["κ=10  All standard", "~1.6×10⁻⁴", "~10⁻³² (negligible)"],
        ["κ=10  BN-on-Linear", "0.959", "4.3×10⁻³⁵"],
        ["κ=10⁴  Vanilla SGD", "0.046", "2.1×10⁻³"],
        ["κ=10⁴  Adam/Mom/AdamW", "~0.078", "~10⁻⁴"],
        ["κ=10⁷  Vanilla SGD", "0.417", "0.098"],
        ["κ=10⁷  Mom/Adam/AdamW", "31 – 33 ❌", "~10⁻⁴ to 10⁻¹⁰"],
    ]
    _add_table(s12, 7, 3, bv_data, 7.3, 1.0, 5.5, 3.5,
               col_widths=[2.5, 1.5, 1.5])

    _add_bullet_frame(s12, 7.3, 5.0, 5.5, 2.0, [
        "Adam/Momentum at κ=10⁷: Bias² ≈ 33 → catastrophically biased",
        "BN-on-Linear: Bias² ≈ 0.98 always → deterministically wrong",
        "Vanilla SGD: High variance (0.098) but trending toward θ*"
    ], font_size=14)

    # ================================================================
    # SLIDE 13 — Resource Usage
    # ================================================================
    s13 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s13, BG_LIGHT)
    _add_section_number(s13, 11)
    _add_textbox(s13, 1.0, 0.3, 11, 0.6,
                 "Result 4 — Computational Resource Usage", font_size=28, bold=True, color=DARK)

    _add_image(s13, "4_resource_usage.png", 0.5, 1.0, 6.5, 4.8)

    res_data = [
        ["Method", "Time (κ=10⁷)", "RAM (κ=10⁷)"],
        ["Vanilla SGD", "10.7s", "706 MB"],
        ["Momentum SGD", "10.8s", "766 MB"],
        ["Adam", "14.9s", "825 MB"],
        ["BN-on-Linear", "24.1s (slowest)", "885 MB"],
        ["AdamW", "15.1s", "942 MB (most)"],
    ]
    _add_table(s13, 6, 3, res_data, 7.5, 1.2, 5.3, 3.0,
               col_widths=[2.0, 1.5, 1.8])

    _add_bullet_frame(s13, 7.5, 4.5, 5.3, 2.5, [
        "BN-on-Linear is slowest (extra BN computation)",
        "AdamW uses the most memory (optimizer states)",
        "Time scales linearly with epochs",
        "  (epochs scale with κ, capped at 50,000)"
    ], font_size=14)

    # ================================================================
    # SLIDE 14 — Mechanistic Explanations
    # ================================================================
    s14 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s14, BG_LIGHT)
    _add_section_number(s14, 12)
    _add_textbox(s14, 1.0, 0.3, 11, 0.6,
                 "Why These Results Occur — Mechanistic Analysis", font_size=28, bold=True, color=DARK)

    _add_bullet_frame(s14, 0.5, 1.2, 6.0, 6.0, [
        "Adam / AdamW Parameter Explosion:",
        "  θ_{t+1} = θ_t − α · m_t / (√v_t + ε)",
        "  Along flat eigenvectors: true gradient ≈ 0",
        "  But stochastic noise ≠ 0 → small v_t denominator",
        "  → Noise amplified into large parameter steps",
        "  → Parameters explode along uninformative directions",
        "",
        "Momentum Drift:",
        "  v_t = β · v_{t-1} + g_t",
        "  β* ≈ 0.999 at κ=10⁷ (retains 99.9% of history)",
        "  Without restoring gradient on flat dirs,",
        "  accumulated noise → uncontrolled random walk"
    ], font_size=14)

    _add_bullet_frame(s14, 6.8, 1.2, 6.0, 6.0, [
        "Vanilla SGD's Geometric Safety:",
        "  θ_{t+1} = θ_t − η · g_t",
        "  Step is proportional to gradient magnitude",
        "  Flat direction → tiny gradient → tiny step",
        "  → Naturally ignores noise-dominated directions",
        "  → Slower but fundamentally safer",
        "",
        "BN-on-Linear's Structural Impossibility:",
        "  BN normalizes: X̂ = (X − μ) / σ",
        "  For linear model: BN(c·θᵀx) = BN(θᵀx)",
        "  Output is invariant to scale of θ",
        "  → Gradient w.r.t. radial scale is exactly zero",
        "  → Can find direction but never magnitude"
    ], font_size=14)

    # ================================================================
    # SLIDE 15 — Architectural Summary Table
    # ================================================================
    s15 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s15, BG_LIGHT)
    _add_section_number(s15, 13)
    _add_textbox(s15, 1.0, 0.3, 11, 0.6,
                 "Architectural Summary", font_size=28, bold=True, color=DARK)

    summary_data = [
        ["Optimizer", "Loss", "Param (κ ≤ 10⁴)", "Param (κ = 10⁷)", "Failure Mode"],
        ["Adam / AdamW", "~10⁻⁷ ✅", "0.28 error", "5.76 error ❌", "Inverse-variance noise amp."],
        ["Momentum SGD", "~10⁻⁷ ✅", "0.28 error", "5.76 error ❌", "Velocity drift on flat subspace"],
        ["Vanilla SGD", "~10⁻⁷ ✅", "0.22 error", "0.70 (falling) ✅", "Slow convergence (safe)"],
        ["BN-on-Linear", "~10⁻⁵ ⚠️", "0.98 error ❌", "0.99 error ❌", "Scale elimination"],
        ["Ridge L2", "DIVERGED 💥", "DIVERGED", "DIVERGED", "LR too aggressive"],
    ]
    _add_table(s15, 6, 5, summary_data, 0.5, 1.2, 12.3, 3.0,
               col_widths=[2.0, 1.8, 2.5, 3.0, 3.0])

    _add_textbox(s15, 0.5, 4.5, 12.3, 2.5,
                 "Key Takeaway: Loss monitoring alone is dangerously misleading.\n\n"
                 "• Adam/Momentum are excellent at minimizing loss but catastrophic at parameter recovery under ill-conditioning.\n"
                 "• Vanilla SGD's simplicity is its strength — gradient-proportional steps provide inherent geometric safety.\n"
                 "• Robust system identification requires geometric stabilization (Ridge/L2), not just faster optimizers.",
                 font_size=15, color=CHARCOAL)

    # ================================================================
    # SLIDE 16 — Conclusion & Thank You
    # ================================================================
    s16 = prs.slides.add_slide(BLANK)
    _set_slide_bg(s16, DARK)
    _add_textbox(s16, 1.5, 1.0, 10, 1.0,
                 "Conclusion", font_size=36, bold=True, color=WHITE,
                 alignment=PP_ALIGN.CENTER)

    _add_bullet_frame(s16, 1.5, 2.2, 10, 3.5, [
        "1.  Modern optimizers can achieve perfect loss while parameters\n"
        "     completely diverge from reality — the \"Optimization Mirage\".",
        "",
        "2.  The failure is localized to flat, noise-dominated eigen-directions\n"
        "     of the Hessian, not to the overall optimization process.",
        "",
        "3.  Vanilla SGD's gradient-proportional steps are inherently self-limiting\n"
        "     and geometrically safe, despite being the simplest method.",
        "",
        "4.  Geometric stabilization (Ridge regularization with correct LR)\n"
        "     is essential for reliable parameter recovery.",
        "",
        "5.  You cannot blindly trust a low error rate.\n"
        "     Structure-aware diagnostics are critical for real-world ML."
    ], font_size=16, color=MUTED)

    _add_textbox(s16, 1.5, 6.0, 10, 0.8,
                 "Thank You!  •  Questions?",
                 font_size=28, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)

    # ── Save ────────────────────────────────────────────────────────
    output_path = os.path.join(BASE_DIR, "AdaptiveStabilization_Presentation.pptx")
    prs.save(output_path)
    print(f"✅ Presentation saved to: {output_path}")
    print(f"   → {len(prs.slides)} slides generated")


if __name__ == "__main__":
    build_presentation()
