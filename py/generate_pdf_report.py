"""
Generate a professional PDF report for Adaptive Stabilization project.
Uses reportlab for layout and matplotlib for math formula rendering.
"""
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.pdfgen import canvas as pdfcanvas
from PIL import Image as PILImage

# ─── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = "/Users/kushagra/.gemini/antigravity-ide/brain/ff0e7933-3f4b-4fe6-acf9-20777cf54674"
OUTPUT_PDF = os.path.join(PROJECT_DIR, "AdaptiveStabilization_Final_Report.pdf")

PLOT_PATHS = {
    "convergence":  os.path.join(ARTIFACT_DIR, "1_convergence.png"),
    "stability":    os.path.join(ARTIFACT_DIR, "2_numerical_stability.png"),
    "bias_var":     os.path.join(ARTIFACT_DIR, "3_bias_variance.png"),
    "resources":    os.path.join(ARTIFACT_DIR, "4_resource_usage.png"),
}

# ─── Math Formula Renderer ─────────────────────────────────────────────────────
def render_formula(latex: str, fontsize: int = 16) -> io.BytesIO:
    """Render a LaTeX formula cleanly using matplotlib mathtext."""
    # Use a tall enough figure so ascenders/descenders are never clipped
    fig = plt.figure(figsize=(8, 1.1))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5, 0.5,
        f"${latex}$",
        fontsize=fontsize,
        ha='center', va='center',
        transform=ax.transAxes,
        usetex=False,
        fontfamily='DejaVu Sans',
        color='black',
    )
    buf = io.BytesIO()
    fig.savefig(
        buf, format='png', dpi=220,
        bbox_inches='tight', pad_inches=0.12,
        transparent=False, facecolor='white'
    )
    plt.close(fig)
    buf.seek(0)
    return buf


def formula_image(latex: str, width_cm: float = 10, fontsize: int = 16):
    buf = render_formula(latex, fontsize=fontsize)
    # Constrain height so formula images never overflow a page
    pil = PILImage.open(buf)
    orig_w, orig_h = pil.size
    aspect = orig_h / orig_w
    target_w = width_cm * cm
    target_h = target_w * aspect
    max_h = 3.5 * cm
    if target_h > max_h:
        target_h = max_h
        target_w = target_h / aspect
    buf.seek(0)
    img = Image(buf, width=target_w, height=target_h)
    img.hAlign = 'CENTER'
    return img


# ─── Numbered Canvas ───────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
HEADER_TEXT = "Adaptive Stabilization Techniques — Ill-Conditioned Linear Regression"

class NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for page_num, state in enumerate(self._saved_page_states, 1):
            self.__dict__.update(state)
            self._draw_decorations(page_num, total_pages)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _draw_decorations(self, page_num, total_pages):
        # Skip cover and abstract
        if page_num <= 2:
            return

        self.saveState()
        # Header
        self.setStrokeColor(colors.HexColor('#2C3E50'))
        self.setLineWidth(0.5)
        self.line(MARGIN, PAGE_H - 1.5 * cm, PAGE_W - MARGIN, PAGE_H - 1.5 * cm)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor('#555555'))
        self.drawString(MARGIN, PAGE_H - 1.3 * cm, HEADER_TEXT)

        # Footer
        self.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor('#555555'))
        self.drawCentredString(PAGE_W / 2, 1.0 * cm, str(page_num))
        self.restoreState()



# ─── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name=name, **kw)

    styles = {
        "title": S("Title",
            fontName="Helvetica-Bold", fontSize=22,
            textColor=colors.HexColor('#1A252F'),
            alignment=TA_CENTER, spaceAfter=8, leading=28),
        "subtitle": S("Subtitle",
            fontName="Helvetica", fontSize=14,
            textColor=colors.HexColor('#2C3E50'),
            alignment=TA_CENTER, spaceAfter=6, leading=20),
        "authors": S("Authors",
            fontName="Helvetica", fontSize=12,
            textColor=colors.HexColor('#34495E'),
            alignment=TA_CENTER, spaceAfter=4, leading=18),
        "chapter": S("Chapter",
            fontName="Helvetica-Bold", fontSize=14,
            textColor=colors.HexColor('#1A252F'),
            spaceBefore=18, spaceAfter=8,
            borderPad=4, leading=18,
            backColor=colors.HexColor('#EBF5FB'),
            leftIndent=0, rightIndent=0),
        "section": S("Section",
            fontName="Helvetica-Bold", fontSize=11.5,
            textColor=colors.HexColor('#2C3E50'),
            spaceBefore=12, spaceAfter=5, leading=16),
        "body": S("Body",
            fontName="Helvetica", fontSize=10.5,
            textColor=colors.HexColor('#212121'),
            alignment=TA_JUSTIFY, spaceAfter=6, leading=16),
        "bullet": S("Bullet",
            fontName="Helvetica", fontSize=10.5,
            textColor=colors.HexColor('#212121'),
            alignment=TA_LEFT, leftIndent=16,
            spaceAfter=4, leading=15,
            bulletIndent=0),
        "toc_item": S("TOC",
            fontName="Helvetica", fontSize=10.5,
            textColor=colors.HexColor('#2C3E50'),
            leftIndent=0, spaceAfter=3, leading=15),
        "toc_sub": S("TOCSub",
            fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor('#555555'),
            leftIndent=24, spaceAfter=2, leading=14),
        "caption": S("Caption",
            fontName="Helvetica-Oblique", fontSize=9.5,
            textColor=colors.HexColor('#555555'),
            alignment=TA_CENTER, spaceAfter=4, leading=13),
        "warning_box": S("Warning",
            fontName="Helvetica", fontSize=10.5,
            textColor=colors.HexColor('#5D3A00'),
            backColor=colors.HexColor('#FFF3CD'),
            alignment=TA_LEFT, spaceAfter=6, leading=15,
            leftIndent=12, rightIndent=12,
            borderPad=6),
        "kw_label": S("KWLabel",
            fontName="Helvetica-Bold", fontSize=10.5,
            textColor=colors.HexColor('#1A252F'),
            alignment=TA_LEFT, spaceAfter=6, leading=15),
        "abstract_body": S("AbsBody",
            fontName="Helvetica", fontSize=10.5,
            textColor=colors.HexColor('#212121'),
            alignment=TA_JUSTIFY, spaceAfter=6, leading=16),
        "numbered": S("Numbered",
            fontName="Helvetica", fontSize=10.5,
            textColor=colors.HexColor('#212121'),
            alignment=TA_JUSTIFY, leftIndent=20,
            spaceAfter=4, leading=15),
        "ref": S("Ref",
            fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor('#212121'),
            alignment=TA_LEFT, leftIndent=24, firstLineIndent=-24,
            spaceAfter=5, leading=14),
        "section_header_abstract": S("SectionHeaderAbstract",
            fontName="Helvetica-Bold", fontSize=14,
            textColor=colors.HexColor('#1A252F'),
            spaceBefore=12, spaceAfter=8,
            alignment=TA_CENTER, leading=18),
    }
    return styles


def safe_img(path: str, width_cm: float = 15, max_height_cm: float = 14):
    if not path or not os.path.exists(path):
        return None
    pil = PILImage.open(path)
    orig_w, orig_h = pil.size
    aspect = orig_h / orig_w
    target_w = width_cm * cm
    target_h = target_w * aspect
    # If too tall, scale down from height
    if target_h > max_height_cm * cm:
        target_h = max_height_cm * cm
        target_w = target_h / aspect
    img = Image(path, width=target_w, height=target_h)
    img.hAlign = 'CENTER'
    return img


# ─── Cover Page ────────────────────────────────────────────────────────────────
def cover_page(story, styles):
    story.append(Spacer(1, 3.5 * cm))

    # College / Department info block
    story.append(Paragraph("Kathmandu University", ParagraphStyle(
        name='college', fontName='Helvetica-Bold', fontSize=13,
        textColor=colors.HexColor('#2C3E50'), alignment=TA_CENTER, leading=18)))
    story.append(Paragraph("School of Engineering", ParagraphStyle(
        name='dept', fontName='Helvetica', fontSize=11,
        textColor=colors.HexColor('#555555'), alignment=TA_CENTER, leading=16, spaceAfter=4)))
    story.append(Spacer(1, 0.5 * cm))

    story.append(HRFlowable(width="70%", thickness=1.5,
                            color=colors.HexColor('#2C3E50'), hAlign='CENTER'))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph(
        "Adaptive Stabilization Techniques for<br/>Ill-Conditioned Linear Regression",
        styles["title"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Numerical Method Project Report", ParagraphStyle(
        name='rtype', fontName='Helvetica-Oblique', fontSize=13,
        textColor=colors.HexColor('#7F8C8D'), alignment=TA_CENTER, leading=18, spaceAfter=4)))

    story.append(Spacer(1, 1.2 * cm))
    story.append(HRFlowable(width="40%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 1.2 * cm))

    story.append(Paragraph("<b>Submitted by:</b>", ParagraphStyle(
        name='sub_label', fontName='Helvetica-Bold', fontSize=11,
        textColor=colors.HexColor('#34495E'), alignment=TA_CENTER, leading=16)))
    story.append(Paragraph(
        "Utsab Baral<br/>Aman Sapkota<br/>Kushagra Poudel",
        ParagraphStyle(name='authors_cv', fontName='Helvetica', fontSize=12,
                       textColor=colors.HexColor('#212121'), alignment=TA_CENTER,
                       leading=20, spaceAfter=8)))

    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("Department of Computer Science &amp; Engineering", ParagraphStyle(
        name='dept2', fontName='Helvetica', fontSize=10,
        textColor=colors.HexColor('#7F8C8D'), alignment=TA_CENTER, leading=14)))
    story.append(Paragraph("2025 / 2026", ParagraphStyle(
        name='year', fontName='Helvetica', fontSize=10,
        textColor=colors.HexColor('#7F8C8D'), alignment=TA_CENTER, leading=14)))


# ─── Abstract Page ─────────────────────────────────────────────────────────────
def abstract_page(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("Abstract", styles["section_header_abstract"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "This project empirically investigates a critical phenomenon in machine learning: the disconnect "
        "between loss minimization and true parameter recovery when the data matrix is ill-conditioned, "
        "known as the <i>\"Optimization Mirage.\"</i> It evaluates multiple optimization algorithms "
        "(Vanilla SGD, Momentum SGD, Adam, AdamW, Ridge L2, and Batch Normalization) across varying "
        "condition numbers (κ) to analyze convergence properties, numerical stability, and bias–variance "
        "tradeoffs. The research demonstrates that modern adaptive optimizers can achieve perfect training "
        "loss while their parameters completely diverge from the true generative solution in highly "
        "ill-conditioned spaces. The expected outcome is a comprehensive understanding of geometric "
        "stabilization and the limitations of empirical loss tracking.",
        styles["abstract_body"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>Keywords:</b> Ill-Conditioned Linear Regression, Optimization Mirage, Stochastic Gradient "
        "Descent, Adam, Batch Normalization, Condition Number.",
        styles["body"]))


# ─── Table of Contents ─────────────────────────────────────────────────────────
def toc_page(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("Table of Contents", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.3 * cm))

    toc = [
        ("Abstract", ""),
        ("Acronyms", ""),
        ("CHAPTER 1: INTRODUCTION", ""),
        ("    1.1  Background", ""),
        ("    1.2  Objectives", ""),
        ("    1.3  Motivation and Significance", ""),
        ("CHAPTER 2: RELATED WORKS", ""),
        ("CHAPTER 3: METHODOLOGY", ""),
        ("    3.1  Research Types", ""),
        ("    3.2  Overall Approach", ""),
        ("    3.3  Breakdown", ""),
        ("CHAPTER 4: SYSTEM REQUIREMENT SPECIFICATION", ""),
        ("    4.1  Software Specification", ""),
        ("    4.2  Hardware Specification", ""),
        ("CHAPTER 5: PROJECT PLANNING AND SCHEDULING", ""),
        ("CHAPTER 6: EMPIRICAL RESULTS", ""),
        ("    6.1  Global Convergence Trajectories", ""),
        ("    6.2  Numerical Stability Under Ill-Conditioning", ""),
        ("    6.3  Bias–Variance Decomposition", ""),
        ("    6.4  Computational Resource Usage", ""),
        ("    6.5  Mechanistic Proofs &amp; Formulas", ""),
        ("CHAPTER 7: CONCLUSION", ""),
        ("CHAPTER 8: REFERENCE", ""),
    ]

    for label, page in toc:
        is_sub = label.startswith("    ")
        st = styles["toc_sub"] if is_sub else styles["toc_item"]
        bold = not is_sub
        txt = f"<b>{label.strip()}</b>" if bold else label.strip()
        story.append(Paragraph(txt, st))


# ─── Acronyms ──────────────────────────────────────────────────────────────────
def acronyms_page(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("Acronyms", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.3 * cm))

    data = [
        ["Acronym", "Meaning"],
        ["SGD", "Stochastic Gradient Descent"],
        ["Adam", "Adaptive Moment Estimation"],
        ["AdamW", "Adam with Decoupled Weight Decay"],
        ["BN", "Batch Normalization"],
        ["MLP", "Multilayer Perceptron"],
        ["SVD", "Singular Value Decomposition"],
        ["CPU", "Central Processing Unit"],
        ["RAM", "Random Access Memory"],
        ["MSE", "Mean Squared Error"],
    ]

    col_widths = [3.5 * cm, 12 * cm]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10.5),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#EBF5FB'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#BDC3C7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)


# ─── Chapter 1: Introduction ───────────────────────────────────────────────────
def chapter1(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 1: INTRODUCTION", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("1.1 Background", styles["section"]))
    story.append(Paragraph(
        "A structural deformity in a data matrix <i>X</i>, quantified by the condition number κ, "
        "characterizes an ill-conditioned matrix. It is the ratio of the maximum to minimum eigenvalues "
        "of the Hessian matrix <i>H = X</i><sup>T</sup><i>X</i>:",
        styles["body"]))
    story.append(formula_image(r"\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}", width_cm=8, fontsize=14))
    story.append(Paragraph(
        "An ill-conditioned matrix (e.g. \u03ba = 10<super>7</super>) creates a loss landscape characterized by steep "
        "ravines along \u03bb<sub>max</sub> and nearly flat, infinitely long valleys along \u03bb<sub>min</sub>. "
        "During optimization, the same step size has to serve the steepest and flattest directions at "
        "once \u2014 and \u03ba measures how badly those demands conflict.",
        styles["body"]))

    story.append(Paragraph("1.2 Objectives", styles["section"]))
    story.append(Paragraph("The main objectives of the Adaptive Stabilization project are:", styles["body"]))
    objectives = [
        ("To empirically investigate the <i>\"Optimization Mirage\"</i> — where modern optimizers achieve "
         "near-perfect training loss while internal parameters diverge catastrophically from reality."),
        ("To compare six major optimization paradigms (Vanilla SGD, Momentum, Adam, AdamW, Ridge L2, and "
         "Batch Normalization) under precisely controlled synthetic environments spanning well-conditioned "
         "(\u03ba = 10) to severely ill-conditioned (\u03ba = 10<super>7</super>) regimes."),
        ("To demonstrate mathematically and empirically the exact failure modes (e.g., Adam's step "
         "inflation, Momentum's random drift) when the structural signal-to-noise ratio approaches zero."),
    ]
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", styles["bullet"]))

    story.append(Paragraph("1.3 Motivation and Significance", styles["section"]))
    story.append(Paragraph(
        "This project dismantles the common assumption that empirical training loss minimization equates "
        "to successful system identification. Modern optimizers frequently achieve minimal loss but diverge "
        "structurally along noise-dominated minimum eigenspaces. Stabilizing parameter recovery through "
        "understanding techniques — such as diagonal preconditioning, L2 penalization, or momentum scaling "
        "— is fundamental to building more robust real-world machine learning pipelines.",
        styles["body"]))


# ─── Chapter 2: Related Works ──────────────────────────────────────────────────
def chapter2(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 2: RELATED WORKS", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    works = [
        ("<b>Polyak (1964) — Heavy-Ball Momentum:</b>",
         "Established that accumulating gradient history provides consistent speed along flat directions "
         "while alternating-sign gradients cancel out, improving convergence rate dependence on κ for "
         "quadratic losses."),
        ("<b>Hoerl &amp; Kennard (1970) — Ridge Regression:</b>",
         "Introduced L2 regularization, shifting eigenvalues by λ to numerically fix ill-conditioned "
         "X<sup>T</sup>X, trading conditioning for bias."),
        ("<b>Kingma &amp; Ba (2014) — Adam:</b>",
         "Developed a running per-coordinate second-moment estimate acting as an approximate diagonal "
         "preconditioner."),
        ("<b>Ioffe &amp; Szegedy (2015) &amp; Santurkar et al. (2018) — Batch Normalization:</b>",
         "Highlighted how BN smooths the loss landscape. Our experiments confirmed that BN applied to a "
         "pure linear model collapses into simple input standardization, destroying absolute scale "
         "information and fundamentally preventing true parameter identification."),
        ("<b>Loshchilov &amp; Hutter (2017) — AdamW:</b>",
         "Demonstrated that L2 regularization integrated inside the loss differs from decoupled weight "
         "decay when adaptive denominators are involved."),
    ]

    for title, desc in works:
        story.append(Paragraph(title, styles["section"]))
        story.append(Paragraph(desc, styles["body"]))


# ─── Chapter 3: Methodology ────────────────────────────────────────────────────
def chapter3(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 3: METHODOLOGY", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("3.1 Research Types", styles["section"]))
    story.append(Paragraph(
        "The project employs an <b>Empirical and Simulation-based Research</b> approach, utilizing exact "
        "mathematical constructions to evaluate real optimization phenomena.",
        styles["body"]))
    story.append(Paragraph(
        "• <b>Modeling/Theoretical Research:</b> Directly manipulating the Singular Value Decomposition "
        "(SVD) of matrices to achieve precise κ condition numbers without approximation.",
        styles["bullet"]))
    story.append(Paragraph(
        "• <b>Experimental Research:</b> Executing large-scale automated sweeps across varying κ values, "
        "optimization methods, and initialization seeds.",
        styles["bullet"]))

    story.append(Paragraph("3.2 Overall Approach", styles["section"]))
    story.append(Paragraph(
        "The evaluation revolves around synthetic linear regression, where the loss landscape is purely "
        "quadratic, yielding a mathematically guaranteed closed-form minimum. This isolates optimizer "
        "dynamics from compounding neural network factors.",
        styles["body"]))
    story.append(Paragraph(
        "• <b>Data Generation:</b> SVD is directly constructed as <i>X = UΣV</i><sup>T</sup> using "
        "σ<sub>min</sub> = σ<sub>max</sub> / √κ. Three datasets are generated with condition numbers "
        "10, 10<super>4</super>, and 10<super>7</super>.",
        styles["bullet"]))
    story.append(Paragraph(
        "• <b>Optimization Loop:</b> Models are trained using 6 methods. Each method undergoes 5 random "
        "initialization seeds to produce statistically robust mean and standard deviation boundaries.",
        styles["bullet"]))
    story.append(Paragraph(
        "• <b>Analytical Metrics Tracking:</b> Tracking includes Train Loss, Test Loss, Parameter Error, "
        "Gradient Norm, and Eigen-Direction Errors.",
        styles["bullet"]))

    story.append(Paragraph("3.3 Breakdown", styles["section"]))
    breakdown = [
        ("<b>Well-Conditioned Space (κ = 10):</b>",
         "Symmetrical, isotropic curvature allowing all optimizers to naturally discover the true "
         "parameters efficiently."),
        ("<b>Moderate Ill-Conditioned Space (\u03ba = 10<super>4</super>):</b>",
         "Elongated valleys appear. Adaptive optimizers quickly settle but leave parameter errors at ~28%."),
        ("<b>Severe Ill-Conditioned Space (\u03ba = 10<super>7</super>):</b>",
         "Ultra-flat valley floor bordered by extreme walls. Adam, Momentum, and AdamW suffer catastrophic "
         "parameter explosion (error \u2248 5.76) due to noise amplification. Vanilla SGD behaves safely due "
         "to gradient-proportional stepping (error falling to \u2248 0.70)."),
        ("<b>Baseline Solvers:</b>",
         "The naive solve (X<sup>T</sup>X)<super>-1</super>X<sup>T</sup>y and the stable pseudoinverse solve "
         "are directly compared up to \u03ba = 10<super>7</super> in float64 precision."),
    ]
    for i, (label, desc) in enumerate(breakdown, 1):
        story.append(Paragraph(f"{i}. {label} {desc}", styles["numbered"]))


# ─── Chapter 4: System Requirements ───────────────────────────────────────────
def chapter4(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 4: SYSTEM REQUIREMENT SPECIFICATION", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("4.1 Software Specification", styles["section"]))
    sw = [
        "<b>Programming Languages:</b> Python 3.8+",
        "<b>Deep Learning Framework:</b> PyTorch 2.0+",
        ("<b>APIs and Libraries:</b> NumPy, SciPy for precision linear algebra operations and direct "
         "solvers; Matplotlib for comprehensive performance visualizations; psutil (version 5.8.0+) for "
         "tracking runtime resources (CPU % and RAM MB)."),
        ("<b>Architecture:</b> A modular pipeline (data_generator.py, models.py, optimizers.py, "
         "train.py, plot_results.py)."),
    ]
    for item in sw:
        story.append(Paragraph(f"• {item}", styles["bullet"]))

    story.append(Paragraph("4.2 Hardware Specification", styles["section"]))
    hw = [
        ("<b>Processor:</b> Standard multi-core CPU (due to purely linear environments with 1,000 "
         "samples, CPU processing is efficient and preferred)."),
        ("<b>RAM:</b> Minimum requirement of ~2 GB (highest tracking during \u03ba = 10<super>7</super> was 942 MB using "
         "AdamW)."),
        ("<b>Precision Type:</b> Forced strictly to float64 during operations to bypass premature "
         "float32 machine epsilon deterioration bounds when evaluating extreme κ."),
    ]
    for item in hw:
        story.append(Paragraph(f"• {item}", styles["bullet"]))


# ─── Chapter 5: Project Planning ──────────────────────────────────────────────
def chapter5(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 5: PROJECT PLANNING AND SCHEDULING", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph(
        "The project's architectural execution was segmented into five phases:",
        styles["body"]))

    # Use Paragraph cells so text wraps properly — plain strings with \n cause overlap
    cell_style = ParagraphStyle(
        name='TblCell', fontName='Helvetica', fontSize=9.5,
        textColor=colors.HexColor('#212121'), leading=14, spaceAfter=0)
    hdr_style = ParagraphStyle(
        name='TblHdr', fontName='Helvetica-Bold', fontSize=10,
        textColor=colors.white, leading=14, spaceAfter=0)
    phase_style = ParagraphStyle(
        name='TblPhase', fontName='Helvetica-Bold', fontSize=9.5,
        textColor=colors.HexColor('#1A252F'), leading=14, spaceAfter=0)

    data = [
        [Paragraph("Phase", hdr_style), Paragraph("Description", hdr_style)],
        [Paragraph("Phase 1: Environment &amp; Setup", phase_style),
         Paragraph("Configuration structures in config.py and precise condition number creation in data_generator.py.", cell_style)],
        [Paragraph("Phase 2: Model &amp; Algorithm Implementation", phase_style),
         Paragraph("Establishing the linear models and MLP variants in models.py along with 6 independent loops in optimizers.py.", cell_style)],
        [Paragraph("Phase 3: Execution and Metrics Tracking", phase_style),
         Paragraph("Orchestration via train.py representing 3 × 6 × 5 = 90 successful training iterations capped at 50,000 epochs.", cell_style)],
        [Paragraph("Phase 4: Diagnostics and Analysis", phase_style),
         Paragraph("Developing plot_results.py for grid visualizations covering convergence plots, stability, bias–variance bar charts, and resource usage graphics.", cell_style)],
        [Paragraph("Phase 5: Documentation &amp; Validation", phase_style),
         Paragraph("Final comprehensive review compiled into completed analysis artifacts detailing the Optimizer Mirage and stabilization metrics.", cell_style)],
    ]

    col_widths = [5 * cm, 11 * cm]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#EBF5FB'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#BDC3C7')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(tbl)


# ─── Chapter 6: Empirical Results ─────────────────────────────────────────────
def chapter6(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 6: EMPIRICAL RESULTS", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    # 6.1
    story.append(Paragraph("6.1 Global Convergence Trajectories", styles["section"]))
    story.append(Paragraph(
        "The grid below illustrates training/test losses, parameter error, and specific eigenvector "
        "errors across all algorithms and condition numbers.",
        styles["body"]))
    img = safe_img(PLOT_PATHS["convergence"], width_cm=15)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Figure 6.1 — Global convergence trajectories: training loss, test loss, parameter error, "
            "and eigen-direction error across all datasets and methods.",
            styles["caption"]))

    # Warning box
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>WARNING \u2014 The Optimization Mirage:</b> For Adam, AdamW, and Momentum in the severe regime "
        "(\u03ba = 10<super>7</super>), train and test losses fall rapidly to the noise floor "
        "(~10<super>-6</super>). However, their corresponding parameter errors "
        "(||\u03b8 \u2212 \u03b8*||<sub>2</sub>) explode up to 5.76. The optimizer satisfies the "
        "objective function by fitting noise along the flat \u03bb<sub>min</sub> directions.",
        styles["warning_box"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "• <b>Vanilla SGD</b> acts as the only unregularized method resistant to this explosion. "
        "Its parameter error steadily trends downward to ≈ 0.70.",
        styles["bullet"]))
    story.append(Paragraph(
        "• <b>BN-on-Linear</b> achieves low loss but permanently locks its relative parameter error "
        "at ≈ 0.98 across all condition numbers due to loss of absolute scale.",
        styles["bullet"]))

    # 6.2
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("6.2 Numerical Stability Under Ill-Conditioning", styles["section"]))
    story.append(Paragraph(
        "Before iterative optimization, direct algebraic solvers were evaluated.",
        styles["body"]))
    img2 = safe_img(PLOT_PATHS["stability"], width_cm=15)
    if img2:
        story.append(img2)
        story.append(Paragraph(
            "Figure 6.2 — Numerical stability of direct solvers as a function of κ.",
            styles["caption"]))
    story.append(Paragraph(
        "In float64 precision, the naive solver (X<sup>T</sup>X)<super>-1</super>X<sup>T</sup>y and the "
        "stable pseudoinverse produced identical relative errors even at \u03ba = 10<super>7</super>, confirming that error "
        "growth stems from inherent noise amplification:",
        styles["body"]))
    story.append(formula_image(
        r"\frac{\|\delta\theta\|}{\|\theta\|} \approx \kappa \times \frac{\|\delta b\|}{\|b\|}",
        width_cm=9, fontsize=13))
    story.append(Paragraph(
        "This is inherent to the problem geometry — not a numerical floating-point breakdown.",
        styles["body"]))

    # 6.3
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("6.3 Bias–Variance Decomposition", styles["section"]))
    story.append(Paragraph(
        "Tracking the mean squared bias (Bias²) and variance across 5 initialization seeds reveals "
        "how failure modes differ between optimizers.",
        styles["body"]))
    img3 = safe_img(PLOT_PATHS["bias_var"], width_cm=15)
    if img3:
        story.append(img3)
        story.append(Paragraph(
            "Figure 6.3 \u2014 Bias\u2013variance decomposition at \u03ba = 10<super>7</super>.",
            styles["caption"]))
    story.append(Paragraph(
        "At \u03ba = 10<super>7</super>, Adam and Momentum exhibit Bias\u00b2 \u2248 33, rendering them catastrophically biased "
        "away from \u03b8*. Conversely, BN-on-Linear suffers from a consistently high bias (0.98) across "
        "all datasets due to the structural destruction of absolute scale information.",
        styles["body"]))

    # 6.4
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("6.4 Computational Resource Usage", styles["section"]))
    story.append(Paragraph(
        "The tracking of Wall Time and Peak RAM consumption (tracked via psutil) highlights the "
        "computational overhead of modern optimizers.",
        styles["body"]))
    img4 = safe_img(PLOT_PATHS["resources"], width_cm=15)
    if img4:
        story.append(img4)
        story.append(Paragraph(
            "Figure 6.4 — Wall time and peak RAM usage across optimizers and datasets.",
            styles["caption"]))

    # 6.5 Mechanistic Proofs
    story.append(PageBreak())
    story.append(Paragraph("6.5 Mechanistic Proofs &amp; Formulas", styles["section"]))
    story.append(Paragraph(
        "The failure of optimizers in ill-conditioned spaces is governed by their distinct update "
        "mechanics. Each formula below is presented with its exact mathematical notation:",
        styles["body"]))

    # 1. Adam
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("1. Adam Parameter Explosion", styles["section"]))
    story.append(formula_image(
        r"\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon}\, m_t",
        width_cm=10, fontsize=14))
    story.append(Paragraph(
        "Along flat eigenvectors, the true gradient ≈ 0, but stochastic noise persists. The second "
        "moment v<sub>t</sub> accumulates this noise, leading to a small denominator that artificially "
        "amplifies noise into large, erratic parameter steps.",
        styles["body"]))

    # 2. Momentum
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("2. Momentum Drift", styles["section"]))
    story.append(formula_image(
        r"v_t = \beta\, v_{t-1} + g_t",
        width_cm=7, fontsize=14))
    story.append(Paragraph(
        "At \u03ba = 10<super>7</super>, optimal momentum is \u03b2 \u2248 0.999. Lacking a strong restoring gradient along flat "
        "minimum eigenspaces, this velocity accumulation becomes an unconstrained random walk.",
        styles["body"]))

    # 3. SGD
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("3. Vanilla SGD's Geometric Safety", styles["section"]))
    story.append(formula_image(
        r"\theta_{t+1} = \theta_t - \eta\, g_t",
        width_cm=7, fontsize=14))
    story.append(Paragraph(
        "Step size is directly proportional to gradient magnitude. Along flat directions, the gradient "
        "is tiny, so the step is inherently limited — rendering SGD immune to noise explosion.",
        styles["body"]))

    # 4. Batch Norm
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("4. Batch Normalization — Scale Invariance", styles["section"]))
    story.append(formula_image(
        r"\hat{X} = (X - \mu)\,\Sigma^{-1/2}",
        width_cm=8, fontsize=14))
    story.append(Paragraph(
        "BN normalizes the feature space. For a linear model, the output is invariant to the scale of θ: "
        "BN(cθ<sup>T</sup>x) = BN(θ<sup>T</sup>x). This mathematically zeroes out the radial gradient, "
        "making true parameter identification impossible.",
        styles["body"]))

    # 5. Ridge
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("5. Ridge Regularization Stabilization", styles["section"]))
    story.append(formula_image(
        r"H_{\text{reg}} = X^T X + \lambda I",
        width_cm=9, fontsize=14))
    story.append(Paragraph(
        "Ridge regression introduces a penalty that shifts the entire spectrum of eigenvalues upward "
        "by λ. This protects the minimum curvature, effectively cutting off adaptive step explosion "
        "by ensuring a continuous restoring force.",
        styles["body"]))


# ─── Chapter 7: Conclusion ─────────────────────────────────────────────────────
def chapter7(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 7: CONCLUSION", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph(
        "The project mathematically and empirically proves that modern optimizers can achieve perfect "
        "loss while internal parameters catastrophically diverge from reality — exposing a critical "
        "vulnerability in applied machine learning pipelines known as the <b>\"Optimization Mirage.\"</b>",
        styles["body"]))
    story.append(Spacer(1, 0.2 * cm))

    conclusions = [
        ("<b>Adam &amp; Momentum</b> suffer from inverse-variance noise amplification and unconstrained "
         "velocity drift along the flat valley floors of highly ill-conditioned (\u03ba = 10<super>7</super>) landscapes."),
        ("<b>Vanilla SGD's</b> simplicity offers inherent geometric safety, proving to be the most "
         "robust unregularized optimizer because its steps directly respect the underlying curvature "
         "magnitude."),
        ("<b>Batch Normalization</b> fundamentally destroys scale information, permanently preventing "
         "parameter recovery."),
        ("<b>Geometric Stabilization (Ridge/L2)</b> provides the ultimate cure by anchoring the Hessian "
         "matrix spectrum, restoring the lost curvature that modern optimizers require to maintain "
         "directional stability."),
    ]
    for i, c in enumerate(conclusions, 1):
        story.append(Paragraph(f"{i}. {c}", styles["numbered"]))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Final Takeaway:</b> Loss monitoring alone is dangerously misleading. Robust system "
        "identification requires structure-aware diagnostics and geometric stabilization, not just "
        "aggressive optimization.",
        styles["warning_box"]))


# ─── Chapter 8: References ─────────────────────────────────────────────────────
def chapter8(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 8: REFERENCE", styles["chapter"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#BDC3C7'), hAlign='CENTER'))
    story.append(Spacer(1, 0.2 * cm))

    refs = [
        "Polyak, B. T. (1964). <i>Some methods of speeding up the convergence of iteration methods.</i>",
        ("Hoerl, A. E., &amp; Kennard, R. W. (1970). <i>Ridge regression: Biased estimation for "
         "nonorthogonal problems.</i>"),
        "Kingma, D. P., &amp; Ba, J. (2014). <i>Adam: A method for stochastic optimization.</i>",
        ("Ioffe, S., &amp; Szegedy, C. (2015). <i>Batch Normalization: Accelerating Deep Network "
         "Training by Reducing Internal Covariate Shift.</i>"),
        ("Santurkar, S., Tsipras, D., Ilyas, A., &amp; Madry, A. (2018). <i>How Does Batch "
         "Normalization Help Optimization?</i>"),
        ("Loshchilov, I., &amp; Hutter, F. (2017). <i>Decoupled Weight Decay Regularization "
         "(AdamW).</i>"),
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"{i}. {ref}", styles["ref"]))


# ─── Main ──────────────────────────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=2.5 * cm,
        bottomMargin=2.2 * cm,
        title="Adaptive Stabilization Techniques — Final Report",
        author="Utsab Baral, Aman Sapkota, Kushagra Poudel",
    )

    styles = make_styles()
    story = []

    cover_page(story, styles)
    abstract_page(story, styles)
    toc_page(story, styles)
    acronyms_page(story, styles)
    chapter1(story, styles)
    chapter2(story, styles)
    chapter3(story, styles)
    chapter4(story, styles)
    chapter5(story, styles)
    chapter6(story, styles)
    chapter7(story, styles)
    chapter8(story, styles)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✓ PDF saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
