import os
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

# ----------------------------------------------------
# Two-Pass Canvas for Page Numbering and Headers
# ----------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#475569")) # Charcoal
        
        # Draw header on all pages after page 1
        if self._pageNumber > 1:
            self.drawString(54, 745, "AIMA 203 – Numerical Methods Progress Report: Adaptive Stabilization")
            self.setStrokeColor(colors.HexColor("#cbd5e1")) # Light grey divider
            self.setLineWidth(0.5)
            self.line(54, 737, 558, 737)
            
        # Draw footer on all pages
        self.drawString(54, 40, "Kathmandu University, Nepal  |  May 2026")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        
        # Bottom thin divider line
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)
        
        self.restoreState()

def build_pdf_report():
    pdf_filename = "/Users/kushagra/Desktop/Numerical Methods/adaptive_stabilization_report.pdf"
    
    # Page setup
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    # ----------------------------------------------------
    # Styles Setup
    # ----------------------------------------------------
    styles = getSampleStyleSheet()
    
    # Color palette
    NAVY = colors.HexColor("#0f172a")        # Slate-900 (Primary)
    TEAL = colors.HexColor("#0d9488")        # Teal-600 (Secondary)
    CHARCOAL = colors.HexColor("#334155")    # Slate-700 (Body)
    LIGHT_BG = colors.HexColor("#f8fafc")    # Slate-50 (Panels)
    BORDER_GREY = colors.HexColor("#e2e8f0") # Slate-200
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=13,
        leading=16,
        textColor=TEAL,
        alignment=TA_LEFT,
        spaceAfter=15
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=CHARCOAL,
        alignment=TA_LEFT,
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=TEAL,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=CHARCOAL,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )
    
    math_style = ParagraphStyle(
        'MathBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9.5,
        leading=12,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=6,
        leftIndent=20,
        rightIndent=20
    )
    
    caption_style = ParagraphStyle(
        'ImageCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=CHARCOAL,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10
    )
    
    # ----------------------------------------------------
    # Building Story
    # ----------------------------------------------------
    story = []
    
    # 1. Document Header Block (Page 1)
    story.append(Paragraph("Adaptive Stabilization Techniques for Ill-Conditioned Problems in ML", title_style))
    story.append(Paragraph("AIMA 203 – Numerical Methods Semester Project Progress Report", subtitle_style))
    
    # Metadata Box
    meta_text = """
    <b>Institution:</b> Kathmandu University, School of Science, Nepal<br/>
    <b>Authors:</b> Kushagra Poudel, Aman Sapkota, Utsab Baral<br/>
    <b>Date:</b> May 2026<br/>
    <b>Repository:</b> <font color="#0d9488">github.com/UB90/AdaptiveStabilization</font>
    """
    story.append(Paragraph(meta_text, meta_style))
    
    # Horizontal separator rule
    header_table = Table([[""]], colWidths=[504])
    header_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 1.5, TEAL),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0)
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))
    
    # 2. Executive Summary / Abstract
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "Modern machine learning optimization is intrinsically a branch of numerical analysis. When training models, "
        "algorithms minimize high-dimensional loss functions, which often involves computing gradients that depend on "
        "the local curvature of the loss surface. For linear or near-linear models, this curvature is determined by "
        "the Gram matrix <i>G = XᵀX</i>, whereas for deep networks, it is dictated by the Hessian matrix of parameter space. "
        "When the features are highly correlated, the condition number <i>κ</i> of these matrices explodes, "
        "inducing severe <i>ill-conditioning</i>. Naive optimization algorithms, such as Vanilla Stochastic Gradient Descent (SGD), "
        "suffer severely in these settings, experiencing gradient oscillation, numerical overflow, or extremely slow convergence. "
        "This project models an ill-conditioned regression problem and compares three mathematical stabilization techniques: "
        "<b>Ridge Regularization (L2)</b>, <b>Batch Normalization</b>, and the <b>Adam Optimizer</b>. Our progress shows that "
        "while standard regularization stabilizes linear computations, coordinate-wise adaptive preconditioning (Adam) and "
        "structural modifications (Batch Normalization) are vastly superior for training multi-layer neural network architectures.",
        body_style
    ))
    
    # 3. Problem Statement & Mathematical Context
    story.append(Paragraph("2. Mathematical Context & The Danger of Ill-Conditioning", h1_style))
    story.append(Paragraph(
        "Let <i>X</i> be a data matrix of dimensions <i>N × d</i>. In linear regression, the closed-form least-squares "
        "solution requires solving the normal equations:", body_style
    ))
    story.append(Paragraph("<b>XᵀX w = Xᵀy</b>", math_style))
    story.append(Paragraph(
        "The numerical stability of solving this system depends on the condition number of the Gram matrix <i>G = XᵀX</i>, defined as:",
        body_style
    ))
    story.append(Paragraph("<b>κ(G) = λ_max(G) / λ_min(G)</b>", math_style))
    story.append(Paragraph(
        "where <i>λ_max</i> and <i>λ_min</i> represent the largest and smallest eigenvalues of <i>G</i>. If <i>κ(G)</i> is small "
        "(close to 1), the loss contours are spherical and standard optimization converges rapidly. However, if features in <i>X</i> "
        "are highly multicollinear, the columns are nearly linearly dependent, pushing <i>λ_min</i> close to zero. Consequently, "
        "<i>κ(G)</i> diverges to infinity, making the matrix near-singular.",
        body_style
    ))
    story.append(Paragraph(
        "In optimization terms, the gradient descent step is governed by the eigenvalues of the Hessian. To guarantee convergence, "
        "the learning rate <i>η</i> is strictly upper-bounded by:", body_style
    ))
    story.append(Paragraph("<b>η < 2 / λ_max(H)</b>", math_style))
    story.append(Paragraph(
        "In an ill-conditioned system where <i>λ_max</i> is large, <i>η</i> must be kept tiny. However, in the direction of the "
        "smallest eigenvalue, the gradient update step size is proportional to <i>η λ_min</i>. Since <i>λ_min</i> is very close to zero, "
        "the optimization update along these directions is virtually negligible. This results in the classic optimization failure mode: "
        "the gradients oscillate wildly perpendicular to the ravine walls while crawling imperceptibly along the ravine floor.",
        body_style
    ))
    
    story.append(PageBreak()) # Clean break to Phase 1 & 2
    
    # 4. Phase 1: Inducing Ill-Conditioning
    story.append(Paragraph("3. Phase 1: Generating Ill-Conditioned Data", h1_style))
    story.append(Paragraph(
        "To systematically analyze this phenomenon, we synthetically induced severe multicollinearity in a dataset. We generated 200 samples "
        "with 10 features, deliberately injecting correlation coefficients larger than 0.95 between feature pairs "
        "(e.g., <i>x₁ ≈ 0.95 · x₀</i> and <i>x₃ ≈ 0.97 · x₂</i>).", body_style
    ))
    story.append(Paragraph(
        "Upon computing the eigenvalues of the resulting Gram matrix, the minimum eigenvalue was found to be <b>2.42 × 10⁻⁴</b> and "
        "the maximum eigenvalue was <b>4.94</b>. This yielded a condition number of:", body_style
    ))
    story.append(Paragraph("<b>κ = 4.94 / (2.42 × 10⁻⁴) ≈ 2.04 × 10⁴</b>", math_style))
    story.append(Paragraph(
        "Due to this extreme ratio, direct analytical inversion of the Gram matrix using <i>np.linalg.inv()</i> failed numerically, "
        "exhibiting significant floating-point round-off errors and yielding values that were numerically singular. "
        "Solving required stable SVD-based methods such as <i>np.linalg.lstsq()</i>.",
        body_style
    ))
    
    # Insert Figure 1
    fig1_path = 'eigenvalue_spectrum.png'
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=320, height=160))
        story.append(Paragraph("Figure 1: Log-scale eigenvalue spectrum of the generated ill-conditioned Gram matrix, demonstrating 4 orders of magnitude spread.", caption_style))
        
    # 5. Phase 2: Vanilla SGD Control Baseline
    story.append(Paragraph("4. Phase 2: Vanilla SGD Baseline (Control Experiment)", h1_style))
    story.append(Paragraph(
        "We set up a control baseline by training a simple 3-layer Multi-Layer Perceptron (10 → 64 → 64 → 1) with ReLU activations. "
        "The model was trained on the synthetic dataset using pure Vanilla SGD (no momentum, no learning rate schedule) at a learning "
        "rate of 0.001. Over 300 epochs, convergence was severely degraded:",
        body_style
    ))
    story.append(Paragraph("• <b>Sluggish Convergence:</b> The MSE loss decreased only from 0.6749 to 0.5198.", bullet_style))
    story.append(Paragraph("• <b>Severe Oscillations:</b> The early loss curve (especially in the first 50 epochs) showed high-frequency oscillations, proving that the updates were bouncing off the ravine walls rather than traversing down the optimization valley.", bullet_style))
    
    # Insert Figure 2
    fig2_path = 'vanilla_sgd_loss.png'
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=320, height=160))
        story.append(Paragraph("Figure 2: Training loss curve under Vanilla SGD, showing oscillations and lack of convergence.", caption_style))

    story.append(PageBreak())

    # 6. Adaptive Stabilization: Theory of Phases 3–5
    story.append(Paragraph("5. Stabilization Techniques (Phases 3–5)", h1_style))
    
    story.append(Paragraph("Phase 3: Ridge Regularization (L2 Regularization)", h2_style))
    story.append(Paragraph(
        "Ridge regularization modifies the loss landscape by adding a quadratic penalty on the parameter weights. In a linear context, "
        "this is mathematically equivalent to adding a diagonal scaling term <i>λI</i> to the Gram matrix. The condition number is modified as:",
        body_style
    ))
    story.append(Paragraph("<b>κ' = (λ_max + λ) / (λ_min + λ)</b>", math_style))
    story.append(Paragraph(
        "By setting a positive regularizer <i>λ</i> (e.g., 0.05), the eigenvalues are shifted away from zero. This effectively shrinks the "
        "condition number (reducing it from 20,400 down to ~100), smoothing the curvature of the ravines.",
        body_style
    ))
    
    story.append(Paragraph("Phase 4: The Adam Optimizer", h2_style))
    story.append(Paragraph(
        "The Adam optimizer addresses ill-conditioning dynamically by maintaining running estimates of the first moment (gradient mean) "
        "and second moment (uncentered variance) of the gradients:", body_style
    ))
    story.append(Paragraph(
        "<b>m_t = β₁ m_{t-1} + (1 - β₁) g_t<br/>"
        "v_t = β₂ v_{t-1} + (1 - β₂) g_t²</b>",
        math_style
    ))
    story.append(Paragraph(
        "The parameter updates are normalized by dividing by the square root of the running variance. Coordinate-wise, this scales "
        "each step relative to the curvature in that direction. Mathematically, it operates as a diagonal Hessian preconditioner, "
        "stretching out the flat directions and dampening the steep ones.",
        body_style
    ))
    
    story.append(Paragraph("Phase 5: Batch Normalization", h2_style))
    story.append(Paragraph(
        "Batch Normalization introduces structural stabilization by standardizing the activations of each layer across a batch. "
        "This standardization restricts the variance of intermediate layers. Numerically, this centers and bounds the eigenvalues of "
        "the Jacobian matrices throughout the network layers, smoothing the overall loss surface and allowing the use of learning rates "
        "that are an order of magnitude larger without causing numerical divergence.",
        body_style
    ))
    
    # 7. Comparative Results
    story.append(Paragraph("6. Experimental Results & Benchmarks", h1_style))
    story.append(Paragraph(
        "We executed all configurations under controlled seeds. The loss curves of all five training configurations are compared "
        "in Figure 3 on a logarithmic scale.", body_style
    ))
    
    # Insert Comparison Figure
    fig3_path = 'comparison_loss_curves.png'
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=360, height=216))
        story.append(Paragraph("Figure 3: Log-scale loss convergence curves comparison for all stabilization techniques.", caption_style))
        
    # Table of Results
    story.append(Spacer(1, 8))
    table_data = [
        ["Configuration", "Learning Rate", "Final MSE Loss (300 Epochs)", "Convergence Quality"],
        ["Vanilla SGD (Control)", "0.001", "0.5198", "Extremely Slow / Oscillating"],
        ["Ridge (L2) Regularization", "0.001", "0.4812", "Slightly Smoother / Slow"],
        ["Batch Normalization (Low)", "0.001", "0.4501", "Stable / Slow"],
        ["Batch Normalization (High)", "0.010", "0.2815", "Stable / Moderately Fast"],
        ["Adam Optimizer", "0.005", "0.0048", "Instant (< 20 epochs) / Stable"]
    ]
    
    results_table = Table(table_data, colWidths=[150, 80, 150, 124])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_GREY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG])
    ]))
    story.append(results_table)
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("7. Discussion & Takeaways", h1_style))
    story.append(Paragraph(
        "The experiments provide clean mathematical verification of optimization dynamics. "
        "L2 Regularization (Ridge) shifts the eigenvalues of the linear system, making the calculations numerically stable, "
        "but because the network's layers remain subject to coordinate-wise imbalances, convergence speed is barely improved. "
        "Batch Normalization structurally smooths the landscape, which is proven by its ability to train at <i>lr = 0.01</i> (10x higher) "
        "without exploding, resulting in a substantially lower final loss. Finally, the Adam Optimizer is clearly the most effective "
        "remedy. By dynamically scaling the learning rate of each parameter, it directly neutralizes the ill-conditioned scale "
        "mismatches, yielding convergence that is orders of magnitude faster than any SGD-based option.",
        body_style
    ))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Report compiled successfully!")

if __name__ == '__main__':
    build_pdf_report()
