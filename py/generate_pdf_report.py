import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

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
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_filename = os.path.join(BASE_DIR, "AdaptiveStabilization_Final_Report.pdf")
    
    # Page setup
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=24, spaceAfter=20, textColor=colors.HexColor("#1e293b")))
    styles.add(ParagraphStyle(name='ReportSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=14, alignment=TA_CENTER, spaceAfter=30, textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle(name='SubSectionHeader', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor("#334155")))
    styles.add(ParagraphStyle(name='CustomBodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=11, spaceAfter=10, leading=16, alignment=TA_JUSTIFY, textColor=colors.HexColor("#1e293b")))
    styles.add(ParagraphStyle(name='CustomBodyTextBold', parent=styles['CustomBodyText'], fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ImportantBox', parent=styles['Normal'], fontName='Helvetica', fontSize=11, spaceAfter=15, spaceBefore=15, leading=16, textColor=colors.HexColor("#b91c1c"), backColor=colors.HexColor("#fef2f2"), borderPadding=10, borderColor=colors.HexColor("#f87171"), borderWidth=1))

    story = []

    # --- TITLE PAGE ---
    story.append(Spacer(1, 100))
    story.append(Paragraph("Adaptive Stabilization Techniques for Ill-Conditioned Linear Regression", styles['ReportTitle']))
    story.append(Paragraph("Final Project Analysis Report", styles['ReportSubtitle']))
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Authors:</b> Utsab Baral, Aman Sapkota, Kushagra Poudel", ParagraphStyle(name='CenterB', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Institution:</b> Kathmandu University, Nepal", ParagraphStyle(name='CenterN', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Course:</b> AIMA 203 – Numerical Methods", ParagraphStyle(name='CenterN', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12)))
    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Abstract</b>", styles['SectionHeader']))
    story.append(Paragraph("This report presents a comprehensive empirical investigation into a critical phenomenon in machine learning: the disconnect between loss minimization and true parameter recovery when the data matrix is highly ill-conditioned. By generating synthetic datasets with exact control over condition numbers ($\\kappa$) and testing 6 different optimization techniques, we demonstrate that modern optimizers (Adam, Momentum, AdamW) can achieve near-perfect training loss while their parameters diverge catastrophically from the true solution — a phenomenon we term the <b>'Optimization Mirage.'</b>", styles['CustomBodyText']))
    
    story.append(PageBreak())

    # --- SECTION 1: Problem Definition ---
    story.append(Paragraph("1. Problem Definition & The Condition Number", styles['SectionHeader']))
    story.append(Paragraph("In real-world scenarios, datasets often feature severe multicollinearity, leading to structural deformities in the data matrix X. This deformity is quantified by the condition number \u03ba, defined as the ratio of the maximum to minimum eigenvalues of the Hessian matrix H = X^T X (\u03ba = \u03bb_max / \u03bb_min).", styles['CustomBodyText']))
    story.append(Paragraph("An ill-conditioned matrix (\u03ba >> 1, e.g., 10^7) creates a loss landscape characterized by steep ravines along \u03bb_max and nearly flat, infinitely long valleys along \u03bb_min. This poses an inherent challenge to optimizers, as true structural signals along flat directions are buried beneath the stochastic noise floor.", styles['CustomBodyText']))
    
    story.append(Paragraph("The Convergence Illusion", styles['SubSectionHeader']))
    story.append(Paragraph("We hypothesize that extreme ill-conditioning allows the optimizer to 'satisfy' the objective function without ever solving the underlying linear system. Movement of thousands of units along flat vectors produces a change in the loss value smaller than standard floating-point precision, enabling parameter explosion despite near-zero loss.", styles['ImportantBox']))

    # --- SECTION 2: Methodology ---
    story.append(Paragraph("2. Experimental Methodology", styles['SectionHeader']))
    story.append(Paragraph("To isolate the optimization dynamics from the compounding variables of deep neural networks (e.g., local minima, saddle points), we utilize Linear Regression, which provides a purely quadratic loss landscape with a mathematically guaranteed constant Hessian.", styles['CustomBodyText']))
    
    story.append(Paragraph("Evaluated Algorithms:", styles['CustomBodyTextBold']))
    data_algos = [
        ["Algorithm", "Key Mechanism", "Expected Behavior"],
        ["Vanilla SGD", "Raw gradient descent", "Slow, steady tracking"],
        ["Momentum SGD", "Velocity accumulation", "Noise drift amplification"],
        ["Adam / AdamW", "Adaptive variance scaling", "Parameter explosion on flat valleys"],
        ["Ridge (L2)", "Regularization penalty", "Anchored stable convergence"],
        ["Batch Normalization", "Scale Invariance", "Loss of parameter magnitude"]
    ]
    t1 = Table(data_algos, colWidths=[1.2*inch, 2*inch, 3*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Datasets:", styles['CustomBodyTextBold']))
    story.append(Paragraph("We generated three synthetic datasets with N=1000 and D=50 using SVD decomposition to exert exact control over \u03ba:", styles['CustomBodyText']))
    data_ds = [
        ["Dataset", "Condition Number (\u03ba)", "Epoch Horizon", "Description"],
        ["Well Conditioned", "10", "1,005", "Isotropic loss bowl; easy to optimize"],
        ["Moderate", "10^4", "6,000", "Elongated valley; divergence begins"],
        ["Severe", "10^7", "50,000", "Ultra-flat valley; testing limits"]
    ]
    t2 = Table(data_ds, colWidths=[1.2*inch, 1.5*inch, 1*inch, 2.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t2)
    story.append(PageBreak())

    # --- SECTION 3: Results ---
    story.append(Paragraph("3. Empirical Results & Visualizations", styles['SectionHeader']))

    # Convergence Plot
    story.append(Paragraph("3.1 Global Convergence Trajectories", styles['SubSectionHeader']))
    story.append(Paragraph("The grid below illustrates the training/test losses, parameter error, and specific eigenvector errors across all algorithms and condition numbers.", styles['CustomBodyText']))
    try:
        img_path = os.path.join(BASE_DIR, "plots", "1_convergence.png")
        if os.path.exists(img_path):
            img = Image(img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 10))
    except Exception as e:
        story.append(Paragraph(f"[Missing Image: 1_convergence.png]", styles['CustomBodyText']))

    story.append(Paragraph("<b>The Optimization Mirage:</b> For Adam, AdamW, and Momentum in the severe regime (\u03ba = 10^7), train and test losses fall rapidly to the noise floor (~10^-6). However, their corresponding parameter errors (||θ - θ*||) explode up to 6.0. The optimizer satisfies the objective function by fitting noise along the flat \u03bb_min directions.", styles['CustomBodyText']))

    story.append(PageBreak())

    # Numerical Stability Plot
    story.append(Paragraph("3.2 Numerical Stability Under Ill-Conditioning", styles['SubSectionHeader']))
    story.append(Paragraph("Before iterative optimization, we analyzed direct algebraic solvers. The plot below demonstrates the relative error scaling of the naive inverse vs the stable pseudoinverse.", styles['CustomBodyText']))
    try:
        img_path2 = os.path.join(BASE_DIR, "plots", "2_numerical_stability.png")
        if os.path.exists(img_path2):
            img2 = Image(img_path2, width=5*inch, height=3.5*inch)
            story.append(img2)
            story.append(Spacer(1, 10))
    except Exception:
        story.append(Paragraph(f"[Missing Image: 2_numerical_stability.png]", styles['CustomBodyText']))
    
    # Bias Variance
    story.append(Paragraph("3.3 Bias-Variance Decomposition", styles['SubSectionHeader']))
    story.append(Paragraph("Tracking the mean squared bias (Bias^2) and variance across 5 initialization seeds reveals how failure modes differ between optimizers.", styles['CustomBodyText']))
    try:
        img_path3 = os.path.join(BASE_DIR, "plots", "3_bias_variance.png")
        if os.path.exists(img_path3):
            img3 = Image(img_path3, width=5*inch, height=3.5*inch)
            story.append(img3)
            story.append(Spacer(1, 10))
    except Exception:
        pass
    
    story.append(Paragraph("At \u03ba = 10^7, Adam and Momentum exhibit a Bias^2 \u2248 33, rendering them catastrophically biased away from \u03b8*. Conversely, BN-on-Linear suffers from a consistently high bias (0.98) across all datasets due to the structural destruction of absolute scale information.", styles['CustomBodyText']))

    story.append(PageBreak())

    # Resource Usage
    story.append(Paragraph("3.4 Computational Resource Usage", styles['SubSectionHeader']))
    story.append(Paragraph("The tracking of Wall Time and Peak RAM consumption (tracked via psutil) highlights the computational overhead of modern optimizers.", styles['CustomBodyText']))
    try:
        img_path4 = os.path.join(BASE_DIR, "plots", "4_resource_usage.png")
        if os.path.exists(img_path4):
            img4 = Image(img_path4, width=5*inch, height=3.5*inch)
            story.append(img4)
            story.append(Spacer(1, 10))
    except Exception:
        pass

    # --- SECTION 4: Conclusion ---
    story.append(Paragraph("4. Conclusion & Architectural Summary", styles['SectionHeader']))
    story.append(Paragraph("The project proves mathematically and empirically that modern optimizers can achieve perfect loss while their internal parameters completely diverge from reality, exposing a critical vulnerability in applied machine learning pipelines.", styles['CustomBodyText']))

    story.append(Paragraph("1. <b>Adam & Momentum:</b> Inverse-variance scaling and velocity accumulation cause noise amplification along flat eigen-directions. Parameter errors explode.<br/>"
                           "2. <b>Vanilla SGD:</b> Surprisingly robust. Because step sizes are directly proportional to gradient magnitude, it naturally ignores flat directions. Parameter error remains bounded.<br/>"
                           "3. <b>Batch Normalization:</b> Destroys scale information, permanently locking parameter error near 1.0 even in well-conditioned spaces.<br/>"
                           "4. <b>Ridge Regularization (L2):</b> Acts as a mathematical anchor. By shifting the entire spectrum of eigenvalues upward, it provides a permanent restoring force toward the origin, stabilizing convergence.", styles['CustomBodyText']))
    
    story.append(Paragraph("Loss monitoring alone is dangerously misleading. Robust system identification requires geometric stabilization.", styles['ImportantBox']))
    
    # Build
    doc.build(story, canvasmaker=NumberedCanvas)

if __name__ == "__main__":
    build_pdf_report()
