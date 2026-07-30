import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

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
        self.setFillColor(colors.HexColor("#475569"))
        if self._pageNumber > 1:
            self.drawString(54, 745, "AIMA 203 – Numerical Methods Progress Report: Adaptive Stabilization")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 737, 558, 737)
            
        self.drawString(54, 40, "Kathmandu University, Nepal  |  May 2026")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)
        
        self.restoreState()

def build_pdf_report():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_filename = os.path.join(BASE_DIR, "AdaptiveStabilization_Final_Report.pdf")
    
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=24, spaceAfter=20, textColor=colors.HexColor("#1e293b")))
    styles.add(ParagraphStyle(name='ReportSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=14, alignment=TA_CENTER, spaceAfter=30, textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle(name='SubSectionHeader', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor("#334155")))
    styles.add(ParagraphStyle(name='CustomBodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=11, spaceAfter=10, leading=16, alignment=TA_JUSTIFY, textColor=colors.HexColor("#1e293b")))
    styles.add(ParagraphStyle(name='CustomListText', parent=styles['Normal'], fontName='Helvetica', fontSize=11, spaceAfter=5, leading=16, alignment=TA_LEFT, textColor=colors.HexColor("#1e293b")))
    styles.add(ParagraphStyle(name='ImportantBox', parent=styles['Normal'], fontName='Helvetica', fontSize=11, spaceAfter=15, spaceBefore=15, leading=16, textColor=colors.HexColor("#b91c1c"), backColor=colors.HexColor("#fef2f2"), borderPadding=10, borderColor=colors.HexColor("#f87171"), borderWidth=1))
    
    story = []

    # Title Page
    story.append(Spacer(1, 100))
    story.append(Paragraph("Adaptive Stabilization Techniques for Ill-Conditioned Linear Regression", styles['ReportTitle']))
    story.append(Paragraph("Final Project Analysis Report", styles['ReportSubtitle']))
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Submitted by:</b>", ParagraphStyle(name='CenterB', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12)))
    story.append(Paragraph("Utsab Baral<br/>Aman Sapkota<br/>Kushagra Poudel", ParagraphStyle(name='CenterN', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, leading=16)))
    story.append(Spacer(1, 40))
    
    # Abstract
    story.append(Paragraph("<b>Abstract</b>", styles['SectionHeader']))
    story.append(Paragraph("This project empirically investigates a critical phenomenon in machine learning: the disconnect between loss minimization and true parameter recovery when the data matrix is ill-conditioned, known as the 'Optimization Mirage.' It evaluates multiple optimization algorithms (Vanilla SGD, Momentum SGD, Adam, AdamW, Ridge L2, and Batch Normalization) across varying condition numbers (κ) to analyze convergence properties, numerical stability, and bias-variance tradeoffs. The research demonstrates that modern adaptive optimizers can achieve perfect training loss while their parameters completely diverge from the true generative solution in highly ill-conditioned spaces. The expected outcome is a comprehensive understanding of geometric stabilization and the limitations of empirical loss tracking.", styles['CustomBodyText']))
    story.append(Paragraph("<b>Keywords:</b> Ill-Conditioned Linear Regression, Optimization Mirage, Stochastic Gradient Descent, Adam, Batch Normalization, Condition Number.", styles['CustomBodyText']))
    
    story.append(PageBreak())

    # Table of Content
    story.append(Paragraph("Table of Content", styles['SectionHeader']))
    toc_items = [
        "1. Abstract",
        "2. Acronyms",
        "3. CHAPTER 1: INTRODUCTION",
        "   - 1.1 Background",
        "   - 1.2 Objectives",
        "   - 1.3 Motivation and Significance",
        "4. CHAPTER 2: RELATED WORKS",
        "5. CHAPTER 3: METHODOLOGY",
        "   - 3.1 Research Types",
        "   - 3.2 Overall Approach",
        "   - 3.3 Breakdown",
        "6. CHAPTER 4: SYSTEM REQUIREMENT SPECIFICATION",
        "   - 4.1 Software Specification",
        "   - 4.2 Hardware Specification",
        "7. CHAPTER 5: PROJECT PLANNING AND SCHEDULING",
        "8. CHAPTER 6: EMPIRICAL RESULTS",
        "   - 6.1 Global Convergence Trajectories",
        "   - 6.2 Numerical Stability Under Ill-Conditioning",
        "   - 6.3 Bias-Variance Decomposition",
        "   - 6.4 Computational Resource Usage",
        "   - 6.5 Mechanistic Proofs & Formulas",
        "9. CHAPTER 7: CONCLUSION",
        "10. CHAPTER 8: REFERENCE"
    ]
    for item in toc_items:
        if item.startswith("   -"):
            story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;" + item, styles['CustomListText']))
        else:
            story.append(Paragraph(item, styles['CustomListText']))

    story.append(PageBreak())
    
    # Acronyms
    story.append(Paragraph("Acronyms", styles['SectionHeader']))
    acronyms = [
        "<b>SGD</b>: Stochastic Gradient Descent",
        "<b>Adam</b>: Adaptive Moment Estimation",
        "<b>AdamW</b>: Adam with Decoupled Weight Decay",
        "<b>BN</b>: Batch Normalization",
        "<b>MLP</b>: Multilayer Perceptron",
        "<b>SVD</b>: Singular Value Decomposition",
        "<b>CPU</b>: Central Processing Unit",
        "<b>RAM</b>: Random Access Memory",
        "<b>MSE</b>: Mean Squared Error"
    ]
    for acr in acronyms:
        story.append(Paragraph("- " + acr, styles['CustomListText']))

    story.append(PageBreak())

    # Chapter 1
    story.append(Paragraph("CHAPTER 1: INTRODUCTION", styles['SectionHeader']))
    story.append(Paragraph("1.1 Background", styles['SubSectionHeader']))
    story.append(Paragraph("A structural deformity in a data matrix X, quantified by the condition number κ, characterizes an ill-conditioned matrix. It is the ratio of the maximum to minimum eigenvalues of the Hessian matrix H = X<sup>T</sup> X (i.e. κ = λ<sub>max</sub> / λ<sub>min</sub>). An ill-conditioned matrix (e.g. κ = 10<sup>7</sup>) creates a loss landscape characterized by steep ravines along λ<sub>max</sub> and nearly flat, infinitely long valleys along λ<sub>min</sub>. During optimization, the same step size has to serve the steepest and flattest directions at once, and κ measures how badly those demands conflict.", styles['CustomBodyText']))
    
    story.append(Paragraph("1.2 Objectives", styles['SubSectionHeader']))
    story.append(Paragraph("The main objectives of the Adaptive Stabilization project are:<br/>"
                           "- To empirically investigate the 'Optimization Mirage' where modern optimizers achieve near-perfect training loss while internal parameters diverge catastrophically from reality.<br/>"
                           "- To compare six major optimization paradigms (Vanilla SGD, Momentum, Adam, AdamW, Ridge L2, and Batch Normalization) under precisely controlled synthetic environments spanning well-conditioned (κ=10) to severely ill-conditioned (κ=10<sup>7</sup>) regimes.<br/>"
                           "- To demonstrate mathematically and empirically the exact failure modes (e.g., Adam's step inflation, Momentum's random drift) when the structural signal-to-noise ratio approaches zero.", styles['CustomBodyText']))
    
    story.append(Paragraph("1.3 Motivation and Significance", styles['SubSectionHeader']))
    story.append(Paragraph("This project dismantles the common assumption that empirical training loss minimization equates to successful system identification. Modern optimizers frequently achieve minimal loss but diverge structurally along noise-dominated minimum eigenspaces. Stabilizing parameter recovery through understanding techniques—such as diagonal preconditioning, L2 penalization, or momentum scaling—is fundamental to building more robust real-world machine learning pipelines.", styles['CustomBodyText']))

    story.append(PageBreak())

    # Chapter 2
    story.append(Paragraph("CHAPTER 2: RELATED WORKS", styles['SectionHeader']))
    related_works = [
        "<b>Polyak (1964) — Heavy-Ball Momentum</b>: Established that accumulating gradient history provides consistent speed along flat directions while alternating-sign gradients cancel out, improving convergence rate dependence on κ for quadratic losses.",
        "<b>Hoerl & Kennard (1970) — Ridge Regression</b>: Introduced L2 regularization, shifting eigenvalues by λ to numerically fix ill-conditioned X<sup>T</sup> X, trading conditioning for bias.",
        "<b>Kingma & Ba (2014) — Adam</b>: Developed a running per-coordinate second-moment estimate acting as an approximate diagonal preconditioner.",
        "<b>Ioffe & Szegedy (2015) & Santurkar et al. (2018) — Batch Normalization</b>: Highlighted how BN smooths the loss landscape. Our experiments confirmed that BN applied to a pure linear model collapses into simple input standardization, destroying absolute scale information and fundamentally preventing true parameter identification.",
        "<b>Loshchilov & Hutter (2017) — AdamW</b>: Demonstrated that L2 regularization integrated inside the loss differs from decoupled weight decay when adaptive denominators are involved."
    ]
    for rw in related_works:
        story.append(Paragraph("- " + rw, styles['CustomListText']))

    story.append(PageBreak())

    # Chapter 3
    story.append(Paragraph("CHAPTER 3: METHODOLOGY", styles['SectionHeader']))
    
    story.append(Paragraph("3.1 Research Types", styles['SubSectionHeader']))
    story.append(Paragraph("The project employs an <b>Empirical and Simulation-based Research</b> approach, utilizing exact mathematical constructions to evaluate real optimization phenomena.<br/>"
                           "- <b>Modeling/Theoretical Research:</b> Directly manipulating the Singular Value Decomposition (SVD) of matrices to achieve precise κ condition numbers without approximation.<br/>"
                           "- <b>Experimental Research:</b> Executing large-scale automated sweeps across varying κ values, optimization methods, and initialization seeds.", styles['CustomBodyText']))

    story.append(Paragraph("3.2 Overall Approach", styles['SubSectionHeader']))
    story.append(Paragraph("The evaluation revolves around synthetic linear regression, where the loss landscape is purely quadratic, yielding a mathematically guaranteed closed-form minimum. This isolates optimizer dynamics from compounding neural network factors.<br/>"
                           "- <b>Data Generation:</b> SVD is directly constructed X = U Σ V<sup>T</sup> using σ<sub>min</sub> = σ<sub>max</sub> / √κ. Three datasets are generated with condition numbers 10, 10<sup>4</sup>, and 10<sup>7</sup>.<br/>"
                           "- <b>Optimization Loop:</b> Models are trained using 6 methods. Each method undergoes 5 random initialization seeds to produce statistically robust mean and standard deviation boundaries.<br/>"
                           "- <b>Analytical Metrics Tracking:</b> Tracking includes Train Loss, Test Loss, Parameter Error, Gradient Norm, and Eigen-Direction Errors.", styles['CustomBodyText']))
    
    story.append(Paragraph("3.3 Breakdown", styles['SubSectionHeader']))
    story.append(Paragraph("1. <b>Well-Conditioned Space (κ = 10):</b> Symmetrical, isotropic curvature allowing all optimizers to naturally discover the true parameters efficiently.<br/>"
                           "2. <b>Moderate Ill-Conditioned Space (κ = 10<sup>4</sup>):</b> Elongated valleys appear. Adaptive optimizers quickly settle but leave parameter errors at ~28%.<br/>"
                           "3. <b>Severe Ill-Conditioned Space (κ = 10<sup>7</sup>):</b> Ultra-flat valley floor bordered by extreme walls. Adam, Momentum, and AdamW suffer catastrophic parameter explosion (error ≈ 5.76) due to noise amplification. Vanilla SGD behaves safely due to gradient-proportional stepping (error falling to ≈ 0.70).<br/>"
                           "4. <b>Baseline Solvers:</b> The naive solve (X<sup>T</sup> X)<sup>-1</sup> X<sup>T</sup> y and the stable pseudoinverse solve are directly compared up to κ=10<sup>7</sup> in float64 precision.", styles['CustomBodyText']))

    story.append(PageBreak())

    # Chapter 4
    story.append(Paragraph("CHAPTER 4: SYSTEM REQUIREMENT SPECIFICATION", styles['SectionHeader']))
    
    story.append(Paragraph("4.1 Software Specification", styles['SubSectionHeader']))
    story.append(Paragraph("- <b>Programming Languages:</b> Python 3.8+<br/>"
                           "- <b>Deep Learning Framework:</b> PyTorch 2.0+<br/>"
                           "- <b>APIs and Libraries:</b><br/>"
                           "&nbsp;&nbsp;* NumPy, SciPy for precision linear algebra operations and direct solvers.<br/>"
                           "&nbsp;&nbsp;* Matplotlib for comprehensive performance visualizations.<br/>"
                           "&nbsp;&nbsp;* psutil (version 5.8.0+) for tracking runtime resources (CPU % and RAM MB).<br/>"
                           "- <b>Architecture:</b> A modular pipeline (data_generator.py, models.py, optimizers.py, train.py, plot_results.py).", styles['CustomBodyText']))
    
    story.append(Paragraph("4.2 Hardware Specification", styles['SubSectionHeader']))
    story.append(Paragraph("- <b>Processor:</b> Standard multi-core CPU (due to purely linear environments with 1,000 samples, CPU processing is efficient and preferred).<br/>"
                           "- <b>RAM:</b> Minimum requirement of ~2 GB (highest tracking during κ=10<sup>7</sup> was 942 MB using AdamW).<br/>"
                           "- <b>Precision Type:</b> Forced strictly to float64 during operations to bypass premature float32 machine epsilon deterioration bounds when evaluating extreme κ.", styles['CustomBodyText']))

    # Chapter 5
    story.append(Paragraph("CHAPTER 5: PROJECT PLANNING AND SCHEDULING", styles['SectionHeader']))
    story.append(Paragraph("The project's architectural execution was segmented as follows:<br/>"
                           "- <b>Phase 1: Environment & Setup</b> - Configuration structures in config.py and precise condition number creation in data_generator.py.<br/>"
                           "- <b>Phase 2: Model & Algorithm Implementation</b> - Establishing the linear models and MLP variants in models.py along with 6 independent loops in optimizers.py.<br/>"
                           "- <b>Phase 3: Execution and Metrics Tracking</b> - Orchestration via train.py representing 3 × 6 × 5 = 90 successful training iterations capped at 50,000 epochs.<br/>"
                           "- <b>Phase 4: Diagnostics and Analysis</b> - Developing plot_results.py for grid visualizations covering convergence plots, stability, bias-variance bar charts, and resource usage graphics.<br/>"
                           "- <b>Phase 5: Documentation & Validation</b> - Final comprehensive review compiled into the completed analysis artifacts detailing the Optimizer Mirage and stabilization metrics.", styles['CustomBodyText']))

    story.append(PageBreak())

    # Chapter 6 (Results)
    story.append(Paragraph("CHAPTER 6: EMPIRICAL RESULTS", styles['SectionHeader']))
    
    # 6.1
    story.append(Paragraph("6.1 Global Convergence Trajectories", styles['SubSectionHeader']))
    story.append(Paragraph("The grid below illustrates the training/test losses, parameter error, and specific eigenvector errors across all algorithms and condition numbers.", styles['CustomBodyText']))
    
    # Add plot 1
    img_path1 = os.path.join(BASE_DIR, "plots", "1_convergence.png")
    if os.path.exists(img_path1):
        story.append(Image(img_path1, width=6.5*inch, height=4*inch))
        story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>The Optimization Mirage:</b> For Adam, AdamW, and Momentum in the severe regime (κ = 10<sup>7</sup>), train and test losses fall rapidly to the noise floor (~ 10<sup>-6</sup>). However, their corresponding parameter errors (||θ - θ*||<sub>2</sub>) explode up to 5.76. The optimizer satisfies the objective function by fitting noise along the flat λ<sub>min</sub> directions.", styles['ImportantBox']))
    story.append(Paragraph("- <b>Vanilla SGD</b> acts as the only unregularized method resistant to this explosion. Its parameter error steadily trends downward to ≈ 0.70.<br/>"
                           "- <b>BN-on-Linear</b> achieves low loss but permanently locks its relative parameter error at ≈ 0.98 across all condition numbers due to loss of absolute scale.", styles['CustomBodyText']))

    story.append(Paragraph("6.2 Numerical Stability Under Ill-Conditioning", styles['SubSectionHeader']))
    story.append(Paragraph("Before iterative optimization, direct algebraic solvers were evaluated. In float64 precision, the naive solver (X<sup>T</sup> X)<sup>-1</sup> X<sup>T</sup> y and stable pseudoinverse produced identical relative errors even at κ=10<sup>7</sup>, confirming that error growth stems from inherent noise amplification (||δθ|| / ||θ|| ≈ κ × ||δb|| / ||b||) rather than numerical floating-point breakdown.", styles['CustomBodyText']))
    
    # Add plot 2
    img_path2 = os.path.join(BASE_DIR, "plots", "2_numerical_stability.png")
    if os.path.exists(img_path2):
        story.append(Image(img_path2, width=4*inch, height=3*inch))
        story.append(Spacer(1, 10))
        
    story.append(PageBreak())

    # 6.3
    story.append(Paragraph("6.3 Bias-Variance Decomposition", styles['SubSectionHeader']))
    story.append(Paragraph("Tracking the mean squared bias (Bias<sup>2</sup>) and variance across 5 initialization seeds reveals how failure modes differ between optimizers.", styles['CustomBodyText']))
    
    # Add plot 3
    img_path3 = os.path.join(BASE_DIR, "plots", "3_bias_variance.png")
    if os.path.exists(img_path3):
        story.append(Image(img_path3, width=5.5*inch, height=3.5*inch))
        story.append(Spacer(1, 10))
        
    story.append(Paragraph("At κ = 10<sup>7</sup>, Adam and Momentum exhibit a Bias<sup>2</sup> ≈ 33, rendering them catastrophically biased away from θ*. Conversely, BN-on-Linear suffers from a consistently high bias (0.98) across all datasets due to the structural destruction of absolute scale information.", styles['CustomBodyText']))

    # 6.4
    story.append(Paragraph("6.4 Computational Resource Usage", styles['SubSectionHeader']))
    story.append(Paragraph("The tracking of Wall Time and Peak RAM consumption (tracked via psutil) highlights the computational overhead of modern optimizers.", styles['CustomBodyText']))
    
    # Add plot 4
    img_path4 = os.path.join(BASE_DIR, "plots", "4_resource_usage.png")
    if os.path.exists(img_path4):
        story.append(Image(img_path4, width=5.5*inch, height=3.5*inch))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # 6.5
    story.append(Paragraph("6.5 Mechanistic Proofs & Formulas", styles['SubSectionHeader']))
    story.append(Paragraph("The failure of optimizers in ill-conditioned spaces is governed by their distinct update mechanics:", styles['CustomBodyText']))
    story.append(Paragraph("<b>1. Adam Parameter Explosion:</b><br/>θ<sub>t+1</sub> = θ<sub>t</sub> - (α / (√v<sub>t</sub> + ε)) m<sub>t</sub><br/>Along flat eigenvectors, the true gradient ≈ 0, but stochastic noise persists. The second moment v<sub>t</sub> accumulates this noise, leading to a small denominator that artificially amplifies noise into large, erratic parameter steps.", styles['CustomBodyText']))
    story.append(Paragraph("<b>2. Momentum Drift:</b><br/>v<sub>t</sub> = β v<sub>t-1</sub> + g<sub>t</sub><br/>At κ=10<sup>7</sup>, optimal momentum is β ≈ 0.999. Lacking a strong restoring gradient along flat minimum eigenspaces, this velocity accumulation becomes an unconstrained random walk.", styles['CustomBodyText']))
    story.append(Paragraph("<b>3. Vanilla SGD's Geometric Safety:</b><br/>θ<sub>t+1</sub> = θ<sub>t</sub> - η g<sub>t</sub><br/>Step size is directly proportional to gradient magnitude. Along flat directions, the gradient is tiny, rendering SGD immune to noise explosion.", styles['CustomBodyText']))
    story.append(Paragraph("<b>4. Batch Normalization (Scale Invariance):</b><br/>X̂ = (X - μ) Σ<sup>-1/2</sup><br/>BN normalizes the feature space. For a linear model, the output is invariant to the scale of θ: BN(cθ<sup>T</sup> x) = BN(θ<sup>T</sup> x). This mathematically zeroes out the radial gradient, making true parameter identification impossible.", styles['CustomBodyText']))
    story.append(Paragraph("<b>5. Ridge Regularization Stabilization:</b><br/>H<sub>reg</sub> = X<sup>T</sup> X + λ I<br/>Ridge regression introduces a penalty that shifts the entire spectrum of eigenvalues upward by λ. This protects the minimum curvature, effectively cutting off adaptive step explosion.", styles['CustomBodyText']))

    story.append(PageBreak())

    # Chapter 7 (Conclusion)
    story.append(Paragraph("CHAPTER 7: CONCLUSION", styles['SectionHeader']))
    story.append(Paragraph("The project mathematically and empirically proves that modern optimizers can achieve perfect loss while internal parameters catastrophically diverge from reality—exposing a critical vulnerability in applied machine learning pipelines known as the <b>'Optimization Mirage.'</b>", styles['CustomBodyText']))
    
    story.append(Paragraph("1. <b>Adam & Momentum</b> suffer from inverse-variance noise amplification and unconstrained velocity drift along the flat valley floors of highly ill-conditioned (κ=10<sup>7</sup>) landscapes.<br/><br/>"
                           "2. <b>Vanilla SGD's</b> simplicity offers inherent geometric safety, proving to be the most robust unregularized optimizer because its steps directly respect the underlying curvature magnitude.<br/><br/>"
                           "3. <b>Batch Normalization</b> fundamentally destroys scale information, permanently preventing parameter recovery.<br/><br/>"
                           "4. <b>Geometric Stabilization (Ridge/L2)</b> provides the ultimate cure by anchoring the Hessian matrix spectrum, restoring the lost curvature that modern optimizers require to maintain directional stability.", styles['CustomBodyText']))
    
    story.append(Paragraph("<b>Final Takeaway:</b> Loss monitoring alone is dangerously misleading. Robust system identification requires structure-aware diagnostics and geometric stabilization, not just aggressive optimization.", styles['ImportantBox']))

    story.append(PageBreak())

    # Chapter 8
    story.append(Paragraph("CHAPTER 8: REFERENCE", styles['SectionHeader']))
    references = [
        "1. Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods.",
        "2. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems.",
        "3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.",
        "4. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.",
        "5. Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization?",
        "6. Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization (AdamW)."
    ]
    for ref in references:
        story.append(Paragraph(ref, styles['CustomListText']))

    doc.build(story, canvasmaker=NumberedCanvas)

if __name__ == "__main__":
    build_pdf_report()
