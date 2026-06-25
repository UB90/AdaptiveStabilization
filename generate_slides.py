import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    # Set slide dimensions to widescreen (16:9)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ----------------------------------------------------
    # Design Constants
    # ----------------------------------------------------
    COLOR_BG = RGBColor(248, 250, 252)       # Soft off-white
    COLOR_NAVY = RGBColor(15, 23, 42)        # Primary Navy
    COLOR_TEAL = RGBColor(13, 148, 136)      # Secondary Teal
    COLOR_CHARCOAL = RGBColor(71, 85, 105)   # Body text Charcoal
    COLOR_WHITE = RGBColor(255, 255, 255)
    
    FONT_TITLE = "Trebuchet MS"
    FONT_BODY = "Arial"

    # Helper to set background color
    def set_bg(slide, color=COLOR_BG):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    # Helper to create a slide with header and footer
    def add_standard_slide(title_text):
        slide = prs.slides.add_slide(prs.slide_layouts[6]) # blank layout
        set_bg(slide)
        
        # Add Header
        title_box = slide.shapes.add_textbox(Inches(0.75), Inches(0.5), Inches(11.83), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = FONT_TITLE
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = COLOR_NAVY
        
        # Add Thin Divider Line
        shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, 
            Inches(0.75), Inches(1.3), Inches(11.83), Inches(0.04)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_TEAL
        shape.line.fill.background()
        
        # Add Footer
        footer_box = slide.shapes.add_textbox(Inches(0.75), Inches(6.9), Inches(11.83), Inches(0.4))
        ftf = footer_box.text_frame
        fp = ftf.paragraphs[0]
        fp.text = "AIMA 203 – Numerical Methods  |  Kathmandu University  |  Progress Report"
        fp.font.name = FONT_BODY
        fp.font.size = Pt(10)
        fp.font.color.rgb = COLOR_CHARCOAL
        
        return slide

    # ----------------------------------------------------
    # SLIDE 1: Title Slide
    # ----------------------------------------------------
    slide_layout = prs.slide_layouts[6] # blank
    slide1 = prs.slides.add_slide(slide_layout)
    set_bg(slide1, COLOR_NAVY)
    
    # Title Box (White text on Navy background)
    title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.33), Inches(2.2))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Adaptive Stabilization Techniques\nfor Ill-Conditioned Problems in ML"
    p.font.name = FONT_TITLE
    p.font.size = Pt(42)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    
    # Subtitle Box (Teal accent text)
    sub_box = slide1.shapes.add_textbox(Inches(1.0), Inches(3.9), Inches(11.33), Inches(0.8))
    stf = sub_box.text_frame
    stf.word_wrap = True
    sp = stf.paragraphs[0]
    sp.text = "A comparative study on numerical regularizations and optimizers"
    sp.font.name = FONT_BODY
    sp.font.size = Pt(20)
    sp.font.italic = True
    sp.font.color.rgb = COLOR_TEAL
    
    # Authors and Course Info
    info_box = slide1.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(11.33), Inches(1.5))
    itf = info_box.text_frame
    itf.word_wrap = True
    
    ip1 = itf.paragraphs[0]
    ip1.text = "Course: AIMA 203 – Numerical Methods"
    ip1.font.name = FONT_BODY
    ip1.font.size = Pt(14)
    ip1.font.bold = True
    ip1.font.color.rgb = COLOR_BG
    
    ip2 = itf.add_paragraph()
    ip2.text = "Authors: Kushagra Poudel, Aman Sapkota, Utsab Baral"
    ip2.font.name = FONT_BODY
    ip2.font.size = Pt(13)
    ip2.font.color.rgb = COLOR_BG
    
    ip3 = itf.add_paragraph()
    ip3.text = "Kathmandu University, Nepal  |  May 2026"
    ip3.font.name = FONT_BODY
    ip3.font.size = Pt(12)
    ip3.font.color.rgb = COLOR_TEAL

    # ----------------------------------------------------
    # SLIDE 2: Introduction & Motivation
    # ----------------------------------------------------
    slide2 = add_standard_slide("Introduction & Motivation")
    
    # Left Content Column (Core theory and explanation)
    text_box = slide2.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(5.8), Inches(4.8))
    tf = text_box.text_frame
    tf.word_wrap = True
    
    # Point 1
    p1 = tf.paragraphs[0]
    p1.text = "• ML Training as Numerical Computation"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(4)
    
    p1_sub = tf.add_paragraph()
    p1_sub.text = "Modern ML training involves solving linear systems, computing high-dimensional matrix operations, and performing iterative optimization. Numerical stability is critical."
    p1_sub.font.name = FONT_BODY
    p1_sub.font.size = Pt(14)
    p1_sub.font.color.rgb = COLOR_CHARCOAL
    p1_sub.space_after = Pt(12)
    
    # Point 2
    p2 = tf.add_paragraph()
    p2.text = "• The Threat of Ill-Conditioning"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_NAVY
    p2.space_after = Pt(4)
    
    p2_sub = tf.add_paragraph()
    p2_sub.text = "When data exhibits high multicollinearity, the Gram matrix G = XᵀX has a massive condition number κ. In linear/deep landscapes, this forms highly distorted loss ravines."
    p2_sub.font.name = FONT_BODY
    p2_sub.font.size = Pt(14)
    p2_sub.font.color.rgb = COLOR_CHARCOAL
    p2_sub.space_after = Pt(12)
    
    # Point 3
    p3 = tf.add_paragraph()
    p3.text = "• Optimization Failure Mode"
    p3.font.name = FONT_TITLE
    p3.font.size = Pt(18)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_NAVY
    p3.space_after = Pt(4)
    
    p3_sub = tf.add_paragraph()
    p3_sub.text = "Standard gradient descent oscillates wildly in high-curvature directions and crawls in flat ones, leading to gradient explosion, vanishing steps, and divergence."
    p3_sub.font.name = FONT_BODY
    p3_sub.font.size = Pt(14)
    p3_sub.font.color.rgb = COLOR_CHARCOAL

    # Right Content Column (Impact graphic or callout box)
    callout_bg = slide2.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, 
        Inches(7.2), Inches(1.8), Inches(5.38), Inches(4.4)
    )
    callout_bg.fill.solid()
    callout_bg.fill.fore_color.rgb = COLOR_NAVY
    callout_bg.line.color.rgb = COLOR_TEAL
    callout_bg.line.width = Pt(1.5)
    
    callout_box = slide2.shapes.add_textbox(Inches(7.4), Inches(2.0), Inches(5.0), Inches(4.0))
    ctf = callout_box.text_frame
    ctf.word_wrap = True
    
    cp1 = ctf.paragraphs[0]
    cp1.text = "Numerical Method Focus:"
    cp1.font.name = FONT_TITLE
    cp1.font.size = Pt(20)
    cp1.font.bold = True
    cp1.font.color.rgb = COLOR_TEAL
    cp1.space_after = Pt(14)
    
    cp2 = ctf.add_paragraph()
    cp2.text = "Condition Number (κ):\nκ = λ_max / λ_min\n\nIf κ >> 1, small computational round-off errors in matrix math or gradient updates get amplified exponentially."
    cp2.font.name = FONT_BODY
    cp2.font.size = Pt(15)
    cp2.font.color.rgb = COLOR_BG
    cp2.space_after = Pt(14)
    
    cp3 = ctf.add_paragraph()
    cp3.text = "Our Goal: Investigate how adaptive optimization and numerical regularization stabilize training and restore convergence."
    cp3.font.name = FONT_BODY
    cp3.font.size = Pt(15)
    cp3.font.italic = True
    cp3.font.color.rgb = COLOR_TEAL

    # ----------------------------------------------------
    # SLIDE 3: Phase 1: Inducing Ill-Conditioning
    # ----------------------------------------------------
    slide3 = add_standard_slide("Phase 1: Generating Ill-Conditioned Data")
    
    # Left Description
    text_box = slide3.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(5.8), Inches(4.8))
    tf = text_box.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "• Inducing Multicollinearity"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(4)
    
    p1_sub = tf.add_paragraph()
    p1_sub.text = "Generated 200 samples with 10 features. Artificially forced extreme correlations: feature 1 ≈ 0.95·feature 0, feature 3 ≈ 0.97·feature 2."
    p1_sub.font.name = FONT_BODY
    p1_sub.font.size = Pt(14)
    p1_sub.font.color.rgb = COLOR_CHARCOAL
    p1_sub.space_after = Pt(12)
    
    p2 = tf.add_paragraph()
    p2.text = "• Gram Matrix Instability"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_NAVY
    p2.space_after = Pt(4)
    
    p2_sub = tf.add_paragraph()
    p2_sub.text = "Gram matrix G = XᵀX exhibits an extremely wide eigenvalue spread spanning 4 orders of magnitude:\n  - Minimum eigenvalue: ~2.42×10⁻⁴\n  - Maximum eigenvalue: ~4.94\n  - Condition Number: κ ≈ 2.04×10⁴"
    p2_sub.font.name = FONT_BODY
    p2_sub.font.size = Pt(14)
    p2_sub.font.color.rgb = COLOR_CHARCOAL
    p2_sub.space_after = Pt(12)
    
    p3 = tf.add_paragraph()
    p3.text = "• Direct Inversion Breakdown"
    p3.font.name = FONT_TITLE
    p3.font.size = Pt(18)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_NAVY
    p3.space_after = Pt(4)
    
    p3_sub = tf.add_paragraph()
    p3_sub.text = "Direct computation of (XᵀX)⁻¹ fails numerically due to singularity. Solving requires SVD-based stable solvers (e.g., NumPy's lstsq)."
    p3_sub.font.name = FONT_BODY
    p3_sub.font.size = Pt(14)
    p3_sub.font.color.rgb = COLOR_CHARCOAL

    # Right Image
    img_path = os.path.join(BASE_DIR, 'eigenvalue_spectrum.png')
    if os.path.exists(img_path):
        slide3.shapes.add_picture(img_path, Inches(7.0), Inches(1.8), Inches(5.6), Inches(4.2))

    # ----------------------------------------------------
    # SLIDE 4: Phase 2: Vanilla SGD Baseline (Control)
    # ----------------------------------------------------
    slide4 = add_standard_slide("Phase 2: Vanilla SGD Control Baseline")
    
    # Left Description
    text_box = slide4.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(5.8), Inches(4.8))
    tf = text_box.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "• Control Setup"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(4)
    
    p1_sub = tf.add_paragraph()
    p1_sub.text = "Train a 3-layer feedforward neural network (10 → 64 → 64 → 1) with ReLU activations. Task is regression with MSE Loss."
    p1_sub.font.name = FONT_BODY
    p1_sub.font.size = Pt(14)
    p1_sub.font.color.rgb = COLOR_CHARCOAL
    p1_sub.space_after = Pt(12)
    
    p2 = tf.add_paragraph()
    p2.text = "• Pure Vanilla Optimization"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_NAVY
    p2.space_after = Pt(4)
    
    p2_sub = tf.add_paragraph()
    p2_sub.text = "Optimizer: SGD with lr = 0.001 and no momentum. Trained for 300 epochs."
    p2_sub.font.name = FONT_BODY
    p2_sub.font.size = Pt(14)
    p2_sub.font.color.rgb = COLOR_CHARCOAL
    p2_sub.space_after = Pt(12)
    
    p3 = tf.add_paragraph()
    p3.text = "• Severe Oscillations & Slow Convergence"
    p3.font.name = FONT_TITLE
    p3.font.size = Pt(18)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_NAVY
    p3.space_after = Pt(4)
    
    p3_sub = tf.add_paragraph()
    p3_sub.text = "Loss decreases very sluggishly from 0.6749 to 0.5198. Significant loss curve oscillations in early epochs demonstrate the optimizer bouncing off ravine walls."
    p3_sub.font.name = FONT_BODY
    p3_sub.font.size = Pt(14)
    p3_sub.font.color.rgb = COLOR_CHARCOAL

    # Right Image
    img_path = os.path.join(BASE_DIR, 'vanilla_sgd_loss.png')
    if os.path.exists(img_path):
        slide4.shapes.add_picture(img_path, Inches(7.0), Inches(1.8), Inches(5.6), Inches(4.2))

    # ----------------------------------------------------
    # SLIDE 5: Adaptive Stabilization: Theory
    # ----------------------------------------------------
    slide5 = add_standard_slide("Stabilization Techniques (Phases 3-5)")
    
    # 3 Column text layout for the 3 techniques
    col_width = Inches(3.6)
    col_gap = Inches(0.4)
    left_start = Inches(0.75)
    
    techniques = [
        {
            "num": "Phase 3",
            "name": "Ridge Regularization (L2)",
            "theory": "Modifies the optimization objective by adding an L2 penalty on weights.",
            "effect": "In linear terms, this shifts the Gram matrix to (XᵀX + λI). All eigenvalues shift up by λ, which dramatically shrinks the condition number to κ' = (λ_max + λ)/(λ_min + λ), smoothing the loss contours."
        },
        {
            "num": "Phase 4",
            "name": "Adam Optimizer",
            "theory": "Uses first (momentum) and second (RMSprop) raw moments of gradients.",
            "effect": "Adapts learning rate per parameter by dividing by the square root of the running average of squared gradients. Behaves as an approximate diagonal Hessian preconditioner, neutralizing ravine scaling."
        },
        {
            "num": "Phase 5",
            "name": "Batch Normalization",
            "theory": "Normalizes layer activations within each mini-batch.",
            "effect": "Keeps activation distributions centered with unit variance. Binds the singular values of the Jacobian, preventing gradient explosion/vanishing, smoothing loss surfaces, and allowing much higher learning rates."
        }
    ]
    
    for i, tech in enumerate(techniques):
        col_left = left_start + i * (col_width + col_gap)
        
        # Draw background panel
        panel = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col_left, Inches(1.8), col_width, Inches(4.8))
        panel.fill.solid()
        panel.fill.fore_color.rgb = COLOR_WHITE
        panel.line.color.rgb = COLOR_TEAL
        panel.line.width = Pt(1)
        
        box = slide5.shapes.add_textbox(col_left + Inches(0.15), Inches(1.95), col_width - Inches(0.3), Inches(4.5))
        tf = box.text_frame
        tf.word_wrap = True
        
        # Phase header
        p_num = tf.paragraphs[0]
        p_num.text = tech["num"]
        p_num.font.name = FONT_TITLE
        p_num.font.size = Pt(14)
        p_num.font.bold = True
        p_num.font.color.rgb = COLOR_TEAL
        p_num.space_after = Pt(2)
        
        # Tech Name
        p_name = tf.add_paragraph()
        p_name.text = tech["name"]
        p_name.font.name = FONT_TITLE
        p_name.font.size = Pt(18)
        p_name.font.bold = True
        p_name.font.color.rgb = COLOR_NAVY
        p_name.space_after = Pt(12)
        
        # Theory
        p_th = tf.add_paragraph()
        p_th.text = "How it works:"
        p_th.font.name = FONT_TITLE
        p_th.font.size = Pt(13)
        p_th.font.bold = True
        p_th.font.color.rgb = COLOR_CHARCOAL
        p_th.space_after = Pt(2)
        
        p_th_sub = tf.add_paragraph()
        p_th_sub.text = tech["theory"]
        p_th_sub.font.name = FONT_BODY
        p_th_sub.font.size = Pt(13)
        p_th_sub.font.color.rgb = COLOR_CHARCOAL
        p_th_sub.space_after = Pt(12)
        
        # Numerical Effect
        p_eff = tf.add_paragraph()
        p_eff.text = "Numerical Effect:"
        p_eff.font.name = FONT_TITLE
        p_eff.font.size = Pt(13)
        p_eff.font.bold = True
        p_eff.font.color.rgb = COLOR_CHARCOAL
        p_eff.space_after = Pt(2)
        
        p_eff_sub = tf.add_paragraph()
        p_eff_sub.text = tech["effect"]
        p_eff_sub.font.name = FONT_BODY
        p_eff_sub.font.size = Pt(12.5)
        p_eff_sub.font.color.rgb = COLOR_CHARCOAL

    # ----------------------------------------------------
    # SLIDE 6: Comparative Results
    # ----------------------------------------------------
    slide6 = add_standard_slide("Phase 3-5: Experimental Comparisons")
    
    # Left Description
    text_box = slide6.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(5.8), Inches(4.8))
    tf = text_box.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "• Ridge Regularization"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(16)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(2)
    
    p1_sub = tf.add_paragraph()
    p1_sub.text = "Slightly smoother convergence than Vanilla SGD, but still restricted to a very slow learning rate due to the large maximum eigenvalue. MSE stays high (~0.48)."
    p1_sub.font.name = FONT_BODY
    p1_sub.font.size = Pt(13)
    p1_sub.font.color.rgb = COLOR_CHARCOAL
    p1_sub.space_after = Pt(8)
    
    p2 = tf.add_paragraph()
    p2.text = "• Batch Normalization"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_NAVY
    p2.space_after = Pt(2)
    
    p2_sub = tf.add_paragraph()
    p2_sub.text = "At equivalent lr = 0.001, Batch Norm converges more reliably than Vanilla SGD. Crucially, it allows scaling the learning rate 10x higher (lr = 0.01) without exploding, achieving very fast and stable loss decrease."
    p2_sub.font.name = FONT_BODY
    p2_sub.font.size = Pt(13)
    p2_sub.font.color.rgb = COLOR_CHARCOAL
    p2_sub.space_after = Pt(8)
    
    p3 = tf.add_paragraph()
    p3.text = "• Adam Optimizer (Clear Winner)"
    p3.font.name = FONT_TITLE
    p3.font.size = Pt(16)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_NAVY
    p3.space_after = Pt(2)
    
    p3_sub = tf.add_paragraph()
    p3_sub.text = "Converges almost instantly, dropping loss from ~0.67 to ~0.005 in less than 20 epochs. Per-parameter step sizing efficiently resolves the ravines of the ill-conditioned landscape."
    p3_sub.font.name = FONT_BODY
    p3_sub.font.size = Pt(13)
    p3_sub.font.color.rgb = COLOR_CHARCOAL

    # Right Image
    img_path = os.path.join(BASE_DIR, 'comparison_loss_curves.png')
    if os.path.exists(img_path):
        slide6.shapes.add_picture(img_path, Inches(6.8), Inches(1.8), Inches(6.0), Inches(4.5))

    # ----------------------------------------------------
    # SLIDE 7: Summary & Conclusions
    # ----------------------------------------------------
    slide7 = add_standard_slide("Summary & Conclusions")
    
    text_box = slide7.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(11.83), Inches(4.8))
    tf = text_box.text_frame
    tf.word_wrap = True
    
    # Conclusion 1
    p1 = tf.paragraphs[0]
    p1.text = "1. Condition Number Dictates Optimization Success"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(4)
    
    p1_sub = tf.add_paragraph()
    p1_sub.text = "Problems in ML are often viewed from a model-capacity lens, but numerical stability and optimization speed are fundamentally constrained by the condition number of the data's Hessian matrix."
    p1_sub.font.name = FONT_BODY
    p1_sub.font.size = Pt(14)
    p1_sub.font.color.rgb = COLOR_CHARCOAL
    p1_sub.space_after = Pt(14)
    
    # Conclusion 2
    p2 = tf.add_paragraph()
    p2.text = "2. Adaptive Preconditioning (Adam) beats Pure Regularization (Ridge)"
    p2.font.name = FONT_TITLE
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_NAVY
    p2.space_after = Pt(4)
    
    p2_sub = tf.add_paragraph()
    p2_sub.text = "While L2 regularization shifts the eigenvalues away from zero and stabilizes the math, it does not normalize the relative gradients in steep vs. flat directions. Adam's coordinate-wise adaptive step sizes act as a dynamic diagonal preconditioner that directly targets and neutralizes this landscape distortion."
    p2_sub.font.name = FONT_BODY
    p2_sub.font.size = Pt(14)
    p2_sub.font.color.rgb = COLOR_CHARCOAL
    p2_sub.space_after = Pt(14)
    
    # Conclusion 3
    p3 = tf.add_paragraph()
    p3.text = "3. Batch Normalization Empowers Higher Learning Rates"
    p3.font.name = FONT_TITLE
    p3.font.size = Pt(18)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_NAVY
    p3.space_after = Pt(4)
    
    p3_sub = tf.add_paragraph()
    p3_sub.text = "Batch Norm's normalization step restricts the scale of activations. This bounds the Jacobian matrix singular values, which translates to a smooth and stable Hessian, allowing developers to safely scale up the learning rate by order of magnitude without risk of divergence."
    p3_sub.font.name = FONT_BODY
    p3_sub.font.size = Pt(14)
    p3_sub.font.color.rgb = COLOR_CHARCOAL

    # Save presentation to the parent workspace folder
    save_path = '/Users/kushagra/Desktop/Numerical Methods/adaptive_stabilization_progress.pptx'
    prs.save(save_path)
    print(f"Presentation saved successfully to: {save_path}")

if __name__ == '__main__':
    create_presentation()
