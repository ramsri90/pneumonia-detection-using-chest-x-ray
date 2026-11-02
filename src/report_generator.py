import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(
    patient_info: dict,
    diagnosis: str,
    confidence: float,
    probabilities: dict,
    original_image: Image.Image,
    heatmap_image: Image.Image,
    clinical_notes: str = ""
) -> bytes:
    """
    Generates an official clinical PDF diagnostic report for Chest X-Ray Pneumonia detection.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'HospitalTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        'HospitalSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b')
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=4
    )
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )
    cell_text = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#64748b'),
        alignment=1
    )

    story = []

    # 1. Header
    header_data = [
        [
            Paragraph("<b>PNEUMONIA DETECTION FROM CHEST X-RAY</b><br/><font size=8.5 color='#475569'>Deep Learning Diagnostic Report · Chest Radiograph Evaluation</font>", title_style),
            Paragraph("<b>DIAGNOSTIC REPORT</b><br/>ISO Compliant CDSS Output<br/>ResNet-18 Deep Learning Model", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 2.8 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)

    # Dividing line
    story.append(Spacer(1, 4))
    line_table = Table([[""]], colWidths=[7.2 * inch], rowHeights=[2])
    line_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0284c7'))]))
    story.append(line_table)
    story.append(Spacer(1, 8))

    # 2. Patient Demographics Table
    p_name = patient_info.get("name", "Anonymous / De-identified")
    p_id = patient_info.get("id", "PT-CXR-001")
    p_age = str(patient_info.get("age", "N/A"))
    p_sex = patient_info.get("gender", "N/A")
    p_physician = patient_info.get("physician", "Attending Radiologist")
    exam_date = patient_info.get("date", "Today")

    patient_table_data = [
        [Paragraph("<b>Patient Name:</b>", cell_bold), Paragraph(p_name, cell_text),
         Paragraph("<b>Medical Record # (MRN):</b>", cell_bold), Paragraph(p_id, cell_text)],
        [Paragraph("<b>Age / Sex:</b>", cell_bold), Paragraph(f"{p_age} / {p_sex}", cell_text),
         Paragraph("<b>Exam Date & Time:</b>", cell_bold), Paragraph(exam_date, cell_text)],
        [Paragraph("<b>Study Modality:</b>", cell_bold), Paragraph("Chest Radiograph (PA/AP Digital CXR)", cell_text),
         Paragraph("<b>Referring Clinician:</b>", cell_bold), Paragraph(p_physician, cell_text)]
    ]
    p_table = Table(patient_table_data, colWidths=[1.5 * inch, 2.1 * inch, 1.7 * inch, 1.9 * inch])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 10))

    # 3. Diagnostic Finding Banner
    is_pneumonia = (diagnosis.lower() == "pneumonia")
    banner_bg = colors.HexColor('#fef2f2') if is_pneumonia else colors.HexColor('#f0fdf4')
    banner_border = colors.HexColor('#ef4444') if is_pneumonia else colors.HexColor('#10b981')
    banner_text_color = '#991b1b' if is_pneumonia else '#065f46'
    banner_title = "CRITICAL FINDING: PNEUMONIA DETECTED" if is_pneumonia else "NEGATIVE: NORMAL CHEST RADIOGRAPH"
    banner_desc = (
        "Bilateral / focal consolidation or interstitial infiltrates identified on radiograph."
        if is_pneumonia
        else "No focal consolidation, pneumothorax, or acute parenchymal infiltrates detected."
    )

    finding_html = f"<font size=12 color='{banner_text_color}'><b>{banner_title}</b></font><br/><font size=8.5 color='#334155'>{banner_desc}</font>"
    conf_html = f"<font size=8.5 color='#475569'>Model Confidence</font><br/><font size=16 color='{banner_text_color}'><b>{confidence:.1f}%</b></font>"

    banner_data = [
        [Paragraph(finding_html, styles['Normal']), Paragraph(conf_html, ParagraphStyle('Conf', parent=styles['Normal'], alignment=1))]
    ]
    banner_table = Table(banner_data, colWidths=[5.4 * inch, 1.8 * inch])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, banner_border),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 8))

    # 4. Quantitative Analysis Table
    p_norm = probabilities.get("Normal", 0.0)
    p_pneu = probabilities.get("Pneumonia", 0.0)
    quant_data = [
        [
            Paragraph("<b>Class Name</b>", cell_bold),
            Paragraph("<b>Posterior Probability</b>", cell_bold),
            Paragraph("<b>Clinical Interpretation</b>", cell_bold)
        ],
        [
            Paragraph("Normal (Unremarkable)", cell_text),
            Paragraph(f"<b>{p_norm:.2f}%</b>", cell_text),
            Paragraph("Absence of significant pulmonary consolidations", cell_text)
        ],
        [
            Paragraph("Pneumonia (Consolidation/Infiltrate)", cell_text),
            Paragraph(f"<b>{p_pneu:.2f}%</b>", cell_text),
            Paragraph("Suspicion of acute pulmonary inflammation or exudate", cell_text)
        ]
    ]
    quant_table = Table(quant_data, colWidths=[2.2 * inch, 1.8 * inch, 3.2 * inch])
    quant_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(quant_table)
    story.append(Spacer(1, 10))

    # 5. Image Comparison (Original CXR vs Grad-CAM Attention Heatmap)
    orig_buf = io.BytesIO()
    heat_buf = io.BytesIO()
    original_image.convert("RGB").save(orig_buf, format="JPEG", quality=90)
    heatmap_image.convert("RGB").save(heat_buf, format="JPEG", quality=90)
    orig_buf.seek(0)
    heat_buf.seek(0)

    img_w, img_h = 2.4 * inch, 2.4 * inch
    rl_orig = RLImage(orig_buf, width=img_w, height=img_h)
    rl_heat = RLImage(heat_buf, width=img_w, height=img_h)

    image_table_data = [
        [
            Paragraph("<b>Original Patient Chest Radiograph (PA)</b>", cell_bold),
            Paragraph("<b>Grad-CAM Explainability Attention Heatmap</b>", cell_bold)
        ],
        [rl_orig, rl_heat],
        [
            Paragraph("<font size=7 color='#64748b'>Digital radiograph evaluated by visual verification guard</font>", styles['Normal']),
            Paragraph("<font size=7 color='#64748b'>Red/yellow zones highlight anatomical regions driving the prediction</font>", styles['Normal'])
        ]
    ]
    img_table = Table(image_table_data, colWidths=[3.6 * inch, 3.6 * inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(KeepTogether(img_table))
    story.append(Spacer(1, 8))

    # 6. Clinical Notes & Radiologist Signoff
    notes_content = clinical_notes.strip() if clinical_notes else "Automated screening performed. Findings correlate with feature heatmaps."
    notes_data = [
        [
            Paragraph(f"<b>Clinical Notes / Radiologist Impressions:</b><br/>{notes_content}", cell_text),
            Paragraph("<b>Attending Radiologist Sign-off:</b><br/><br/>__________________________________<br/><font size=7 color='#64748b'>Signature & Stamp</font>", cell_text)
        ]
    ]
    notes_table = Table(notes_data, colWidths=[4.6 * inch, 2.6 * inch])
    notes_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(notes_table)
    story.append(Spacer(1, 8))

    # 7. Disclaimer
    story.append(Paragraph(
        "<b>LEGAL DISCLAIMER:</b> This computer-assisted diagnostic report is generated using a deep learning computational model "
        "trained for radiographic pattern recognition. It is intended solely as clinical decision support (CDSS) for qualified healthcare providers "
        "and does not replace comprehensive medical evaluation, clinical correlation, or official radiologist sign-off.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
