"""
PDF Generation Utility (utils/pdf_generator.py)
================================================
This utility creates professional, boardroom-ready PDF reports
summarizing the flood damage assessment using ReportLab.
"""
import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import qrcode
from PIL import Image as PILImage

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='CenterTitle', alignment=1, fontSize=24, spaceAfter=20, textColor=colors.darkblue))
        self.styles.add(ParagraphStyle(name='SubTitle', fontSize=14, spaceAfter=10, textColor=colors.dimgrey))
        self.styles.add(ParagraphStyle(name='SectionHeader', fontSize=16, spaceAfter=10, textColor=colors.darkblue, spaceBefore=15))
        self.styles.add(ParagraphStyle(name='NormalText', fontSize=12, spaceAfter=8))
        self.styles.add(ParagraphStyle(name='AlertText', fontSize=12, textColor=colors.red, spaceAfter=8))

    def generate_report(self, original_img_path: str, mask_img_path: str, data: dict) -> bytes:
        """
        Builds the PDF document in memory and returns it as raw bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        
        elements = []
        
        # 1. Header & Title
        elements.append(Paragraph("🌊 Flood Damage Assessment Report", self.styles['CenterTitle']))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        elements.append(Paragraph(f"Generated at: {timestamp}", self.styles['SubTitle']))
        elements.append(Spacer(1, 0.2 * inch))

        # 2. Executive Summary (Statistics)
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        status = "FLOOD DETECTED" if data.get('flood') else "NO FLOOD DETECTED"
        status_style = self.styles['AlertText'] if data.get('flood') else self.styles['NormalText']
        elements.append(Paragraph(f"<b>Status:</b> {status}", status_style))
        elements.append(Paragraph(f"<b>Severity Level:</b> {data.get('severity', 'Unknown')}", self.styles['NormalText']))
        elements.append(Paragraph(f"<b>AI Confidence:</b> {data.get('confidence', 0)}%", self.styles['NormalText']))
        elements.append(Paragraph(f"<b>Flood Coverage Area:</b> {data.get('flood_percentage', 0)}%", self.styles['NormalText']))
        
        objects = data.get('objects', [])
        obj_str = ", ".join(objects) if objects else "None detected"
        elements.append(Paragraph(f"<b>Impacted Infrastructure Detected:</b> {obj_str}", self.styles['NormalText']))
        
        elements.append(Spacer(1, 0.3 * inch))

        # 3. Visual Imagery (Original vs Processed)
        elements.append(Paragraph("Imagery Analysis", self.styles['SectionHeader']))
        
        # Create a 1x2 table to display images side-by-side
        try:
            # We constrain the images to 3 inches wide to fit on a letter page
            img1 = RLImage(original_img_path, width=3*inch, height=2.5*inch)
            img2 = RLImage(mask_img_path, width=3*inch, height=2.5*inch)
            img_table = Table([[img1, img2]], colWidths=[3.2*inch, 3.2*inch])
            
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(img_table)
            
            label_table = Table([["Original Satellite/Drone View", "AI Water Mask & YOLO Detections"]], colWidths=[3.2*inch, 3.2*inch])
            label_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.dimgrey),
                ('FONTSIZE', (0,0), (-1,-1), 10),
            ]))
            elements.append(label_table)
        except Exception as e:
            elements.append(Paragraph(f"(Image rendering failed: {e})", self.styles['AlertText']))
            
        elements.append(Spacer(1, 0.4 * inch))

        # 4. Recommendations
        elements.append(Paragraph("Actionable Recommendations", self.styles['SectionHeader']))
        severity = data.get('severity', 'Low')
        
        if severity in ['High', 'Severe']:
            elements.append(Paragraph("• 🚨 IMMEDIATE ACTION REQUIRED. Deploy rescue teams to highlighted zones.", self.styles['AlertText']))
            elements.append(Paragraph("• Initiate evacuation protocols for submerged residential sectors.", self.styles['NormalText']))
        
        if "Vehicles" in objects:
            elements.append(Paragraph("• Amphibious recovery vehicles required for stranded transport.", self.styles['NormalText']))
        if "Roads" in objects:
            elements.append(Paragraph("• Access routes compromised. Identify alternative supply corridors.", self.styles['NormalText']))
            
        if not data.get('flood'):
            elements.append(Paragraph("• No immediate threat detected. Maintain standard monitoring intervals.", self.styles['NormalText']))

        # 5. QR Code linking to dashboard/report
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph("Digital Verification", self.styles['SectionHeader']))
        
        try:
            # Generate a QR Code pointing to a hypothetical dashboard URL
            qr = qrcode.QRCode(version=1, box_size=3, border=2)
            qr.add_data(f"https://flood-assessment.app/reports/{data.get('id', 'demo')}")
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Save QR to a temporary buffer and inject it into the PDF
            qr_buffer = io.BytesIO()
            qr_img.save(qr_buffer, format="PNG")
            qr_buffer.seek(0)
            
            rl_qr = RLImage(qr_buffer, width=1.5*inch, height=1.5*inch)
            elements.append(rl_qr)
            elements.append(Paragraph("Scan to view live interactive dashboard.", self.styles['SubTitle']))
        except Exception as e:
            pass

        # Build PDF
        doc.build(elements)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

pdf_builder = PDFReportGenerator()
