"""Export functionality for plots and results."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import io
import base64
from datetime import datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import plotly.graph_objects as go
import plotly.io as pio

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportRequest(BaseModel):
    """Request model for export."""
    format: str  # 'pdf', 'png', 'svg', 'csv', 'json'
    data_type: str  # 'plot', 'results', 'report'
    content: Dict[str, Any]
    filename: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class ReportGenerator:
    """Generate comprehensive PDF reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.styles = getSampleStyleSheet()
        self._add_custom_styles()
    
    def _add_custom_styles(self):
        """Add custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20
        ))
    
    def generate_report(
        self,
        results: Dict[str, Any],
        plots: Optional[List[Dict[str, Any]]] = None
    ) -> bytes:
        """
        Generate comprehensive PDF report.
        
        Args:
            results: Analysis results
            plots: Optional list of plot data
            
        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph(
            "Clinical Trial Statistical Analysis Report",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Metadata
        metadata = Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        )
        story.append(metadata)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeading']))
        summary_data = self._create_summary_table(results)
        story.append(summary_data)
        story.append(Spacer(1, 20))
        
        # Statistical Results
        story.append(Paragraph("Statistical Analysis Results", self.styles['SectionHeading']))
        
        # Main metrics
        metrics_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['P-value', f"{results.get('calculated_p_value', 'N/A'):.4f}", 
             self._interpret_p_value(results.get('calculated_p_value'))],
            ['Statistical Power', f"{results.get('calculated_power', 'N/A'):.2%}",
             self._interpret_power(results.get('calculated_power'))],
            ['Effect Size', f"{results.get('effect_size', 'N/A'):.3f}",
             self._interpret_effect_size(results.get('effect_size'))],
            ['Test Used', results.get('statistical_test_used', 'N/A'), ''],
            ['Sample Size', str(results.get('parameters', {}).get('n_total', 'N/A')), '']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Parameters
        story.append(Paragraph("Study Parameters", self.styles['SectionHeading']))
        params = results.get('parameters', {})
        param_text = "<br/>".join([f"• {k}: {v}" for k, v in params.items()])
        story.append(Paragraph(param_text, self.styles['InfoText']))
        story.append(Spacer(1, 20))
        
        # Add plots if provided
        if plots:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", self.styles['SectionHeading']))
            
            for plot_data in plots:
                if 'image' in plot_data:
                    # Add plot image
                    img = Image(io.BytesIO(base64.b64decode(plot_data['image'])),
                               width=6*inch, height=4*inch)
                    story.append(img)
                    
                    # Add caption if provided
                    if 'caption' in plot_data:
                        caption = Paragraph(plot_data['caption'], self.styles['Normal'])
                        story.append(caption)
                    
                    story.append(Spacer(1, 20))
        
        # Recommendations
        if 'recommendations' in results:
            story.append(Paragraph("Recommendations", self.styles['SectionHeading']))
            for rec in results['recommendations']:
                story.append(Paragraph(f"• {rec}", self.styles['InfoText']))
            story.append(Spacer(1, 20))
        
        # References
        if 'references' in results:
            story.append(PageBreak())
            story.append(Paragraph("References", self.styles['SectionHeading']))
            for i, ref in enumerate(results['references'], 1):
                ref_text = f"{i}. {ref}"
                story.append(Paragraph(ref_text, self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    def _create_summary_table(self, results: Dict[str, Any]) -> Table:
        """Create summary table for report."""
        summary_data = [
            ['Study Type', results.get('suggested_study_type', 'N/A')],
            ['Statistical Test', results.get('statistical_test_used', 'N/A')],
            ['Primary Outcome', 
             'Statistically Significant' if results.get('calculated_p_value', 1) < 0.05 
             else 'Not Statistically Significant'],
            ['Confidence Level', f"{results.get('confidence_level', 0.95):.0%}"]
        ]
        
        table = Table(summary_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table
    
    def _interpret_p_value(self, p_value: Optional[float]) -> str:
        """Interpret p-value."""
        if p_value is None:
            return "N/A"
        elif p_value < 0.001:
            return "Very strong evidence against null"
        elif p_value < 0.01:
            return "Strong evidence against null"
        elif p_value < 0.05:
            return "Moderate evidence against null"
        elif p_value < 0.10:
            return "Weak evidence against null"
        else:
            return "No significant evidence"
    
    def _interpret_power(self, power: Optional[float]) -> str:
        """Interpret statistical power."""
        if power is None:
            return "N/A"
        elif power >= 0.9:
            return "Excellent"
        elif power >= 0.8:
            return "Good"
        elif power >= 0.7:
            return "Marginal"
        else:
            return "Insufficient"
    
    def _interpret_effect_size(self, effect: Optional[float]) -> str:
        """Interpret effect size."""
        if effect is None:
            return "N/A"
        
        abs_effect = abs(effect)
        if abs_effect < 0.2:
            return "Negligible"
        elif abs_effect < 0.5:
            return "Small"
        elif abs_effect < 0.8:
            return "Medium"
        else:
            return "Large"


@router.post("/plot")
async def export_plot(request: ExportRequest) -> Response:
    """
    Export plot in various formats.
    
    Supports: PNG, SVG, PDF, JSON
    """
    try:
        plot_data = request.content.get('plot_data')
        if not plot_data:
            raise HTTPException(status_code=400, detail="No plot data provided")
        
        # Recreate Plotly figure from JSON
        fig = go.Figure(json.loads(plot_data) if isinstance(plot_data, str) else plot_data)
        
        # Set filename
        filename = request.filename or f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if request.format.lower() == 'png':
            # Export as PNG
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            return Response(
                content=img_bytes,
                media_type="image/png",
                headers={"Content-Disposition": f"attachment; filename={filename}.png"}
            )
        
        elif request.format.lower() == 'svg':
            # Export as SVG
            svg_str = pio.to_image(fig, format='svg', width=1200, height=800)
            return Response(
                content=svg_str,
                media_type="image/svg+xml",
                headers={"Content-Disposition": f"attachment; filename={filename}.svg"}
            )
        
        elif request.format.lower() == 'pdf':
            # Export as PDF (via reportlab)
            buffer = io.BytesIO()
            
            # Convert plot to image first
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            
            # Create PDF with embedded image
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # Add title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, 750, request.content.get('title', 'Statistical Analysis Plot'))
            
            # Add image
            from reportlab.lib.utils import ImageReader
            img = ImageReader(io.BytesIO(img_bytes))
            c.drawImage(img, 50, 200, width=500, height=333)
            
            # Add timestamp
            c.setFont("Helvetica", 10)
            c.drawString(100, 150, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            c.save()
            buffer.seek(0)
            
            return Response(
                content=buffer.read(),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}.pdf"}
            )
        
        elif request.format.lower() == 'json':
            # Export as JSON (Plotly format)
            json_str = fig.to_json()
            return Response(
                content=json_str,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}.json"}
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/results")
async def export_results(request: ExportRequest) -> Response:
    """
    Export analysis results in various formats.
    
    Supports: CSV, JSON, PDF
    """
    try:
        results = request.content
        filename = request.filename or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if request.format.lower() == 'csv':
            # Convert results to CSV
            df_data = []
            
            # Main results
            df_data.append({
                'Metric': 'P-value',
                'Value': results.get('calculated_p_value', 'N/A')
            })
            df_data.append({
                'Metric': 'Statistical Power',
                'Value': results.get('calculated_power', 'N/A')
            })
            df_data.append({
                'Metric': 'Effect Size',
                'Value': results.get('effect_size', 'N/A')
            })
            df_data.append({
                'Metric': 'Test Used',
                'Value': results.get('statistical_test_used', 'N/A')
            })
            
            # Add parameters
            for key, value in results.get('parameters', {}).items():
                df_data.append({
                    'Metric': f'Parameter: {key}',
                    'Value': value
                })
            
            df = pd.DataFrame(df_data)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            return Response(
                content=csv_buffer.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
            )
        
        elif request.format.lower() == 'json':
            # Export as JSON
            json_str = json.dumps(results, indent=2)
            return Response(
                content=json_str,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}.json"}
            )
        
        elif request.format.lower() == 'pdf':
            # Generate PDF report
            generator = ReportGenerator()
            pdf_bytes = generator.generate_report(results, request.content.get('plots'))
            
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}.pdf"}
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def export_comprehensive_report(request: ExportRequest) -> Response:
    """
    Export comprehensive analysis report including all results and visualizations.
    
    Always exports as PDF.
    """
    try:
        generator = ReportGenerator()
        
        # Combine all content
        full_content = {
            **request.content.get('results', {}),
            'plots': request.content.get('plots', []),
            'recommendations': request.content.get('recommendations', []),
            'references': request.content.get('references', [])
        }
        
        pdf_bytes = generator.generate_report(full_content)
        
        filename = request.filename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats")
async def get_export_formats() -> Dict[str, List[str]]:
    """Get available export formats for each data type."""
    return {
        "plot": ["png", "svg", "pdf", "json"],
        "results": ["csv", "json", "pdf"],
        "report": ["pdf"]
    }