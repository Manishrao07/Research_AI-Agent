from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime
import os
import re

# ── Colors ──────────────────────────────────────────────
DARK_NAVY   = HexColor("#0F172A")
BLUE        = HexColor("#2563EB")
LIGHT_BLUE  = HexColor("#DBEAFE")
STEEL       = HexColor("#475569")
LIGHT_BG    = HexColor("#F8FAFC")
BORDER      = HexColor("#E2E8F0")
TEXT_DARK   = HexColor("#0F172A")
TEXT_BODY   = HexColor("#334155")
TEXT_MUTED  = HexColor("#94A3B8")
WHITE       = HexColor("#FFFFFF")
GREEN       = HexColor("#059669")

def strip_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

def make_styles():
    base = getSampleStyleSheet()

    heading_main = ParagraphStyle('HeadingMain',
        fontName='Helvetica-Bold', fontSize=28,
        textColor=WHITE, alignment=TA_CENTER,
        spaceAfter=0, spaceBefore=0, leading=34)

    heading_sub = ParagraphStyle('HeadingSub',
        fontName='Helvetica', fontSize=11,
        textColor=HexColor("#93C5FD"), alignment=TA_CENTER,
        spaceAfter=0, spaceBefore=0, leading=16)

    report_title = ParagraphStyle('ReportTitle',
        fontName='Helvetica-Bold', fontSize=18,
        textColor=TEXT_DARK, alignment=TA_CENTER,
        spaceBefore=14, spaceAfter=4, leading=24)

    meta = ParagraphStyle('Meta',
        fontName='Helvetica', fontSize=9,
        textColor=TEXT_MUTED, alignment=TA_CENTER,
        spaceBefore=0, spaceAfter=0, leading=14)

    pipeline = ParagraphStyle('Pipeline',
        fontName='Helvetica', fontSize=9,
        textColor=BLUE, leading=14)

    h2 = ParagraphStyle('H2',
        fontName='Helvetica-Bold', fontSize=13,
        textColor=BLUE, spaceBefore=18,
        spaceAfter=6, leading=18, bulletText=None)

    h3 = ParagraphStyle('H3',
        fontName='Helvetica-Bold', fontSize=11,
        textColor=TEXT_DARK, spaceBefore=12,
        spaceAfter=4, leading=16, bulletText=None)

    body = ParagraphStyle('Body',
        fontName='Helvetica', fontSize=10,
        textColor=TEXT_BODY, spaceAfter=6,
        leading=16, alignment=TA_JUSTIFY)

    bullet = ParagraphStyle('Bullet',
        fontName='Helvetica', fontSize=10,
        textColor=TEXT_BODY, spaceAfter=4,
        leading=15, leftIndent=14, bulletText=None)

    footer = ParagraphStyle('Footer',
        fontName='Helvetica', fontSize=8,
        textColor=TEXT_MUTED, alignment=TA_CENTER, leading=12)

    return dict(heading_main=heading_main, heading_sub=heading_sub,
                report_title=report_title, meta=meta, pipeline=pipeline,
                h2=h2, h3=h3, body=body, bullet=bullet, footer=footer)


def create_pdf_report(topic: str, report_text: str, steps: list, output_path: str = None):
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in topic if c.isalnum() or c in ' -_')[:28].strip()
        output_path = f"reports/{safe}_{ts}.pdf"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.1*cm, rightMargin=1.1*cm,
        topMargin=1.1*cm, bottomMargin=1.1*cm
    )

    S = make_styles()
    story = []
    W = 18.8 * cm  # usable width

    # ── Header block ────────────────────────────────────
    header_inner = [
        [Paragraph("ResearchAI", S['heading_main'])],
        [Paragraph("Autonomous Research Agent", S['heading_sub'])],
    ]
    header_tbl = Table(header_inner, colWidths=[W])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), DARK_NAVY),
        ('TOPPADDING',    (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 18),
        ('LEFTPADDING',   (0,0), (-1,-1), 20),
        ('RIGHTPADDING',  (0,0), (-1,-1), 20),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [DARK_NAVY, DARK_NAVY]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 10))

    # ── Topic title ─────────────────────────────────────
    clean_topic = strip_emojis(topic)
    story.append(Paragraph(clean_topic, S['report_title']))

    now = datetime.now().strftime("%d %B %Y  |  %I:%M %p")
    story.append(Paragraph(f"{now}  |  {len(steps)} research steps", S['meta']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width=W, thickness=2, color=BLUE, spaceAfter=10))

    # ── Pipeline box ────────────────────────────────────
    if steps:
        clean_steps = []
        for s in steps:
            s = strip_emojis(s)
            if "|" in s:
                parts = s.split("|")
                if parts[0] in ("TOOL_CALL", "TOOL_RESULT"):
                    clean_steps.append(parts[1])
                elif parts[0] == "COMPARE_PHASE":
                    clean_steps.append(parts[1])
                elif parts[0] == "COMPARE_START":
                    clean_steps.append(f"Comparing {parts[1]} vs {parts[2]}")
            else:
                clean_steps.append(s.replace("Tool used:", "").strip())
        # Duplicates hatao (TOOL_CALL aur TOOL_RESULT same naam dete hain)
        seen = set()
        unique_steps = []
        for s in clean_steps:
            if s not in seen:
                seen.add(s)
                unique_steps.append(s)
        pipe_text = "  →  ".join(unique_steps)
        pipe_row = [[
            Paragraph("<b>Research pipeline:</b>", S['pipeline']),
            Paragraph(pipe_text, S['pipeline']),
        ]]
        pipe_tbl = Table(pipe_row, colWidths=[3.5*cm, W - 3.5*cm])
        pipe_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_BLUE),
            ('TOPPADDING',    (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 9),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('RIGHTPADDING',  (0,0), (-1,-1), 12),
            ('BOX', (0,0), (-1,-1), 1, BLUE),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(pipe_tbl)
        story.append(Spacer(1, 16))

    # ── Report body ─────────────────────────────────────
    # ── Report body ─────────────────────────────────────
    lines = report_text.split('\n')
    table_rows = []
    in_table = False

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows:
            n_cols = len(table_rows[0])
            col_width = W / n_cols
            cell_style = ParagraphStyle('Cell', fontName='Helvetica', fontSize=8.5,
                                         textColor=TEXT_BODY, leading=11)
            header_style = ParagraphStyle('CellHead', fontName='Helvetica-Bold', fontSize=8.5,
                                           textColor=WHITE, leading=11)
            wrapped = []
            for i, row in enumerate(table_rows):
                style = header_style if i == 0 else cell_style
                wrapped.append([Paragraph(cell, style) for cell in row])
            tbl = Table(wrapped, colWidths=[col_width]*n_cols)
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), BLUE),
                ('BACKGROUND', (0,1), (-1,-1), LIGHT_BG),
                ('GRID', (0,0), (-1,-1), 0.5, BORDER),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 10))
        table_rows = []
        in_table = False

    for raw_line in lines:
        line = strip_emojis(raw_line).strip()

        # Markdown table row
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            # Separator row (|---|---|) skip karo
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                continue
            cells = [re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', c) for c in cells]
            table_rows.append(cells)
            in_table = True
            continue
        else:
            if in_table:
                flush_table()

        if not line:
            story.append(Spacer(1, 5))
            continue

        # H2
        if line.startswith('## '):
            text = line[3:].strip()
            story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceAfter=4))
            story.append(Paragraph(text, S['h2']))

        # H3
        elif line.startswith('### '):
            text = line[4:].strip()
            story.append(Paragraph(text, S['h3']))

        # Bullet
        elif line.startswith('* ') or line.startswith('- '):
            text = line[2:].strip()
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 — \2', text)
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{text}", S['bullet']))

        # Normal
        else:
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 — \2', text)
            story.append(Paragraph(text, S['body']))

    flush_table()

    # ── Footer ──────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width=W, thickness=1, color=BORDER, spaceAfter=6))
    story.append(Paragraph(
        "Generated by ResearchAI Agent  |  LangGraph · Groq LLaMA 3.3 · Tavily · ReportLab",
        S['footer']
    ))

    doc.build(story)
    print(f"PDF saved: {output_path}")
    return output_path


# ── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """## Research Report: Artificial Intelligence in 2025

### Executive Summary
AI is transforming industries at an unprecedented pace in 2025, with generative models leading adoption across sectors.

### Key Findings
* AI adoption has increased by 40% since 2023 across Fortune 500 companies
* **Generative AI** is the fastest growing technology segment globally
* Healthcare and finance are leading enterprise AI adoption
* Over 50 million jobs expected to be impacted by automation

### Latest Developments
Major tech companies including Google, Microsoft, and OpenAI are investing over $100 billion in AI infrastructure in 2025 alone.

### Background & History
Artificial Intelligence research began in the 1950s at Dartmouth College. The field has seen exponential growth since the introduction of transformer architecture in 2017.

### Analysis & Insights
The convergence of large language models with agentic capabilities represents a fundamental paradigm shift in how software is built and deployed.

### Future Outlook
By 2026, AI agents are projected to handle 30% of enterprise workflows autonomously, reducing operational costs by an estimated 25%.

### Sources
* MIT Technology Review — https://technologyreview.com
* World Economic Forum — https://weforum.org
* McKinsey Global Institute — https://mckinsey.com"""

    path = create_pdf_report(
        topic="Artificial Intelligence in 2025",
        report_text=sample,
        steps=["Tool used: search_web", "Tool used: search_wikipedia"],
    )
    print(f"Done: {path}")