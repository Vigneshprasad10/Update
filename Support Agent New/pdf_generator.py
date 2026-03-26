"""
PDF Report Generator for Support AI Agent.

Generates a professional root cause analysis PDF report
after a support ticket has been analyzed and created.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Font Registration
# ─────────────────────────────────────────────
def _register_unicode_fonts():
    """Register Unicode-compatible fonts. Falls back to default if unavailable."""
    try:
        # Try common system font paths for DejaVuSans
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
        ]
        
        fonts_found = []
        for path in font_paths:
            if os.path.exists(path):
                fonts_found.append(path)
        
        if fonts_found:
            if len(fonts_found) >= 1:
                pdfmetrics.registerFont(TTFont("UniFont", fonts_found[0]))
                pdfmetrics.registerFont(TTFont("UniFont-Bold", fonts_found[0]))
            logger.info(f"[PDF] Registered Unicode fonts from: {fonts_found}")
        else:
            logger.warning("[PDF] No Unicode fonts found, using built-in fonts")
    except Exception as e:
        logger.warning(f"[PDF] Font registration failed: {e}, using built-in fonts")


_register_unicode_fonts()

# Font selection (fallback to Helvetica if custom fonts unavailable)
FONT_NORMAL = "UniFont"
FONT_BOLD = "UniFont-Bold"
FONT_MONO = "Courier"

# Try to use registered fonts, fallback to Helvetica
try:
    pdfmetrics.getFont(FONT_NORMAL)
except:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"

# ─────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────
NAVY       = HexColor("#1E2761")
ACCENT     = HexColor("#4A90D9")
ICE_BLUE   = HexColor("#CADCFC")
LIGHT_BG   = HexColor("#EEF2FF")
DARK_BG    = HexColor("#0D1B4B")
GREEN      = HexColor("#10B981")
ORANGE     = HexColor("#F59E0B")
RED        = HexColor("#EF4444")
GRAY       = HexColor("#64748B")
LIGHT_GRAY = HexColor("#F1F5F9")
WHITE      = HexColor("#FFFFFF")


def generate_pdf_report(state: dict, output_dir: str = "reports") -> str:
    """
    Generate a PDF root cause analysis report from agent state.

    Args:
        state: Final SupportAgentState from the agent
        output_dir: Directory to save the PDF

    Returns:
        Path to the generated PDF file
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = state.get("ticket_summary", "report")[:30].replace(" ", "_").replace("/", "-")
    filename = f"RCA_{summary}_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Extract data from state
    resolution     = state.get("resolution") or {}
    jira_ctx       = state.get("jira_context") or {}
    github_ctx     = state.get("github_context") or {}
    confluence_ctx = state.get("confluence_context") or {}
    enriched       = state.get("enriched_ticket") or {}

    # Build document
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Custom Styles ─────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontSize=22,
        fontName=FONT_BOLD,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        fontName=FONT_NORMAL,
        textColor=ICE_BLUE,
        alignment=TA_LEFT,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=13,
        fontName=FONT_BOLD,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        fontName=FONT_NORMAL,
        textColor=HexColor("#1E293B"),
        spaceAfter=4,
        leading=16,
    )
    mono_style = ParagraphStyle(
        "Mono",
        parent=styles["Normal"],
        fontSize=9,
        fontName=FONT_MONO,
        textColor=HexColor("#334155"),
        spaceAfter=3,
        leading=14,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=9,
        fontName=FONT_BOLD,
        textColor=GRAY,
        spaceAfter=2,
    )

    # ── HEADER ────────────────────────────────────────────
    header_data = [
        [
            Paragraph("ROOT CAUSE ANALYSIS", title_style),
            Paragraph(
                f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
                ParagraphStyle("Right", parent=styles["Normal"], fontSize=6,
                               textColor=ICE_BLUE, alignment=TA_RIGHT)
            )
        ],
        [
            Paragraph("Servion Support AI Agent  ·  Powered by LangGraph + Gemini + Confluence + GitHub", subtitle_style),
            Paragraph("", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.5 * inch, 2.5 * inch], rowHeights=[0.6*inch, 0.4*inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("PADDING",    (0, 0), (-1, 0), 14),
        ("PADDING",    (0, 1), (-1, 1), 12),
        ("TOPPADDING", (0, 1), (-1, 1), 8),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [NAVY, NAVY]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 16))

    # ── TICKET DETAILS ────────────────────────────────────
    story.append(Paragraph(" Ticket Details", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8))

    ticket_data = [
        ["Summary",     state.get("ticket_summary", "N/A")],
        ["Priority",    state.get("priority", "N/A").upper()],
        ["Version",     state.get("product_version", "N/A")],
        ["Environment", state.get("environment", "N/A")],
        ["Jira Ticket", jira_ctx.get("ticket_id", "Not created")],
        ["Analyzed At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]

    ticket_table = Table(
        [[Paragraph(k, label_style), Paragraph(str(v), body_style)] for k, v in ticket_data],
        colWidths=[1.4 * inch, 5.6 * inch]
    )
    ticket_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), LIGHT_BG),
        ("BACKGROUND",  (1, 0), (1, -1), WHITE),
        ("GRID",        (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
        ("PADDING",     (0, 0), (-1, -1), 7),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [WHITE, LIGHT_GRAY]),
    ]))
    story.append(ticket_table)
    story.append(Spacer(1, 12))

    # ── DESCRIPTION ───────────────────────────────────────
    desc = state.get("ticket_description", "")
    if desc:
        story.append(Paragraph(" Customer Report", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8))
        desc_table = Table(
            [[Paragraph(desc[:1000], body_style)]],
            colWidths=[7 * inch]
        )
        desc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("PADDING",    (0, 0), (-1, -1), 10),
            ("GRID",       (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
        ]))
        story.append(desc_table)
        story.append(Spacer(1, 12))

    # ── ROOT CAUSE ────────────────────────────────────────
    story.append(Paragraph(" Root Cause Analysis", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=8))

    confidence = int(resolution.get("confidence_score", 0) * 100)
    conf_color = GREEN if confidence >= 70 else ORANGE if confidence >= 50 else RED

    # Confidence badge
    conf_data = [[
        Paragraph("CONFIDENCE SCORE", label_style),
        Paragraph(f"{confidence}%", ParagraphStyle(
            "Conf", parent=styles["Normal"], fontSize=10,
            fontName=FONT_BOLD, textColor=conf_color, alignment=TA_CENTER
        )),
        Paragraph(
            "✓ HIGH" if confidence >= 70 else "⚠ MEDIUM" if confidence >= 50 else "✗ LOW",
            ParagraphStyle("ConfLabel", parent=styles["Normal"], fontSize=10,
                           fontName=FONT_BOLD, textColor=conf_color, alignment=TA_CENTER)
        )
    ]]
    conf_table = Table(conf_data, colWidths=[2 * inch, 2.5 * inch, 2.5 * inch])
    conf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("PADDING",    (0, 0), (-1, -1), 10),
        ("GRID",       (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 8))

    # Root cause text
    root_cause = resolution.get("root_cause_hypothesis", "Under investigation")
    rc_table = Table(
        [[Paragraph(root_cause, body_style)]],
        colWidths=[7 * inch]
    )
    rc_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), HexColor("#FFF7ED")),
        ("PADDING",      (0, 0), (-1, -1), 12),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("GRID",         (0, 0), (-1, -1), 0.5, ORANGE),
        ("LINEAFTER",    (0, 0), (0, -1), 4, ORANGE),
    ]))
    story.append(rc_table)
    story.append(Spacer(1, 8))

    # Workaround
    workaround = resolution.get("workaround")
    if workaround and workaround != "null":
        wk_table = Table(
            [[Paragraph(f"⚡ Workaround: {workaround}", body_style)]],
            colWidths=[7 * inch]
        )
        wk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FFFBEB")),
            ("PADDING",    (0, 0), (-1, -1), 10),
            ("GRID",       (0, 0), (-1, -1), 0.5, HexColor("#FCD34D")),
        ]))
        story.append(wk_table)
        story.append(Spacer(1, 8))

    # Escalation
    if resolution.get("escalation_needed"):
        esc_table = Table(
            [[Paragraph(f" ESCALATION REQUIRED: {resolution.get('escalation_reason', '')}", 
                       ParagraphStyle("Esc", parent=styles["Normal"], fontSize=10,
                                      fontName=FONT_BOLD, textColor=RED))]],
            colWidths=[7 * inch]
        )
        esc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FEF2F2")),
            ("PADDING",    (0, 0), (-1, -1), 10),
            ("GRID",       (0, 0), (-1, -1), 0.5, RED),
        ]))
        story.append(esc_table)
        story.append(Spacer(1, 8))

    # ── IMMEDIATE ACTIONS ─────────────────────────────────
    actions = resolution.get("immediate_actions", [])
    if actions:
        story.append(Paragraph(" Immediate Actions", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8))

        actions_data = []
        for i, action in enumerate(actions, 1):
            actions_data.append([
                Paragraph(str(i), ParagraphStyle(
                    "Num", parent=styles["Normal"], fontSize=12,
                    fontName=FONT_BOLD, textColor=WHITE, alignment=TA_CENTER
                )),
                Paragraph(action, body_style)
            ])

        actions_table = Table(actions_data, colWidths=[0.4 * inch, 6.6 * inch])
        actions_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), ACCENT),
            ("BACKGROUND",  (1, 0), (1, -1), WHITE),
            ("ROWBACKGROUNDS", (1, 0), (1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID",        (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("PADDING",     (0, 0), (-1, -1), 8),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(actions_table)
        story.append(Spacer(1, 12))

    # ── GITHUB SUSPECTS ───────────────────────────────────
    commits = github_ctx.get("blame_suspects", github_ctx.get("recent_commits", []))[:5]
    if commits:
        story.append(Paragraph(" Suspect GitHub Commits", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=GRAY, spaceAfter=8))

        commit_header = [
            [
                Paragraph("SHA", label_style),
                Paragraph("Message", label_style),
                Paragraph("Author", label_style),
                Paragraph("Date", label_style),
            ]
        ]
        commit_rows = []
        for c in commits:
            commit_rows.append([
                Paragraph(c.get("sha", "")[:7], mono_style),
                Paragraph(c.get("message", "")[:60], body_style),
                Paragraph(c.get("author", ""), body_style),
                Paragraph(c.get("date", ""), body_style),
            ])

        commit_table = Table(
            commit_header + commit_rows,
            colWidths=[0.7 * inch, 3.8 * inch, 1.5 * inch, 1 * inch]
        )
        commit_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID",        (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("PADDING",     (0, 0), (-1, -1), 7),
            ("FONTNAME",    (0, 0), (-1, 0), FONT_BOLD),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(commit_table)
        story.append(Spacer(1, 12))

    # ── JIRA BUGS ─────────────────────────────────────────
    bugs = jira_ctx.get("known_bugs", [])[:5]
    if bugs:
        story.append(Paragraph(" Related Jira Bugs", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8))

        bug_header = [[
            Paragraph("ID", label_style),
            Paragraph("Summary", label_style),
            Paragraph("Status", label_style),
            Paragraph("Priority", label_style),
        ]]
        bug_rows = []
        for b in bugs:
            bug_rows.append([
                Paragraph(b.get("id", ""), mono_style),
                Paragraph(b.get("summary", "")[:60], body_style),
                Paragraph(b.get("status", ""), body_style),
                Paragraph(b.get("priority", ""), body_style),
            ])

        bug_table = Table(
            bug_header + bug_rows,
            colWidths=[0.9 * inch, 3.6 * inch, 1.3 * inch, 1.2 * inch]
        )
        bug_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID",        (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("PADDING",     (0, 0), (-1, -1), 7),
            ("FONTNAME",    (0, 0), (-1, 0), FONT_BOLD),
        ]))
        story.append(bug_table)
        story.append(Spacer(1, 12))

    # ── SOURCES CITED ─────────────────────────────────────
    sources = resolution.get("sources_used") or enriched.get("sources_used", {})
    if sources and any(sources.values()):
        story.append(Paragraph(" Sources Cited", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8))

        sources_data = []
        if sources.get("jira"):
            sources_data.append([
                Paragraph("Jira", label_style),
                Paragraph(", ".join(sources["jira"]), mono_style)
            ])
        if sources.get("confluence"):
            sources_data.append([
                Paragraph("Confluence", label_style),
                Paragraph(", ".join(sources["confluence"]), body_style)
            ])
        if sources.get("github"):
            sources_data.append([
                Paragraph("GitHub", label_style),
                Paragraph(", ".join(sources["github"]), mono_style)
            ])

        if sources_data:
            sources_table = Table(sources_data, colWidths=[1.2 * inch, 5.8 * inch])
            sources_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
                ("GRID",       (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
                ("PADDING",    (0, 0), (-1, -1), 8),
                ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(sources_table)
            story.append(Spacer(1, 12))

    # ── FOOTER ────────────────────────────────────────────
    story.append(Spacer(1, 16))
    footer_data = [[
        Paragraph(
            f"Generated by Servion Support AI Agent  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Confidential",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                           textColor=WHITE, alignment=TA_CENTER)
        )
    ]]
    footer_table = Table(footer_data, colWidths=[7 * inch])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("PADDING",    (0, 0), (-1, -1), 8),
    ]))
    story.append(footer_table)

    # Build PDF
    doc.build(story)
    logger.info(f"[PDF] Report generated: {filepath}")
    return filepath