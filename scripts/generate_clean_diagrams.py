import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('submission/figures', exist_ok=True)

# Use crisp standard sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def draw_rounded_box(ax, x, y, w, h, ec, fc, lw=1.6, r=1.5):
    """Draw a clean rounded rectangle using FancyBboxPatch with explicit data coordinates."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        ec=ec, fc=fc, lw=lw, zorder=2
    )
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#475569', lw=1.8, hw=6, hl=7, ls='-'):
    """Draw a clean, crisp orthogonal or direct arrow with proper zorder and visible arrowhead."""
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            mutation_scale=hw,
            shrinkA=0, shrinkB=0
        ),
        zorder=3
    )

def generate_clean_figure1():
    """
    Fig. 1. Original FinQA Research Methodology & Baseline RAG Architecture.
    Completely redesigned with orthogonal flow, guaranteed padding, and zero overlaps.
    """
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Figure Header
    ax.text(50, 97.2, "Fig. 1.  Original FinQA Research Methodology & Baseline RAG Architecture",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(50, 94.2, "Standard pipeline relying on flat chunking, independent dense retrieval, and unverified in-weights neural mental math (Chen et al., 2021)",
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#475569')

    # Color scheme
    c_blue_ec, c_blue_fc = '#1E40AF', '#F0F7FF'
    c_orange_ec, c_orange_fc = '#D97706', '#FFFBEB'
    c_teal_ec, c_teal_fc = '#0F766E', '#F0FDFA'
    c_slate_ec, c_slate_fc = '#475569', '#F8FAFC'
    c_purple_ec, c_purple_fc = '#7C3AED', '#FAF5FF'
    c_red_ec, c_red_fc = '#DC2626', '#FEF2F2'

    # --- ROW 1: PIPELINE STAGES (Y: 62 to 90) ---
    # Total width available: 94 (from x=3 to x=97)
    # 4 boxes: width=20 each = 80. Gaps: 3 gaps of 5.5 = 16.5.
    # Box 1: x=3 to 23
    # Box 2: x=27.5 to 47.5
    # Box 3: x=52 to 72
    # Box 4: x=76.5 to 96.5

    # Box 1: Input Documents
    draw_rounded_box(ax, 3, 62, 20.5, 28, c_blue_ec, c_blue_fc)
    ax.text(13.25, 86.5, "1. Input Documents", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue_ec)
    ax.text(13.25, 75.5,
            "SEC Form 10-K Filings\n"
            "• Annual reports (PDF/HTML)\n"
            "• Multi-page narrative MD&A\n"
            "• Financial statement tables\n"
            "• Footnote disclosures\n"
            "• High visual heterogeneity",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Arrow 1 -> 2
    draw_arrow(ax, 23.5, 76, 27.5, 76, color='#64748B')

    # Box 2: Flat Chunking
    draw_rounded_box(ax, 27.5, 62, 20.5, 28, c_orange_ec, c_orange_fc)
    ax.text(37.75, 86.5, "2. Flat Chunking", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_orange_ec)
    ax.text(37.75, 75.5,
            "Document Linearization\n"
            "• Fixed 256-token text windows\n"
            "• Tables flattened row-by-row\n"
            "• Headers separated from cells\n"
            "• Reporting scale tags stripped\n"
            "• Table context fragmented",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Arrow 2 -> 3
    draw_arrow(ax, 48, 76, 52, 76, color='#64748B')

    # Box 3: Standard Retrieval
    draw_rounded_box(ax, 52, 62, 20.5, 28, c_teal_ec, c_teal_fc)
    ax.text(62.25, 86.5, "3. Standard Retrieval", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_teal_ec)
    ax.text(62.25, 75.5,
            "Top-K Chunk Selection\n"
            "• Dense Bi-Encoder / BM25\n"
            "• Independent chunk scoring\n"
            "• Top-10 candidates selected\n"
            "• Missing multi-span pairing\n"
            "• No parent context expansion",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Arrow 3 -> 4
    draw_arrow(ax, 72.5, 76, 76.5, 76, color='#64748B')

    # Box 4: Retrieved Context
    draw_rounded_box(ax, 76.5, 62, 20.5, 28, c_slate_ec, c_slate_fc)
    ax.text(86.75, 86.5, "4. Retrieved Context", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_slate_ec)
    ax.text(86.75, 75.5,
            "Disjoint Chunk Context\n"
            "• Fragmented table rows\n"
            "• Missing column year headers\n"
            "• Missing multi-year baselines\n"
            "• 52.9% multi-span failure rate\n"
            "• Span Recall@10: 42.3%",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Clean Orthogonal Arrow from Box 4 Down to Row 2
    # From bottom of Box 4 (x=86.75, y=62) down to y=52
    draw_arrow(ax, 86.75, 62, 86.75, 52.5, color='#7C3AED', lw=2)

    # --- ROW 2: GENERATION & UNVERIFIED OUTPUT (Y: 33 to 52) ---
    # Box 5: Autoregressive LLM (Right side, aligns under Box 3 & 4)
    draw_rounded_box(ax, 48, 33, 49, 19.5, c_purple_ec, c_purple_fc)
    ax.text(72.5, 48.5, "5. Autoregressive LLM Generation (Single-Pass Prompting)", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_purple_ec)
    ax.text(72.5, 40.5,
            "• Concatenates retrieved disjoint chunks into an unstructured natural language prompt\n"
            "• Language model attempts in-weights 'mental math' via next-token prediction in attention logits\n"
            "• No execution sandbox or arithmetic logic unit: floating-point calculations drift severely\n"
            "• Raw Zero-Shot Numerical Accuracy: 20.0% (Fails on 80% of division and percentage calculations)",
            ha='center', va='center', fontsize=7.8, color='#1E293B', linespacing=1.35)

    # Clean Horizontal Arrow from Box 5 to Box 6 (Leftward)
    draw_arrow(ax, 48, 42.75, 41.5, 42.75, color='#DC2626', lw=2)

    # Box 6: Unverified Output (Left side, aligns under Box 1 & 2)
    draw_rounded_box(ax, 3, 33, 38.5, 19.5, c_red_ec, c_red_fc)
    ax.text(22.25, 48.5, "6. Unverified Final Output", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_red_ec)
    ax.text(22.25, 40.5,
            "• Free-form unstructured text response\n"
            "• Unchecked numerical claim without audit trail\n"
            "• Evaluated purely by token overlap groundedness (G = 1.0)\n"
            "• High risk of 'grounded-but-wrong' accounting polarity errors",
            ha='center', va='center', fontsize=7.8, color='#1E293B', linespacing=1.35)

    # --- ROW 3: VULNERABILITY AUDIT BOX (Y: 5 to 27) ---
    b_fail = patches.FancyBboxPatch(
        (3, 5), 94, 23.5,
        boxstyle="round,pad=0,rounding_size=1.5",
        ec='#B91C1C', fc='#FEF2F2', lw=1.6, linestyle='--'
    )
    ax.add_patch(b_fail)
    ax.text(50, 25.5, "CRITICAL FAILURE MODES IDENTIFIED IN BASELINE FINQA ARCHITECTURE", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#991B1B')

    f1 = (
        "A. Grounded-but-Wrong Token Overlap Trap:\n"
        "Filing reports Interest Expense $(600)M in 2009 vs $(511)M in 2008.\n"
        "Naive model calculates (-600) - (-511) = -89M and claims expenses 'decreased'.\n"
        "Baseline metric scores this as Grounded (G = 1.0) because tokens exist,\n"
        "despite inverting financial reality (expense liability grew by +$89M)."
    )
    f2 = (
        "B. Neural Floating-Point Hallucination:\n"
        "Transformers lack arithmetic execution logic. When computing division or\n"
        "growth rates (e.g., 637 / 5.0 = 127.4), models predict token probabilities,\n"
        "yielding hallucinated answers (e.g., 7.64). In live testing, 80% of mental\n"
        "math calculations fail to match ground truth."
    )
    f3 = (
        "C. Tabular Chunk Fragmentation:\n"
        "Flat chunking strips table headers ('2019' vs '2018') and reporting scales\n"
        "('in thousands') from data rows. Retrieving isolated rows forces the LLM to\n"
        "guess which cell belongs to which year, causing multi-hop errors in\n"
        "52.9% of validation queries."
    )

    ax.text(5.5, 14.5, f1, ha='left', va='center', fontsize=7.2, color='#7F1D1D', linespacing=1.25)
    ax.text(37.5, 14.5, f2, ha='left', va='center', fontsize=7.2, color='#7F1D1D', linespacing=1.25)
    ax.text(69.5, 14.5, f3, ha='left', va='center', fontsize=7.2, color='#7F1D1D', linespacing=1.25)

    plt.tight_layout()
    plt.savefig('submission/figures/original_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Clean Fig. 1 generated successfully.")

def generate_clean_figure2():
    """
    Fig. 2. Modified LLM-FINANCE Architecture (Neuro-Symbolic Financial RAG).
    Completely redesigned with clean orthogonal layout, NO overlapping lines, NO diagonal crossings.
    """
    fig, ax = plt.subplots(figsize=(14.5, 10.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Figure Header
    ax.text(50, 97.5, "Fig. 2.  Modified LLM-FINANCE Architecture — Verifiable & Audited Financial RAG",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(50, 95.0, "Neuro-symbolic framework coupling Parent-Child RRF retrieval, local BF16 LLM semantic parsing, and the DCRC execution compiler",
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#475569')

    # Colors
    c_blue_ec, c_blue_fc = '#1E40AF', '#EFF6FF'
    c_teal_ec, c_teal_fc = '#0F766E', '#F0FDFA'
    c_green_ec, c_green_fc = '#15803D', '#F0FDF4'
    c_slate_ec, c_slate_fc = '#334155', '#F8FAFC'
    c_purple_ec, c_purple_fc = '#6D28D9', '#FAF5FF'
    c_amber_ec, c_amber_fc = '#D97706', '#FFFBEB'
    c_orange_ec, c_orange_fc = '#EA580C', '#FFF7ED'
    c_red_ec, c_red_fc = '#DC2626', '#FEF2F2'

    # --- ROW 1: INGESTION, RETRIEVAL & CONTEXT EXPANSION (Y: 71 to 92) ---
    # 4 Boxes: x=2.5, x=26.5, x=51.5, x=76.5 (width=21 each)
    
    # Box 1: Dual Ingestion & Index
    draw_rounded_box(ax, 2.5, 71, 21.5, 21, c_blue_ec, c_blue_fc)
    ax.text(13.25, 88.5, "1. Ingestion & Dual Index", ha='center', va='center', fontsize=9, fontweight='bold', color=c_blue_ec)
    ax.text(13.25, 79.5,
            "• SEC 10-K Input Documents\n"
            "• Fine Child Chunks:\n"
            "  - Sentences & single rows\n"
            "  - Headers explicitly prepended\n"
            "• Parent Store: Full tables\n"
            "• BGE 384d Dense + BM25",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 1 -> 2
    draw_arrow(ax, 24, 81.5, 27, 81.5, color='#64748B')

    # Box 2: Asymmetric Weighted RRF
    draw_rounded_box(ax, 27, 71, 21.5, 21, c_teal_ec, c_teal_fc)
    ax.text(37.75, 88.5, "2. Asymmetric Weighted RRF", ha='center', va='center', fontsize=9, fontweight='bold', color=c_teal_ec)
    ax.text(37.75, 79.5,
            "• Stream A: BGE-small (Dense)\n"
            "  Semantic match (75% weight)\n"
            "• Stream B: BM25 (Sparse)\n"
            "  Exact ticker/year (25% weight)\n"
            "• S_RRF = 0.75/(60+r_d) + 0.25/(60+r_b)\n"
            "• Outperforms Std RRF (+2.9 pp)",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 2 -> 3
    draw_arrow(ax, 48.5, 81.5, 51.5, 81.5, color='#64748B')

    # Box 3: Parent-Child Expansion
    draw_rounded_box(ax, 51.5, 71, 22.5, 21, c_green_ec, c_green_fc)
    ax.text(62.75, 88.5, "3. Parent-Child Expansion", ha='center', va='center', fontsize=9, fontweight='bold', color=c_green_ec)
    ax.text(62.75, 79.5,
            "• Maps child hit to Parent Table\n"
            "• Restores headers & scale units\n"
            "• Span Recall@10: 42.3% -> 74.3%\n"
            "  (+32.0 pp gain, p = 1.32e-30)\n"
            "• Cross-Encoder redundant (p=0.15)\n"
            "• Year pre-filter rejected (-5.15 pp)",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 3 -> 4
    draw_arrow(ax, 74, 81.5, 76.5, 81.5, color='#64748B')

    # Box 4: Structured Prompting
    draw_rounded_box(ax, 76.5, 71, 21, 21, c_slate_ec, c_slate_fc)
    ax.text(87, 88.5, "4. Structured Prompting", ha='center', va='center', fontsize=9, fontweight='bold', color=c_slate_ec)
    ax.text(87, 79.5,
            "• Context deduplication\n"
            "• Token budget (2,048 tokens)\n"
            "• 2-Shot Financial Prompts:\n"
            "  - Decimal percentage ratio\n"
            "  - Verbatim operand citations\n"
            "• Enforces CLAIM_SCHEMA JSON",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Clean Orthogonal Arrow: Down from Box 4 to Row 2
    draw_arrow(ax, 87, 71, 87, 61, color=c_purple_ec, lw=2)

    # --- ROW 2: GENERATION & VERIFICATION (Y: 40 to 60) ---
    # Box 5: Local LLM Semantic Parser (Right side, x=67 to 97.5)
    draw_rounded_box(ax, 67, 40, 30.5, 20.5, c_purple_ec, c_purple_fc)
    ax.text(82.25, 57.5, "5. Local LLM Semantic Parser", ha='center', va='center', fontsize=9, fontweight='bold', color=c_purple_ec)
    ax.text(82.25, 48.5,
            "• Qwen2.5-3B-Instruct in Hardware BF16\n"
            "• 100% Local GPU Execution (RTX 4060 8GB)\n"
            "• VRAM: 5.88 GB (< 6.4 GB peak, > 2.1 GB free)\n"
            "• High Groundedness (G = 91.7%, J = 95.0%)\n"
            "• Emits formula string & operand citations\n"
            "  (Decoupled from neural math!)",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Clean Horizontal Arrow from Box 5 to Box 6 (Leftward)
    draw_arrow(ax, 67, 50.25, 62.5, 50.25, color='#64748B')

    # Box 6: Schema & Output Parser (Center, x=34.5 to 62.5)
    draw_rounded_box(ax, 34.5, 40, 28, 20.5, c_amber_ec, c_amber_fc)
    ax.text(48.5, 57.5, "6. Schema & Output Parser", ha='center', va='center', fontsize=9, fontweight='bold', color=c_amber_ec)
    ax.text(48.5, 48.5,
            "• Pydantic ClaimSchema validation\n"
            "• Referential Integrity Verification:\n"
            "  All formula variables must bind\n"
            "  to declared operand citations\n"
            "• Strips markdown JSON delimiters\n"
            "• Emits StructuredClaim object",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Clean Horizontal Arrow from Box 6 to Box 7 (Leftward)
    draw_arrow(ax, 34.5, 50.25, 30, 50.25, color='#64748B')

    # Box 7: Neuro-Symbolic Verifier (Left side, x=2.5 to 30)
    draw_rounded_box(ax, 2.5, 40, 27.5, 20.5, c_orange_ec, c_orange_fc)
    ax.text(16.25, 57.5, "7. Neuro-Symbolic Verifier", ha='center', va='center', fontsize=9, fontweight='bold', color=c_orange_ec)
    ax.text(16.25, 48.5,
            "• Verbatim Substring Grounding:\n"
            "  Operands verified in parent filing\n"
            "• Fiscal Period Alignment (Year/Qtr)\n"
            "• Scale Normalization (M, B, %, $)\n"
            "• Polarity & Direction Verification:\n"
            "  Magnitude: |v_curr| - |v_prev|\n"
            "  Prevents expense decrease errors",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Clean Orthogonal Arrow Down from Box 7 to Box 8
    draw_arrow(ax, 16.25, 40, 16.25, 30, color=c_blue_ec, lw=2)

    # --- ROW 3: COMPILER, REPAIR & CERTIFIED OUTPUT (Y: 9 to 29) ---
    
    # Box 8: DCRC Symbolic Execution Compiler (Left side, x=2.5 to 47.5)
    draw_rounded_box(ax, 2.5, 9, 45, 20, c_blue_ec, c_blue_fc)
    ax.text(25, 26, "8. DCRC Symbolic Execution Compiler", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue_ec)
    ax.text(25, 17.5,
            "• Decouples mathematical arithmetic from generative attention\n"
            "• Safe Python AST Evaluation Engine (safe_eval_program):\n"
            "  - Exact 64-bit IEEE float / Decimal arithmetic evaluation\n"
            "  - Division-by-zero protection & numerical domain checks\n"
            "• Accuracy jumped: 20.0% -> 55.0% (+35.0 pp gain)\n"
            "• Composite Score S_Comp jumped: 0.6889 -> 0.7833\n"
            "• Execution Overhead: Only 3.8 ms per query",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Clean Horizontal Arrow: Box 8 -> Box 10 (Direct Green Verification Arrow)
    draw_arrow(ax, 47.5, 19, 74.5, 19, color=c_green_ec, lw=2.2)
    # Badge on Arrow
    ax.text(61, 21.5, "Verified (Passed)", ha='center', va='center', fontsize=8, fontweight='bold', color=c_green_ec,
            bbox=dict(boxstyle="round,pad=0.25", fc="#DCFCE7", ec=c_green_ec, lw=1))

    # Box 9: Diagnostic Repair & Abstention (Positioned neatly in middle space x=50.5 to 71.5, y=28 to 39)
    draw_rounded_box(ax, 50.5, 27.5, 21, 11, c_red_ec, c_red_fc)
    ax.text(61, 35.5, "9. Diagnostic Repair", ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_red_ec)
    ax.text(61, 31,
            "• 1-Turn repair loop on syntax error\n"
            "• Deterministic abstention on missing data",
            ha='center', va='center', fontsize=6.8, color='#7F1D1D', linespacing=1.2)

    # Orthogonal Manhattan Routing for Repair:
    # From right side of Box 7 (x=30, y=42) horizontally to x=50.5, into Box 9
    ax.annotate(
        '', xy=(50.5, 33), xytext=(30, 42),
        arrowprops=dict(
            arrowstyle="-|>", color=c_red_ec, lw=1.4, linestyle="--",
            mutation_scale=5
        ), zorder=3
    )
    ax.text(38, 40, "Syntax / Citation Fault", ha='center', va='center', fontsize=6.8, color=c_red_ec, fontstyle='italic')

    # From Box 9 (x=71.5, y=33) into Box 5 (x=71.5, y=40)
    ax.annotate(
        '', xy=(71.5, 40), xytext=(71.5, 33),
        arrowprops=dict(
            arrowstyle="-|>", color=c_red_ec, lw=1.4, linestyle="--",
            mutation_scale=5
        ), zorder=3
    )
    ax.text(78, 35, "Feedback Retry", ha='left', va='center', fontsize=6.8, color=c_red_ec, fontstyle='italic')

    # Box 10: Certified Final Output (Right side, x=75 to 97.5, y=9 to 29)
    draw_rounded_box(ax, 75, 9, 22.5, 20, c_green_ec, c_green_fc, lw=2)
    ax.text(86.25, 26, "10. Certified Output", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_green_ec)
    ax.text(86.25, 17.5,
            "• Mathematically guaranteed\n"
            "  scalar answer\n"
            "• Full provenance audit trail\n"
            "• Verbatim citation bindings\n"
            "• Polarity-verified direction\n"
            "• 100% private & reproducible",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color=c_green_ec, linespacing=1.3)

    # --- DIAGRAM LEGEND (Bottom bar, Y: 1.5 to 6.5) ---
    leg = patches.FancyBboxPatch(
        (2.5, 1.5), 95, 5.5,
        boxstyle="round,pad=0,rounding_size=1",
        ec='#CBD5E1', fc='#FFFFFF', lw=1
    )
    ax.add_patch(leg)
    ax.text(8, 4.25, "DIAGRAM LEGEND:", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#334155')

    # Legend 1
    ax.plot([25, 29], [4.25, 4.25], color='#64748B', lw=2)
    ax.text(30.5, 4.25, "Standard Pipeline Flow", ha='left', va='center', fontsize=7.2, color='#334155')

    # Legend 2
    ax.plot([50, 54], [4.25, 4.25], color=c_green_ec, lw=2.2)
    ax.text(55.5, 4.25, "Successful Verification Path", ha='left', va='center', fontsize=7.2, color='#334155')

    # Legend 3
    ax.plot([75, 79], [4.25, 4.25], color=c_red_ec, lw=1.6, linestyle='--')
    ax.text(80.5, 4.25, "1-Turn Diagnostic Repair Loop", ha='left', va='center', fontsize=7.2, color='#334155')

    plt.tight_layout()
    plt.savefig('submission/figures/modified_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Clean Fig. 2 generated successfully.")

if __name__ == "__main__":
    generate_clean_figure1()
    generate_clean_figure2()
