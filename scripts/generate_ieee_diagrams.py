import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('submission/figures', exist_ok=True)

# Set global typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def generate_figure1():
    """
    Fig. 1. Original Research Methodology (Chen et al., EMNLP 2021 / FinQANet Baseline).
    Clean, non-overlapping, strictly formatted block diagram.
    """
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Figure Title & Caption Header
    ax.text(50, 96.5, "Fig. 1.  Original FinQA Research Methodology & Baseline RAG Architecture",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(50, 93.5, "Standard pipeline relying on flat chunking, dense retrieval, and unverified in-weights neural mental math (Chen et al., 2021)",
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#475569')

    # Color Palette (Muted, professional academic)
    c_blue_border = '#1E40AF'
    c_blue_bg = '#EFF6FF'
    c_orange_border = '#C2410C'
    c_orange_bg = '#FFF7ED'
    c_teal_border = '#0F766E'
    c_teal_bg = '#F0FDFA'
    c_purple_border = '#6D28D9'
    c_purple_bg = '#FAF5FF'
    c_red_border = '#B91C1C'
    c_red_bg = '#FEF2F2'
    c_gray_border = '#475569'
    c_gray_bg = '#F8FAFC'

    # --- TOP ROW: DATA INGESTION & RETRIEVAL (Y: 63 to 88) ---
    
    # 1. Input Document
    b1 = patches.FancyBboxPatch((3, 64), 20, 24, boxstyle="round,pad=0.5", ec=c_blue_border, fc=c_blue_bg, lw=1.5)
    ax.add_patch(b1)
    ax.text(13, 85, "1. Input Documents", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue_border)
    ax.text(13, 75, "SEC Form 10-K Filings\n• Annual reports (PDF/HTML)\n• Multi-page narrative MD&A\n• Financial statement tables\n• Footnote disclosures",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Arrow 1 -> 2
    ax.annotate('', xy=(27, 76), xytext=(23, 76),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 2. Document Processing (Flat Chunking)
    b2 = patches.FancyBboxPatch((27, 64), 21, 24, boxstyle="round,pad=0.5", ec=c_orange_border, fc=c_orange_bg, lw=1.5)
    ax.add_patch(b2)
    ax.text(37.5, 85, "2. Flat Chunking", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_orange_border)
    ax.text(37.5, 75, "Document Linearization\n• Fixed 256-token text chunks\n• Tables flattened row-by-row\n• Headers separated from cells\n• Reporting units lost",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Arrow 2 -> 3
    ax.annotate('', xy=(52, 76), xytext=(48, 76),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 3. Standard Retrieval
    b3 = patches.FancyBboxPatch((52, 64), 21, 24, boxstyle="round,pad=0.5", ec=c_teal_border, fc=c_teal_bg, lw=1.5)
    ax.add_patch(b3)
    ax.text(62.5, 85, "3. Standard Retrieval", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_teal_border)
    ax.text(62.5, 75, "Top-K Selection\n• Dense Retriever / BM25\n• Scores chunks independently\n• Selects top-10 candidates\n• No parent expansion",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Arrow 3 -> 4
    ax.annotate('', xy=(77, 76), xytext=(73, 76),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 4. Retrieved Chunks
    b4 = patches.FancyBboxPatch((77, 64), 20, 24, boxstyle="round,pad=0.5", ec=c_gray_border, fc=c_gray_bg, lw=1.5)
    ax.add_patch(b4)
    ax.text(87, 85, "4. Retrieved Context", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_gray_border)
    ax.text(87, 75, "Disjoint Candidates\n• Fragmented table rows\n• Missing multi-year baselines\n• Missing column headers\n• Span Recall@10: 42.3%",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Downward Arrow from Step 4 to Step 5
    ax.annotate('', xy=(87, 53), xytext=(87, 64),
                arrowprops=dict(facecolor='#6D28D9', edgecolor='#6D28D9', width=2, headwidth=6))

    # --- MIDDLE ROW: GENERATION & UNVERIFIED OUTPUT (Y: 34 to 53) ---

    # 5. Autoregressive LLM Generation
    b5 = patches.FancyBboxPatch((48, 34), 49, 19, boxstyle="round,pad=0.5", ec=c_purple_border, fc=c_purple_bg, lw=1.5)
    ax.add_patch(b5)
    ax.text(72.5, 49.5, "5. Autoregressive LLM Generation (Single-Pass Prompting)", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_purple_border)
    ax.text(72.5, 41.5, "• Concatenates retrieved disjoint chunks into unstructured text prompt\n• Model attempts 'mental math' via autoregressive token prediction in transformer weights\n• No execution registers or sandbox: multi-step arithmetic calculated in attention logits\n• Raw Numerical Accuracy: 20.0% (Small models cannot reliably perform multi-digit division)",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # Arrow 5 -> 6 (Leftward)
    ax.annotate('', xy=(39, 43.5), xytext=(48, 43.5),
                arrowprops=dict(facecolor='#B91C1C', edgecolor='#B91C1C', width=1.5, headwidth=5))

    # 6. Unverified Final Output
    b6 = patches.FancyBboxPatch((3, 34), 36, 19, boxstyle="round,pad=0.5", ec=c_red_border, fc=c_red_bg, lw=1.5)
    ax.add_patch(b6)
    ax.text(21, 49.5, "6. Unverified Final Output", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_red_border)
    ax.text(21, 41.5, "• Free-form natural language text\n• Unchecked arithmetic output (no audit trail)\n• Evaluated by token overlap groundedness (G = 1.0)\n• High risk of 'grounded-but-wrong' polarity reversal",
            ha='center', va='center', fontsize=8, color='#1E293B', linespacing=1.35)

    # --- BOTTOM ROW: FAILURE MODES DIAGNOSTIC BOX (Y: 6 to 28) ---
    b_fail = patches.FancyBboxPatch((3, 6), 94, 23, boxstyle="round,pad=0.6", ec='#991B1B', fc='#FEF2F2', lw=1.5, linestyle='--')
    ax.add_patch(b_fail)
    ax.text(50, 26, "CRITICAL FAILURE MODES IDENTIFIED IN BASELINE ARCHITECTURE", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#991B1B')

    fail1 = "A. Grounded-but-Wrong Token Overlap Trap:\nFiling reports Interest Expense $(600)M in 2009 vs $(511)M in 2008. Naive model calculates (-600) - (-511) = -89M and claims expenses 'decreased'. Baseline metric scores this as Grounded (G = 1.0) because tokens exist, despite inverting financial reality (expense liability grew by +$89M)."
    fail2 = "B. Neural Floating-Point Hallucination:\nTransformers lack arithmetic execution logic. When tasked with division or percentage change (e.g., 637 / 5.0 = 127.4), models predict token frequencies, yielding hallucinated answers (e.g. 7.64). In empirical testing, 80% of mental math answers fail."
    fail3 = "C. Tabular Chunk Fragmentation:\nFlat chunking strips table headers ('2019' vs '2018') and reporting scales ('in thousands') from data rows. Retrieving isolated rows forces the LLM to guess which cell belongs to which year, causing multi-hop errors in 52.9% of queries."

    ax.text(5.5, 16.5, fail1, ha='left', va='top', fontsize=7.6, color='#7F1D1D', linespacing=1.25)
    ax.text(52, 16.5, fail2, ha='left', va='top', fontsize=7.6, color='#7F1D1D', linespacing=1.25)
    ax.text(5.5, 9, fail3, ha='left', va='top', fontsize=7.6, color='#7F1D1D', linespacing=1.25)

    # Legend in lower right corner
    leg_box = patches.Rectangle((75, 7), 21, 6.5, ec='#CBD5E1', fc='#FFFFFF', lw=1)
    ax.add_patch(leg_box)
    ax.text(85.5, 12, "LEGEND", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#334155')
    ax.plot([76.5, 79], [9.5, 9.5], color='#64748B', lw=1.5)
    ax.text(80, 9.5, "Standard flow", ha='left', va='center', fontsize=7, color='#334155')
    ax.plot([76.5, 79], [7.8, 7.8], color='#B91C1C', lw=1.5)
    ax.text(80, 7.8, "Unverified/Fail", ha='left', va='center', fontsize=7, color='#334155')

    plt.tight_layout()
    plt.savefig('submission/figures/original_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Regenerated Fig. 1 (original_architecture.png) successfully.")

def generate_figure2():
    """
    Fig. 2. Modified LLM-FINANCE Architecture (Neuro-symbolic RAG + DCRC Compiler).
    Clean, non-overlapping, strictly formatted block diagram.
    """
    fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Figure Title & Caption Header
    ax.text(50, 97.5, "Fig. 2.  Modified LLM-FINANCE Architecture — Verifiable & Audited Financial RAG",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(50, 95.0, "Neuro-symbolic framework coupling Parent-Child RRF retrieval, local BF16 LLM semantic parsing, and the DCRC execution compiler",
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#475569')

    # Palette
    c_blue = '#1E40AF'
    c_blue_bg = '#EFF6FF'
    c_teal = '#0F766E'
    c_teal_bg = '#F0FDFA'
    c_green = '#15803D'
    c_green_bg = '#F0FDF4'
    c_purple = '#6D28D9'
    c_purple_bg = '#FAF5FF'
    c_amber = '#B45309'
    c_amber_bg = '#FFFBEB'
    c_red = '#B91C1C'
    c_red_bg = '#FEF2F2'
    c_slate = '#334155'
    c_slate_bg = '#F8FAFC'

    # --- ROW 1: INGESTION, DUAL RETRIEVAL & CONTEXT EXPANSION (Y: 72 to 92) ---
    
    # 1. Ingestion & Dual Index
    b1 = patches.FancyBboxPatch((2, 72), 21, 20, boxstyle="round,pad=0.5", ec=c_blue, fc=c_blue_bg, lw=1.5)
    ax.add_patch(b1)
    ax.text(12.5, 89.5, "1. Dual Ingestion & Index", ha='center', va='center', fontsize=9, fontweight='bold', color=c_blue)
    ax.text(12.5, 80.5, "• SEC 10-K Input Documents\n• Fine Child Chunks:\n  - Sentences & single rows\n  - Headers explicitly prepended\n• Parent Store: Full tables\n• BGE 384d Dense + BM25",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 1 -> 2
    ax.annotate('', xy=(26, 82), xytext=(23, 82),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 2. Dual-Stream Weighted RRF
    b2 = patches.FancyBboxPatch((26, 72), 22, 20, boxstyle="round,pad=0.5", ec=c_teal, fc=c_teal_bg, lw=1.5)
    ax.add_patch(b2)
    ax.text(37, 89.5, "2. Asymmetric Weighted RRF", ha='center', va='center', fontsize=9, fontweight='bold', color=c_teal)
    ax.text(37, 80.5, "• Stream A: BGE-small (Dense)\n  Semantic match (75% weight)\n• Stream B: BM25 (Sparse)\n  Exact ticker/year (25% weight)\n• S_RRF = 0.75/(60+r_d) + 0.25/(60+r_b)\n• Outperforms Std RRF (+2.9 pp)",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 2 -> 3
    ax.annotate('', xy=(51, 82), xytext=(48, 82),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 3. Parent-Child Expansion
    b3 = patches.FancyBboxPatch((51, 72), 23, 20, boxstyle="round,pad=0.5", ec=c_green, fc=c_green_bg, lw=1.5)
    ax.add_patch(b3)
    ax.text(62.5, 89.5, "3. Parent-Child Expansion", ha='center', va='center', fontsize=9, fontweight='bold', color=c_green)
    ax.text(62.5, 80.5, "• Maps child hit to Parent Table\n• Restores headers & units\n• Span Recall@10: 42.3% -> 74.3%\n  (+32.0 pp gain, p = 1.32e-30)\n• Cross-Encoder redundant (p=0.15)\n• Year pre-filter rejected (-5.15 pp)",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 3 -> 4
    ax.annotate('', xy=(77, 82), xytext=(74, 82),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 4. Context Budgeting & Prompting
    b4 = patches.FancyBboxPatch((77, 72), 21, 20, boxstyle="round,pad=0.5", ec=c_slate, fc=c_slate_bg, lw=1.5)
    ax.add_patch(b4)
    ax.text(87.5, 89.5, "4. Structured Prompting", ha='center', va='center', fontsize=9, fontweight='bold', color=c_slate)
    ax.text(87.5, 80.5, "• Context deduplication\n• Token budget (2,048 tokens)\n• 2-Shot Financial Prompts:\n  - Decimal percentages\n  - Verbatim citations\n• Enforces CLAIM_SCHEMA",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Downward Arrow 4 -> 5
    ax.annotate('', xy=(87.5, 61), xytext=(87.5, 72),
                arrowprops=dict(facecolor=c_purple, edgecolor=c_purple, width=2, headwidth=6))

    # --- ROW 2: GENERATION & VERIFICATION ENGINE (Y: 41 to 61) ---

    # 5. Local LLM Semantic Parser
    b5 = patches.FancyBboxPatch((67, 41), 31, 20, boxstyle="round,pad=0.5", ec=c_purple, fc=c_purple_bg, lw=1.5)
    ax.add_patch(b5)
    ax.text(82.5, 58.5, "5. Local LLM Semantic Parser", ha='center', va='center', fontsize=9, fontweight='bold', color=c_purple)
    ax.text(82.5, 49.5, "• Qwen2.5-3B-Instruct in Hardware BF16\n• 100% Local GPU Execution (RTX 4060 8GB)\n• VRAM: 5.88 GB (< 6.4 GB peak, > 2.1 GB free)\n• High Groundedness (G=91.7%, J=95.0%)\n• Role: Emits symbolic formula string\n  (Decoupled from neural math!)",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 5 -> 6 (Leftward)
    ax.annotate('', xy=(63, 51), xytext=(67, 51),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 6. Schema & Output Parser
    b6 = patches.FancyBboxPatch((36, 41), 27, 20, boxstyle="round,pad=0.5", ec=c_amber, fc=c_amber_bg, lw=1.5)
    ax.add_patch(b6)
    ax.text(49.5, 58.5, "6. Schema & Output Parser", ha='center', va='center', fontsize=9, fontweight='bold', color=c_amber)
    ax.text(49.5, 49.5, "• Pydantic ClaimSchema validation\n• Referential Integrity Check:\n  All formula variables must bind\n  to declared operand citations\n• Strips markdown artifacts\n• Emits StructuredClaim object",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Arrow 6 -> 7 (Leftward)
    ax.annotate('', xy=(32, 51), xytext=(36, 51),
                arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', width=1.5, headwidth=5))

    # 7. Verification Engine
    b7 = patches.FancyBboxPatch((2, 41), 30, 20, boxstyle="round,pad=0.5", ec='#EA580C', fc='#FFF7ED', lw=1.5)
    ax.add_patch(b7)
    ax.text(17, 58.5, "7. Neuro-Symbolic Verifier", ha='center', va='center', fontsize=9, fontweight='bold', color='#EA580C')
    ax.text(17, 49.5, "• Verbatim Substring Grounding:\n  Operands checked against parent text\n• Fiscal Period Alignment (Year/Qtr)\n• Unit Normalization (M, B, %, $)\n• Polarity Check: Magnitude variance\n  |v_curr| - |v_prev| vs. signed diff",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Downward Arrow 7 -> 8
    ax.annotate('', xy=(17, 30), xytext=(17, 41),
                arrowprops=dict(facecolor=c_blue, edgecolor=c_blue, width=2, headwidth=6))

    # --- ROW 3: COMPILER, REPAIR LOOP & CERTIFIED OUTPUT (Y: 10 to 30) ---

    # 8. DCRC Symbolic Execution Compiler
    b8 = patches.FancyBboxPatch((2, 10), 44, 20, boxstyle="round,pad=0.5", ec=c_blue, fc=c_blue_bg, lw=1.5)
    ax.add_patch(b8)
    ax.text(24, 27.5, "8. DCRC Symbolic Execution Compiler", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue)
    ax.text(24, 18.5, "• Decouples mathematical arithmetic from generative attention\n• Safe Python AST Evaluation Engine (safe_eval_program):\n  - Exact 64-bit IEEE float / Decimal arithmetic evaluation\n  - Division-by-zero protection & numerical domain checks\n• Accuracy jumped: 20.0% -> 55.0% (+35.0 pp gain)\n• Composite Score S_Comp jumped: 0.6889 -> 0.7833\n• Compiler Execution Overhead: Only 3.8 ms per query",
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.3)

    # Successful Path Arrow (8 -> 10)
    ax.annotate('', xy=(76, 20), xytext=(46, 20),
                arrowprops=dict(facecolor=c_green, edgecolor=c_green, width=2, headwidth=6))
    ax.text(61, 22.5, "Verified (Passed)", ha='center', va='center', fontsize=8, fontweight='bold', color=c_green)

    # 9. Repair Loop & Abstention Engine (Above / Beside)
    b9 = patches.FancyBboxPatch((50, 27), 23, 11, boxstyle="round,pad=0.4", ec=c_red, fc=c_red_bg, lw=1.4)
    ax.add_patch(b9)
    ax.text(61.5, 34.5, "9. Diagnostic Repair & Abstention", ha='center', va='center', fontsize=8, fontweight='bold', color=c_red)
    ax.text(61.5, 30.5, "• 1-Turn repair loop on syntax error\n• Deterministic abstention on missing data",
            ha='center', va='center', fontsize=6.8, color='#7F1D1D')

    # Arrow from 7 to 9 (Repair branch)
    ax.annotate('', xy=(50, 32), xytext=(32, 45),
                arrowprops=dict(facecolor=c_red, edgecolor=c_red, width=1, headwidth=4, linestyle=':'))
    
    # Arrow from 9 back to 5 (Feedback loop)
    ax.annotate('', xy=(72, 41), xytext=(65, 38),
                arrowprops=dict(facecolor=c_red, edgecolor=c_red, width=1, headwidth=4, linestyle='--'))

    # 10. Final Certified Answer
    b10 = patches.FancyBboxPatch((76, 10), 22, 20, boxstyle="round,pad=0.5", ec=c_green, fc=c_green_bg, lw=1.8)
    ax.add_patch(b10)
    ax.text(87, 27.5, "10. Certified Output", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_green)
    ax.text(87, 18.5, "• Mathematically guaranteed\n  scalar answer\n• Full provenance audit trail\n• Verbatim citation bindings\n• Polarity-verified direction\n• 100% private & reproducible",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color=c_green, linespacing=1.3)

    # LEGEND (Bottom Right)
    leg = patches.Rectangle((2, 2), 96, 6, ec='#CBD5E1', fc='#FFFFFF', lw=1)
    ax.add_patch(leg)
    ax.text(6, 5, "DIAGRAM LEGEND:", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#334155')
    
    # Item 1
    ax.plot([23, 26], [5, 5], color=c_blue, lw=2)
    ax.text(27, 5, "Core pipeline", ha='left', va='center', fontsize=7.2, color='#334155')
    
    # Item 2
    ax.plot([42, 45], [5, 5], color=c_green, lw=2)
    ax.text(46, 5, "Successful verification path", ha='left', va='center', fontsize=7.2, color='#334155')
    
    # Item 3
    ax.plot([67, 70], [5, 5], color=c_red, lw=1.5, linestyle='--')
    ax.text(71, 5, "1-Turn diagnostic repair loop", ha='left', va='center', fontsize=7.2, color='#334155')

    plt.tight_layout()
    plt.savefig('submission/figures/modified_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Regenerated Fig. 2 (modified_architecture.png) successfully.")

if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
