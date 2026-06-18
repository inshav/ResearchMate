import os
# Configure Matplotlib to function completely inside memory-only serverless runtimes
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file
from google import genai
from google.genai import types
from datetime import datetime
import json
import zipfile
import io
import base64
import re

# Vercel Serverless Function Configuration
# Note: Vercel Hobby tier restricts executions to a 10-second ceiling. 
# Sequential LLM generations may time out on Hobby. Vercel Pro is recommended.
maxDuration = 60

app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "researchmate_secure_fallback_key")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ─── HELPER CORE LOGIC PARITY ────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0

def calculate_sections_completed(sections: dict) -> int:
    return sum(1 for v in sections.values() if str(v).strip())

def calculate_readiness_score(setup: dict, research_plan: dict, lit_table: list, citations: list, sections: dict, validation_result: dict) -> int:
    score = 0
    if setup.get("title"): score += 10
    if setup.get("objective"): score += 10
    if research_plan and not research_plan.get("raw") == "": score += 10
    if lit_table and len(lit_table) > 0: score += 10
    if citations and len(citations) >= 5: score += 10
    
    completed_count = calculate_sections_completed(sections)
    section_score = int((completed_count / 7) * 40)
    score += section_score
    
    if validation_result and not validation_result.get("raw") == "": score += 10
    return min(score, 100)

def calculate_page_estimate(sections: dict) -> float:
    total_words = sum(word_count(str(v)) for v in sections.values())
    return round(total_words / 250, 1)

def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not PYMUPDF_AVAILABLE:
        return "[PyMuPDF not installed. Install with: pip install pymupdf]"
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text[:8000]
    except Exception as e:
        return f"[PDF extraction error: {e}]"

def format_authors_latex(authors: list) -> str:
    lines = []
    for a in authors:
        if a.get("name"):
            lines.append(f"\\author{{{a['name']}}} \\affil{{{a.get('department','')}, {a.get('college','')}}}")
    return "\n".join(lines) if lines else "\\author{Author Name}"

# ─── LATEX TEMPLATES ─────────────────────────────────────────────────────────
LATEX_TEMPLATES = {
    "IEEE": r"""\documentclass[conference]{{IEEEtran}}
\IEEEoverridecommandlockouts
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}
\title{{{title}}}
{authors}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}
\section{{Introduction}}
{introduction}
\section{{Related Work}}
{literature_review}
\section{{Methodology}}
{methodology}
\section{{Results}}
{results}
\section{{Discussion}}
{discussion}
\section{{Conclusion}}
{conclusion}
\bibliographystyle{{IEEEtran}}
\bibliography{{references}}
\end{{document}}""",
    "Springer": r"""\documentclass{{svjour3}}
\usepackage{{graphicx}}
\usepackage{{cite}}
\title{{{title}}}
{authors}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}
\section{{Introduction}}
{introduction}
\section{{Literature Review}}
{literature_review}
\section{{Methodology}}
{methodology}
\section{{Results}}
{results}
\section{{Discussion}}
{discussion}
\section{{Conclusion}}
{conclusion}
\bibliographystyle{{spbasic}}
\bibliography{{references}}
\end{{document}}""",
    "Elsevier": r"""\documentclass[preprint,12pt]{{elsarticle}}
\usepackage{{graphicx}}
\usepackage{{amssymb}}
\usepackage{{cite}}
\journal{{{journal}}}
\begin{{document}}
\begin{{frontmatter}}
\title{{{title}}}
{authors}
\begin{{abstract}}
{abstract}
\end{{abstract}}
\end{{frontmatter}}
\section{{Introduction}}
{introduction}
\section{{Literature Review}}
{literature_review}
\section{{Methodology}}
{methodology}
\section{{Results}}
{results}
\section{{Discussion}}
{discussion}
\section{{Conclusion}}
{conclusion}
\bibliographystyle{{elsarticle-num}}
\bibliography{{references}}
\end{{document}}""",
    "College Format": r"""\documentclass[12pt,a4paper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{cite}}
\usepackage{{setspace}}
\usepackage{{titlesec}}
\doublespacing
\title{{{title}}}
{authors}
\date{{\today}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}
\section{{Introduction}}
{introduction}
\section{{Literature Review}}
{literature_review}
\section{{Methodology}}
{methodology}
\section{{Results}}
{results}
\section{{Discussion}}
{discussion}
\section{{Conclusion}}
{conclusion}
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}""",
}

def build_latex_content(setup: dict, authors: list, sections: dict) -> str:
    template = LATEX_TEMPLATES.get(setup.get("format_style", "IEEE"), LATEX_TEMPLATES["College Format"])
    author_str = format_authors_latex(authors)
    return template.format(
        title=setup.get("title", "Research Title"),
        authors=author_str,
        journal=setup.get("journal", "Journal Name"),
        abstract=sections.get("abstract", ""),
        introduction=sections.get("introduction", ""),
        literature_review=sections.get("literature_review", ""),
        methodology=sections.get("methodology", ""),
        results=sections.get("results", ""),
        discussion=sections.get("discussion", ""),
        conclusion=sections.get("conclusion", ""),
    )

def build_bib_content(citations: list) -> str:
    lines = []
    for c in citations:
        lines.append(c.get("bibtex", ""))
    return "\n\n".join(filter(None, lines))

def build_metadata_dict(setup: dict, authors: list, citations: list, sections: dict) -> dict:
    return {
        "title": setup.get("title", ""),
        "domain": setup.get("domain", ""),
        "keywords": setup.get("keywords", ""),
        "authors": authors,
        "journal": setup.get("journal", ""),
        "format_style": setup.get("format_style", ""),
        "sections_completed": calculate_sections_completed(sections),
        "readiness_score": calculate_readiness_score(setup, {}, [], citations, sections, {}),
        "references": len(citations),
        "generated_at": datetime.now().isoformat(),
    }

# ─── CORE GENAI INVOCATION ENGINE ────────────────────────────────────────────
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing on host configuration.")
    return genai.Client(api_key=api_key)

def call_gemini_engine(prompt: str, max_tokens: int = 1500) -> str:
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=max_tokens
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error: {e}]"

# ─── FLASK ENDPOINTS ──────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/generate_plan", methods=["POST"])
def api_generate_plan():
    data = request.json or {}
    setup = data.get("setup", {})
    
    title = setup.get("title", "")
    domain = setup.get("domain", "")
    keywords = setup.get("keywords", "")
    objective = setup.get("objective", "")
    journal = setup.get("journal", "")
    
    if not title or not objective:
        return jsonify({"error": "Please provide both a Title and Objective statement."}), 400

    prompt = f"""You are an academic research advisor. Analyze this research topic and provide a structured plan.

Research Title: {title}
Domain: {domain}
Keywords: {keywords}
Objective: {objective}
Target Journal: {journal}

Provide a concise response in this exact JSON format:
{{
  "scope": "2-3 sentences on the scope of this research",
  "gap": "2-3 sentences identifying the research gap this work addresses",
  "methodology": "2-3 sentences on the most appropriate methodology",
  "structure": ["Section 1 title", "Section 2 title", "Section 3 title", "Section 4 title", "Section 5 title", "Section 6 title", "Section 7 title"]
}}
Return only the JSON, no extra text."""

    result = call_gemini_engine(prompt, 800)
    try:
        clean = re.sub(r"```json|```", "", result).strip()
        parsed = json.loads(clean)
        return jsonify({"research_plan": parsed})
    except Exception:
        return jsonify({"research_plan": {"raw": result}})

@app.route("/api/analyze_papers", methods=["POST"])
def api_analyze_papers():
    if "files" not in request.files:
        return jsonify({"error": "No files found inside attachment stream."}), 400
        
    uploaded_files = request.files.getlist("files")
    extracted_meta_list = []
    extracted_table_rows = []
    
    for pdf_file in uploaded_files:
        if pdf_file.filename == "":
            continue
        try:
            file_bytes = pdf_file.read()
            text = extract_pdf_text(file_bytes)
            
            prompt = f"""Analyze this research paper text and return a JSON object with these fields:
- title: paper title (string)
- authors: author names (string)
- year: publication year (string)
- keywords: main keywords (string)
- methodology: research methodology used (string)
- dataset: dataset or data source used (string)
- findings: key findings (string)
- limitations: stated limitations (string)
- summary: 3-sentence summary (string)

Paper text (first 4000 chars):
{text[:4000]}

Return only valid JSON, no extra text."""

            result = call_gemini_engine(prompt, 600)
            try:
                clean = re.sub(r"```json|```", "", result).strip()
                meta = json.loads(clean)
            except Exception:
                meta = {
                    "title": pdf_file.filename, "summary": result, "authors": "", 
                    "year": "", "keywords": "", "methodology": "", "dataset": "", 
                    "findings": "", "limitations": ""
                }
            
            meta["filename"] = pdf_file.filename
            extracted_meta_list.append(meta)
            
            extracted_table_rows.append({
                "Paper": meta.get("title", pdf_file.filename),
                "Year": meta.get("year", ""),
                "Method": meta.get("methodology", ""),
                "Dataset": meta.get("dataset", ""),
                "Findings": meta.get("findings", ""),
                "Limitations": meta.get("limitations", "")
            })
        except Exception as e:
            return jsonify({"error": f"Error parsing file {pdf_file.filename}: {str(e)}"}), 500
            
    return jsonify({
        "lit_papers": extracted_meta_list,
        "lit_table": extracted_table_rows
    })

@app.route("/api/generate_citation", methods=["POST"])
def api_generate_citation():
    data = request.json or {}
    source = data.get("source", "")
    style = data.get("style", "IEEE")
    
    if not source:
        return jsonify({"error": "Source parameters missing."}), 400
        
    prompt = f"""Format this reference in {style} style.
Reference/DOI/URL: {source}

Return a JSON object:
{{
  "formatted": "the formatted citation string",
  "bibtex": "the BibTeX entry if applicable, else empty string"
}}
Return only JSON. Note: If this is a DOI or URL without full metadata, generate a plausible citation structure but mark it as [NEEDS VERIFICATION]."""

    result = call_gemini_engine(prompt, 400)
    try:
        clean = re.sub(r"```json|```", "", result).strip()
        cite_data = json.loads(clean)
        cite_data["source"] = source
        cite_data["style"] = style
        return jsonify({"citation": cite_data})
    except Exception:
        raw_cite = {"formatted": result, "bibtex": "", "source": source, "style": style}
        return jsonify({"citation": raw_cite})

@app.route("/api/improve_section", methods=["POST"])
def api_improve_section():
    data = request.json or {}
    sec_key = data.get("sec_key", "")
    user_draft = data.get("user_draft", "")
    setup = data.get("setup", {})
    lit_papers = data.get("lit_papers", [])
    
    if not sec_key or not user_draft:
        return jsonify({"error": "Required optimization payloads are empty."}), 400
        
    SECTION_PROMPTS = {
        "abstract": "Write a concise academic abstract. Include: problem, method, results, conclusion.",
        "introduction": "Write an academic introduction. Include: background, problem statement, motivation, and paper organization.",
        "literature_review": "Write a coherent literature review section. Synthesize the uploaded papers and identify research gaps.",
        "methodology": "Write a detailed methodology section. Include: research design, data collection, algorithms/techniques used.",
        "results": "Write a results section. Present findings clearly with reference to figures and tables.",
        "discussion": "Write a discussion section. Interpret results, compare with prior work, address limitations.",
        "conclusion": "Write a conclusion. Summarize contributions, implications, and future work.",
    }
    
    lit_context = ""
    if lit_papers:
        summaries = [f"- {p.get('title','')}: {p.get('summary','')[:200]}" for p in lit_papers[:5]]
        lit_context = "Relevant uploaded literature:\n" + "\n".join(summaries)
        
    prompt = f"""You are an academic writing assistant helping a student improve their research paper section.

Task: {SECTION_PROMPTS.get(sec_key, 'Refine academic quality.')}
Formatting style: {setup.get('format_style','IEEE')}
Research title: {setup.get('title','')}
Research domain: {setup.get('domain','')}
Research objective: {setup.get('objective','')}
Keywords: {setup.get('keywords','')}
{lit_context}

Student's draft:
{user_draft}

Instructions:
- Preserve the student's authorship, voice, and core ideas
- Improve academic quality, clarity, and structure
- Expand where needed but do not completely rewrite
- Use formal academic language
- Do NOT invent citations or claim specific statistics without basis
- Return only the improved section text, no preamble"""

    improved_text = call_gemini_engine(prompt, 1200)
    return jsonify({"improved_text": improved_text})

@app.route("/api/upload_figure", methods=["POST"])
def api_upload_figure():
    if "file" not in request.files:
        return jsonify({"error": "No asset object found in stream."}), 400
        
    file = request.files["file"]
    setup = json.loads(request.form.get("setup", "{}"))
    current_fig_count = int(request.form.get("current_count", "0"))
    
    if file.filename == "":
        return jsonify({"error": "Empty reference package loaded."}), 400
        
    file_bytes = file.read()
    b64_string = base64.b64encode(file_bytes).decode()
    
    prompt = f"""Generate a short academic figure caption (1-2 sentences) for a figure in a {setup.get('domain','')} research paper titled "{setup.get('title','')}".
The figure file is named: {file.filename}
Caption should be formal and describe what this figure likely shows based on its name.
Return only the caption text."""

    caption = call_gemini_engine(prompt, 100)
    latex_block = f"\\begin{{figure}}[h]\n\\centering\n\\includegraphics[width=0.8\\linewidth]{{{file.filename}}}\n\\caption{{{caption}}}\n\\label{{fig:{current_fig_count + 1}}}\n\\end{{figure}}"
    
    return jsonify({
        "figure": {
            "filename": file.filename,
            "b64": b64_string,
            "caption": caption,
            "latex": latex_block,
            "mimetype": file.content_type
        }
    })

@app.route("/api/run_validation", methods=["POST"])
def api_run_validation():
    data = request.json or {}
    setup = data.get("setup", {})
    sections = data.get("sections", {})
    lit_papers = data.get("lit_papers", [])
    
    combined_sections = "\n\n".join(
        f"[{k.upper()}]\n{v}" for k, v in sections.items() if str(v).strip()
    )[:5000]
    
    lit_summaries = "\n".join([f"- {p.get('title','')}: {p.get('findings','')[:150]}" for p in lit_papers[:5]])
    
    prompt = f"""You are an academic peer reviewer. Assess this research paper draft.

Research Title: {setup.get('title','')}
Domain: {setup.get('domain','')}

Uploaded Literature Findings:
{lit_summaries if lit_summaries else "No papers uploaded."}

Paper Sections:
{combined_sections}

Evaluate and return a JSON object with these fields:
{{
  "overall_risk": "Low|Medium|High",
  "hallucination_flags": ["List specific claims that appear unsupported or potentially hallucinated"],
  "weak_sections": ["List section names with quality issues"],
  "strengths": ["List 2-3 strengths of the paper"],
  "recommendations": ["List 3-5 specific improvement recommendations"],
  "reference_note": "Assessment of citation adequacy"
}}
Return only JSON."""

    result = call_gemini_engine(prompt, 1000)
    try:
        clean = re.sub(r"```json|```", "", result).strip()
        parsed = json.loads(clean)
        return jsonify({"validation_result": parsed})
    except Exception:
        return jsonify({"validation_result": {"raw": result}})

@app.route("/api/export_zip", methods=["POST"])
def api_export_zip():
    data = request.json or {}
    setup = data.get("setup", {})
    authors = data.get("authors", [])
    sections = data.get("sections", {})
    citations = data.get("citations", [])
    figures = data.get("figures", [])
    
    latex_content = build_latex_content(setup, authors, sections)
    bib_content = build_bib_content(citations)
    metadata_content = json.dumps(build_metadata_dict(setup, authors, citations, sections), indent=2)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ResearchProject/main.tex", latex_content)
        zf.writestr("ResearchProject/references.bib", bib_content or "% No references")
        zf.writestr("ResearchProject/metadata.json", metadata_content)
        
        for fig in figures:
            try:
                fig_bytes = base64.b64decode(fig["b64"])
                zf.writestr(f"ResearchProject/figures/{fig['filename']}", fig_bytes)
            except Exception:
                pass
                
        zf.writestr("ResearchProject/README.txt",
            "ResearchMate AI Export\n\nTo compile:\n1. Upload to Overleaf (overleaf.com)\n2. Select main.tex as the main file\n3. Compile with pdfLaTeX\n\nNote: Verify all citations and content before submission.")
            
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="ResearchProject.zip"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)