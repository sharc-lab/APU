"""Convert docs/PAPER1_DRAFT.md (paper only) to docs/paper1/paper1.tex (IEEEtran conference).
Also prints words per section (HTML comments removed)."""
import re, sys, json

src, out_tex = sys.argv[1], sys.argv[2]
md = open(src, encoding='utf-8').read()

CITE = {
 'Lost in the Middle': 'liu2024lost', 'RULER': 'hsieh2024ruler', 'LongBench': 'bai2024longbench',
 'LLMs Get Lost in Multi-Turn Conversation': 'laban2025lost', 'IFEval': 'zhou2023ifeval',
 'Can LLMs Follow Simple Rules?': 'mu2023rules', 'StreamingLLM': 'xiao2024streamingllm', 'H2O': 'zhang2023h2o',
 'SnapKV': 'li2024snapkv', 'KIVI': 'liu2024kivi', 'KVQuant': 'hooper2024kvquant', 'LLMLingua': 'jiang2023llmlingua',
 'AgentBench': 'liu2024agentbench', 'tau-bench': 'yao2024taubench', 'Gorilla': 'patil2024gorilla',
 'SWE-bench': 'jimenez2024swebench', 'RouteLLM': 'routellm', 'Hybrid LLM': 'ding2024hybrid',
 'FrugalGPT': 'chen2023frugalgpt', 'Self-Verification': 'weng2023selfverification',
 'SelfCheckGPT': 'manakul2023selfcheckgpt', 'vLLM': 'kwon2023vllm', 'MELTing Point': 'laskaridis2024melt',
 'PowerInfer': 'song2024powerinfer', 'LLM in a Flash': 'edge_llm_serving',
 'EdgeWise': 'shi2026edgewise', 'EC2-Bench': 'huang2026ec2bench', 'Minions': 'narayan2025minions',
 'Arch-Router': 'tran2025archrouter',
}
# Citations written without an author-year in the draft: [name; to verify].
CITE_NOYEAR = {'llmfit': 'llmfit', 'Ollama issue #14259': 'ollama14259'}
# Named in the draft but with no verified source, so no bib entry: printed as a visible marker, never \cite.
UNVERIFIED: set = set()
SECNAMES = {'4.1': 'Introduction', '4.2': 'Background and Related Work', '4.3': 'Platforms, Harness and Methodology',
            '4.4': 'Memory-Architecture Boundaries', '4.5': 'Runtime Policy', '4.6': 'Agent Consequences',
            '4.6a': 'The Window Result', '4.6b': 'The Mechanism on AMD and NVIDIA', '4.7': 'Fixes',
            '4.7p': 'Predictor', '4.8': 'Threats to Validity', '4.9': 'Conclusion'}

ph = []
def hold(x):
    ph.append(x)
    return '\x00%d\x01' % (len(ph) - 1)
def restore(t):
    while '\x00' in t:
        t = re.sub(r'\x00(\d+)\x01', lambda m: ph[int(m.group(1))], t)
    return t

def esc(t):
    t = t.replace('\\', r'\textbackslash{}')
    for a, b in [('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]:
        t = t.replace(a, b)
    t = t.replace('+/-', hold(r'$\pm$'))
    t = t.replace('->', hold(r'$\rightarrow$'))
    t = t.replace('>=', hold(r'$\geq$'))
    t = t.replace('<', hold(r'$<$')).replace('>', hold(r'$>$'))
    return t

def secref(m):
    whole = m.group(0)
    return re.sub(r'4\.\d[a-z]?', lambda k: r'\ref{sec:%s}' % k.group(0), whole).replace('Section ', 'Section~').replace('Sections ', 'Sections~')

def inline(t, comments):
    """t: one paragraph joined to a single line. comments: list to collect register ids (for cells/captions)."""
    # HTML comments; merge adjacent ones first
    t = re.sub(r'-->\s*<!--', ' | ', t)
    def com(m):
        body = ' '.join(m.group(1).split())
        if comments is not None:
            comments.append(body)
            return ''
        return hold('%% reg: %s\n' % body)
    # comment followed by punctuation: attach without space
    t = re.sub(r'\s*<!--((?:(?!-->).)*)-->(?=[.,;:)])', lambda m: (com(m)), t)
    t = re.sub(r'\s*<!--((?:(?!-->).)*)-->\s*', lambda m: ((' ' + com(m)) if comments is None else ' '), t)
    t = re.sub(r' +([.,;:)])', r'\1', t) if comments is not None else t
    # citations
    def cite(m):
        name = m.group(1).strip()
        if name not in CITE:
            raise SystemExit('md2tex: citation %r has no key in CITE; add it and its references.bib entry' % name)
        return hold(r'~\cite{%s}' % CITE[name])
    t = re.sub(r'\s*\[[A-Z][^\]]*? \d{4}, ([^\];]+); to verify\]', cite, t)
    def cite_noyear(m):
        name = m.group(1).strip()
        if name in UNVERIFIED:
            return hold(r'~[\textbf{source not verified}]')
        if name not in CITE_NOYEAR:
            raise SystemExit('md2tex: citation %r has no key in CITE_NOYEAR; add it and its references.bib entry' % name)
        return hold(r'~\cite{%s}' % CITE_NOYEAR[name])
    t = re.sub(r'\s*\[([^\]\d][^\];,]*?); to verify\]', cite_noyear, t)
    # code spans
    t = re.sub(r'`([^`]+)`', lambda m: hold(r'\texttt{%s}' % esc(m.group(1))), t)
    # section refs
    t = re.sub(r'Sections? 4\.\d[a-z]?(?:(?:,| and| to) 4\.\d[a-z]?)*', lambda m: hold(secref(m)), t)
    t = re.sub(r'Table R2-gated', lambda m: hold(r'Table~\ref{tab:r2gated}'), t)
    t = re.sub(r'Table 2\b', lambda m: hold(r'Table~\ref{tab:defaults}'), t)
    t = re.sub(r'Table 1\b', lambda m: hold(r'Table~\ref{tab:platforms}'), t)
    t = re.sub(r'Figure (FH1|F\d)', lambda m: hold(r'Fig.~\ref{fig:%s}' % m.group(1)), t)
    t = esc(t)
    t = re.sub(r'"([^"]*)"', lambda m: '``' + m.group(1) + "''", t)
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'\\emph{\1}', t)
    return t

def words(text):
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
    return len(text.split())

lines = md.split('\n')
title = re.search(r'\*\*Working title:\*\* \*(.+?)\*', md).group(1)
abs_m = re.search(r'\*\*Abstract\.\*\*\n\n(.*?)\n\n', md, re.S)
abstract_md = abs_m.group(1)

# body: from '### 4.1' to '## References'
start = md.index('### 4.1 ')
end = md.index('## References (to verify)')
body = md[start:end]

# words per section
wc = {'Abstract': words(abstract_md)}
for m in re.finditer(r'^#{3,4} (4\.\d[a-z]?) [^\n]*\n(.*?)(?=^#{3,4} 4\.|\Z)', body, re.S | re.M):
    sid, txt = m.group(1), m.group(2)
    # exclude figure/table spec blocks from prose count, count them separately
    prose = re.split(r'\n\*\*(?:Figure specs?|Table 1 and figure spec|Table R2-gated and figure spec)\.\*\*', txt)[0]
    wc[sid] = (words(prose), words(txt) - words(prose))

FIGSIZE = {'FH1': ('figure*', r'0.95\textwidth', '4.2cm'), 'F1': ('figure', r'0.95\columnwidth', '3.6cm')}
def figure(fid, spec_md):
    comments = []
    cap = inline(' '.join(spec_md.split()), comments)
    env, w, h = FIGSIZE.get(fid, ('figure', r'0.95\columnwidth', '3.6cm'))
    out = ['%% reg: %s' % c for c in comments]
    out += [r'\begin{%s}[t]' % env, r'\centering',
            r'\fbox{\parbox[c][%s][c]{%s}{\centering\small Figure %s placeholder (not drawn yet)}}' % (h, w, fid),
            r'\caption{%s}' % cap, r'\label{fig:%s}' % fid, r'\end{%s}' % env]
    return '\n'.join(out)

TABLE1 = r"""\begin{table}[t]
\caption{Platforms. Configuration record, not a measured result; driver, runtime and OS versions are filled from the
platform configuration record before submission.}
\label{tab:platforms}
\centering\footnotesize
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.34\columnwidth}>{\raggedright\arraybackslash}p{0.32\columnwidth}>{\raggedright\arraybackslash}p{0.26\columnwidth}@{}}
\toprule
Platform & Memory & Runtime device used \\
\midrule
AMD Ryzen AI Max+ 395 (Strix Halo) APU, Radeon 8060S iGPU & 128 GB unified; firmware-reserved GPU block of 65536 MiB % reg: fh1-amd-boundary-llama-3.3-70b
& iGPU via ROCm % reg: x2-device-detect-mechanism
\\
NVIDIA RTX 4070 Laptop GPU & 8188 MiB discrete VRAM % reg: blade-k1-layer-split
& discrete GPU, layers moved to CPU above 8192 tokens % reg: blade-k1-layer-split
\\
Intel Core Ultra X7 358H (Panther Lake), Arc B390 iGPU & unified; one shared Vulkan heap % reg: A-24-effective-heap-limit
& CPU as shipped; iGPU when enabled % reg: x2-device-detect-mechanism
\\
\bottomrule
\end{tabular}
\end{table}"""

def md_table(rows, caption, label, comments_extra=()):
    comments = list(comments_extra)
    cells = []
    for r in rows:
        cs = [c.strip() for c in r.strip().strip('|').split('|')]
        cells.append([inline(c, comments) for c in cs])
    hdr, data = cells[0], cells[2:]
    n = len(hdr)
    out = ['%% reg: %s' % c for c in dict.fromkeys(comments)]
    wide = label == 'tab:r2gated'
    env = 'table*' if wide else 'table'
    if wide:
        spec = '@{}l' + 'l' * (n - 1) + '@{}'
    else:
        spec = r'@{}>{\raggedright\arraybackslash}p{0.3\columnwidth}>{\raggedright\arraybackslash}p{0.3\columnwidth}>{\raggedright\arraybackslash}p{0.3\columnwidth}@{}'
    out += [r'\begin{%s}[t]' % env, r'\caption{%s}' % caption, r'\label{%s}' % label,
            r'\centering\footnotesize', r'\begin{tabular}{%s}' % spec, r'\toprule',
            ' & '.join(hdr) + r' \\', r'\midrule']
    out += [' & '.join(r) + r' \\' for r in data]
    out += [r'\bottomrule', r'\end{tabular}', r'\end{%s}' % env]
    return '\n'.join(out)

# process body into blocks
blocks = re.split(r'\n\s*\n', body)
tex = []
i = 0
pending_caption = None
while i < len(blocks):
    b = blocks[i].strip('\n')
    i += 1
    if not b.strip() or b.strip() == '---':
        continue
    hm = re.match(r'^(#{3,4}) (4\.\d[a-z]?) ([^\n]*)(?:\n(.*))?$', b, re.S)
    if hm and hm.group(4):
        blocks.insert(i, hm.group(4))
    if hm:
        sid = hm.group(2)
        cmd = 'section' if len(hm.group(1)) == 3 else 'subsection'
        tex.append('\\%s{%s}\\label{sec:%s}' % (cmd, SECNAMES[sid], sid))
        continue
    if re.fullmatch(r'<!--.*?-->', b.strip(), re.S):
        tex.append('% ' + ' '.join(b.strip()[4:-3].split()))
        continue
    if b.startswith('**Figure spec') or b.startswith('**Table 1 and figure spec') or b.startswith('**Table R2-gated and figure spec'):
        continue
    if b.startswith('- **'):
        # spec bullets
        for item in re.split(r'\n(?=- \*\*)', b):
            m = re.match(r'- \*\*(FH1|F\d|Table 1)(?: \(headline\))?\.\*\*\s*(.*)', item, re.S)
            fid, spec = m.group(1), m.group(2)
            if fid == 'Table 1':
                tex.append(TABLE1)
            else:
                tex.append(figure(fid, spec))
        continue
    if b.startswith('Table R2-gated:'):
        comments = []
        pending_caption = (inline(' '.join(b[len('Table R2-gated:'):].split()), comments), comments)
        continue
    if b.startswith('|'):
        rows = b.split('\n')
        if pending_caption:
            cap, cm = pending_caption
            pending_caption = None
            # table note follows
            note = blocks[i].strip() if i < len(blocks) else ''
            if note.startswith('Table note:'):
                i += 1
                ncm = []
                cap = cap + ' ' + inline(' '.join(note[len('Table note:'):].split()), ncm)
                cm = cm + ncm
            tex.append(md_table(rows, 'Gated silent failures per tier and model on the AMD APU. ' + cap, 'tab:r2gated', cm))
        else:
            capb = blocks[i].strip()
            i += 1
            cm = []
            cap = inline(' '.join(re.sub(r'^Table 2:\s*', '', capb).split()), cm)
            tex.append(md_table(rows, cap[0].upper() + cap[1:], 'tab:defaults', cm))
        continue
    if re.match(r'^\d+\. ', b):
        items = re.split(r'\n(?=\d+\. )', b)
        tex.append('\\begin{enumerate}')
        for it in items:
            tex.append('\\item ' + inline(' '.join(re.sub(r'^\d+\. ', '', it).split()), None))
        tex.append('\\end{enumerate}')
        continue
    tex.append(inline(' '.join(b.split()), None))

body_tex = restore('\n\n'.join(tex))
abstract_tex = restore(inline(' '.join(abstract_md.split()), None))

doc = r"""\documentclass[conference,10pt]{IEEEtran}
% Paper 1 draft v1, typeset from docs/PAPER1_DRAFT.md. Register ids are LaTeX comments (% reg: id) and do not print.
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage{textcomp}

\begin{document}

\title{@@TITLE@@}

\author{\IEEEauthorblockN{Anonymous Author(s)}
\IEEEauthorblockA{Submission under double-blind review}}

\maketitle

\begin{abstract}
@@ABS@@
\end{abstract}

@@BODY@@

\typeout{BODYEND page \thepage}
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
""".replace("@@TITLE@@", title).replace("@@ABS@@", abstract_tex).replace("@@BODY@@", body_tex)
open(out_tex, 'w', encoding='utf-8', newline='\n').write(doc)
print(json.dumps(wc))
