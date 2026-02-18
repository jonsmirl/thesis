#!/usr/bin/env python3
"""Prefix all \label, \ref, \eqref, \bibitem, \cite keys per chapter to avoid collisions."""

import re
import sys

def prefix_file(filepath, prefix):
    with open(filepath, 'r') as f:
        text = f.read()

    original = text

    # 1. Prefix \label{key} -> \label{prefix:key}
    text = re.sub(
        r'\\label\{([^}]+)\}',
        lambda m: f'\\label{{{prefix}:{m.group(1)}}}',
        text
    )

    # 2. Prefix \ref{key} -> \ref{prefix:key}
    text = re.sub(
        r'\\ref\{([^}]+)\}',
        lambda m: f'\\ref{{{prefix}:{m.group(1)}}}',
        text
    )

    # 3. Prefix \eqref{key} -> \eqref{prefix:key}
    text = re.sub(
        r'\\eqref\{([^}]+)\}',
        lambda m: f'\\eqref{{{prefix}:{m.group(1)}}}',
        text
    )

    # 4. Prefix \autoref{key} -> \autoref{prefix:key}
    text = re.sub(
        r'\\autoref\{([^}]+)\}',
        lambda m: f'\\autoref{{{prefix}:{m.group(1)}}}',
        text
    )

    # 5. Prefix \bibitem{key} -> \bibitem{prefix:key}
    text = re.sub(
        r'\\bibitem\{([^}]+)\}',
        lambda m: f'\\bibitem{{{prefix}:{m.group(1)}}}',
        text
    )

    # 6. Prefix \bibcite{key} -> \bibcite{prefix:key} (in .aux files, not needed here)

    # 7. Prefix \cite{key1,key2,...} -> \cite{prefix:key1,prefix:key2,...}
    def prefix_cite_keys(m):
        keys = m.group(1).split(',')
        prefixed = ','.join(f'{prefix}:{k.strip()}' for k in keys)
        return f'\\cite{{{prefixed}}}'

    text = re.sub(r'\\cite\{([^}]+)\}', prefix_cite_keys, text)

    if text != original:
        with open(filepath, 'w') as f:
            f.write(text)
        # Count changes
        labels = len(re.findall(r'\\label\{' + prefix + ':', text))
        refs = len(re.findall(r'\\ref\{' + prefix + ':', text))
        eqrefs = len(re.findall(r'\\eqref\{' + prefix + ':', text))
        bibitems = len(re.findall(r'\\bibitem\{' + prefix + ':', text))
        cites = len(re.findall(r'\\cite\{[^}]*' + prefix + ':', text))
        print(f"  {filepath}: {labels} labels, {refs} refs, {eqrefs} eqrefs, {bibitems} bibitems, {cites} cites")
    else:
        print(f"  {filepath}: no changes needed")

chapters = [
    ('2_ces_triple_role/CES_Triple_Role.tex', 'ces'),
    ('3_complementary_heterogeneity/Complementary_Heterogeneity.tex', 'ch'),
    ('4_endogenous_decentralization/Endogenous_Decentralization.tex', 'ed'),
    ('5_mesh_economy/Mesh_Economy.tex', 'me'),
    ('6_settlement_feedback/Settlement_Feedback.tex', 'sf'),
]

for relpath, prefix in chapters:
    filepath = relpath
    print(f"Processing {prefix}: {relpath}")
    prefix_file(filepath, prefix)
    print()
