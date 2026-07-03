"""Structure-aware document chunking — AstrBot inspired.

AstrBot uses a RecursiveCharacterChunker with separator priority
(\\n\\n > \\n > 。 > ， > . > space > char) and a MarkdownChunker that
respects heading hierarchy. This module brings the same to materials science:

Chunkers:
  - RecursiveCharacterChunker: AstrBot-style recursive split with priority separators
  - MarkdownChunker: Split by headings (##/###), preserve hierarchy in metadata
  - StructureChunker: For CIF/POSCAR files — split by data_ blocks / coordinate sections
  - AutoChunker: Picks the right chunker based on file type/content

All chunkers are async (matching AstrBot pattern) and return Chunk objects
with text + metadata (source, heading_path, chunk_index, element_type).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("huginn.chunker")

@dataclass
class Chunk:
    """A single chunk of a document."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Common metadata keys:
    #   source: original filename
    #   heading_path: list of headings (e.g., ["Methods", "SCF Convergence"])
    #   chunk_index: 0-based position in the document
    #   element_type: "text" | "code" | "table" | "figure_caption" | "structure"
    #   structure_info: for CIF — space_group, lattice_params, etc.


class BaseChunker:
    """Base class for all chunkers."""
    
    async def chunk(self, text: str, **kwargs) -> list[Chunk]:
        ...
    
    def _merge_small_chunks(
        self, chunks: list[Chunk], min_size: int = 100
    ) -> list[Chunk]:
        """Merge chunks smaller than min_size into the previous chunk."""
        if not chunks:
            return []
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk.text) < min_size and merged:
                merged[-1].text += "\n" + chunk.text
            else:
                merged.append(chunk)
        return merged


class RecursiveCharacterChunker(BaseChunker):
    """AstrBot-style recursive character chunker.
    
    Uses a priority-ordered list of separators. Tries the highest-priority
    separator first; if chunks are still too large, recursively splits with
    the next separator. Preserves overlap between adjacent chunks.
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Triple newline (major section break)
        "\n\n",    # Double newline (paragraph break)
        "\n",      # Single newline
        "。",      # Chinese period
        "．",      # Japanese period
        ". ",      # English period + space
        ", ",      # Comma + space
        " ",       # Space
        "",        # Character-level (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    async def chunk(self, text: str, **kwargs) -> list[Chunk]:
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        overlap = kwargs.get("overlap", self.overlap)
        
        if len(text) <= chunk_size:
            return [Chunk(text=text, metadata={"chunk_index": 0})]
        
        chunks = self._split_text(text, chunk_size, overlap, 0)
        return [
            Chunk(text=c, metadata={"chunk_index": i, "separator": "recursive"})
            for i, c in enumerate(chunks)
        ]
    
    def _split_text(
        self, text: str, chunk_size: int, overlap: int, sep_idx: int
    ) -> list[str]:
        """Recursively split text using separator priority."""
        if len(text) <= chunk_size:
            return [text]
        
        if sep_idx >= len(self.separators):
            # Last resort: character-level split
            return self._char_split(text, chunk_size, overlap)
        
        sep = self.separators[sep_idx]
        
        if sep == "":
            return self._char_split(text, chunk_size, overlap)
        
        parts = text.split(sep)
        
        # If separator not found, try next one
        if len(parts) == 1:
            return self._split_text(text, chunk_size, overlap, sep_idx + 1)
        
        # Merge parts into chunks of ~chunk_size
        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                # Start new chunk with overlap
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + sep + part
                else:
                    current = part
            else:
                current = candidate
        if current:
            chunks.append(current)
        
        # If any chunk is still too large, recursively split with next separator
        final = []
        for chunk in chunks:
            if len(chunk) > chunk_size * 1.5:
                final.extend(self._split_text(chunk, chunk_size, overlap, sep_idx + 1))
            else:
                final.append(chunk)
        
        return final
    
    def _char_split(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Character-level split (last resort)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks


class MarkdownChunker(BaseChunker):
    """Split markdown by headings, preserving hierarchy in metadata.
    
    AstrBot's MarkdownChunker respects heading levels. This implementation
    tracks the heading path (e.g., ["Methods", "SCF Convergence"]) as metadata.
    """
    
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(self, chunk_size: int = 800, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    async def chunk(self, text: str, **kwargs) -> list[Chunk]:
        # Find all heading positions
        headings = list(self.HEADING_RE.finditer(text))
        
        if not headings:
            # No headings, fall back to recursive character chunking
            rc = RecursiveCharacterChunker(self.chunk_size, self.overlap)
            return await rc.chunk(text, **kwargs)
        
        chunks = []
        heading_path: list[str] = []
        
        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            # Update heading path
            while len(heading_path) >= level:
                heading_path.pop()
            heading_path.append(title)
            
            # Content from this heading to the next
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[start:end].strip()
            
            if not content:
                continue
            
            # If content is too large, sub-chunk it
            if len(content) > self.chunk_size:
                rc = RecursiveCharacterChunker(self.chunk_size, self.overlap)
                sub_chunks = await rc.chunk(content)
                for j, sc in enumerate(sub_chunks):
                    sc.metadata.update({
                        "heading_path": list(heading_path),
                        "chunk_index": len(chunks),
                        "element_type": "text",
                        "heading_level": level,
                        "heading_title": title,
                    })
                    chunks.append(sc)
            else:
                chunks.append(Chunk(
                    text=content,
                    metadata={
                        "heading_path": list(heading_path),
                        "chunk_index": len(chunks),
                        "element_type": "text",
                        "heading_level": level,
                        "heading_title": title,
                    },
                ))
        
        return self._merge_small_chunks(chunks)


class StructureChunker(BaseChunker):
    """Chunker for crystallographic structure files (CIF, POSCAR).
    
    Extracts structural metadata (space group, lattice parameters, atom count)
    and chunks by logical sections (header, symmetry, positions, etc.).
    """
    
    CIF_BLOCK_RE = re.compile(r'(data_\w+|loop_|_cell_[a-z_]+|_symmetry_[a-z_]+|_atom_[a-z_]+)', re.IGNORECASE)
    
    async def chunk(self, text: str, **kwargs) -> list[Chunk]:
        # Detect format
        if text.strip().startswith("data_"):
            return self._chunk_cif(text)
        elif "Direct" in text or "Cartesian" in text or "Selective dynamics" in text:
            return self._chunk_poscar(text)
        else:
            # Unknown format, fall back to recursive
            rc = RecursiveCharacterChunker()
            return await rc.chunk(text, **kwargs)
    
    def _chunk_cif(self, text: str) -> list[Chunk]:
        """Chunk a CIF file by logical sections."""
        chunks = []
        
        # Extract metadata
        structure_info = {}
        for match in re.finditer(r'_symmetry_space_group_name_H-M\s+[\'"]?([^\'"\n]+)[\'"]?', text):
            structure_info["space_group"] = match.group(1).strip()
            break
        for match in re.finditer(r'_cell_length_a\s+([\d.]+)', text):
            structure_info["lattice_a"] = float(match.group(1))
            break
        for match in re.finditer(r'_cell_length_b\s+([\d.]+)', text):
            structure_info["lattice_b"] = float(match.group(1))
            break
        for match in re.finditer(r'_cell_length_c\s+([\d.]+)', text):
            structure_info["lattice_c"] = float(match.group(1))
            break
        
        # Count atoms in atom_site loop
        atom_count = len(re.findall(r'_atom_site_fract_x', text))
        if atom_count > 0:
            # Count actual atom data lines after the loop
            loop_match = re.search(r'loop_.*?_atom_site_fract_x.*?\n(.*?)(?:\n\n|\nloop_|\Z)', text, re.DOTALL)
            if loop_match:
                atom_lines = [l for l in loop_match.group(1).strip().splitlines() if l.strip() and not l.startswith('_')]
                structure_info["atom_count"] = len(atom_lines)
        
        # Split by data_ blocks and loop_ sections
        sections = re.split(r'(?=data_|loop_)', text)
        sections = [s.strip() for s in sections if s.strip()]
        
        for i, section in enumerate(sections):
            chunks.append(Chunk(
                text=section,
                metadata={
                    "chunk_index": i,
                    "element_type": "structure",
                    "format": "cif",
                    "structure_info": structure_info,
                }
            ))
        
        return chunks if chunks else [Chunk(text=text, metadata={"format": "cif", "element_type": "structure"})]
    
    def _chunk_poscar(self, text: str) -> list[Chunk]:
        """Chunk a POSCAR file by logical sections."""
        lines = text.splitlines()
        if len(lines) < 2:
            return [Chunk(text=text, metadata={"format": "poscar", "element_type": "structure"})]
        
        structure_info = {"comment": lines[0].strip(), "scale": lines[1].strip()}
        
        # Lattice vectors (lines 2-5)
        if len(lines) >= 5:
            lattice = []
            for i in range(2, 5):
                vals = lines[i].split()
                if len(vals) >= 3:
                    try:
                        lattice.append([float(v) for v in vals[:3]])
                    except ValueError:
                        pass
            if lattice:
                structure_info["lattice_vectors"] = lattice
        
        chunks = [
            Chunk(
                text=text,
                metadata={
                    "chunk_index": 0,
                    "element_type": "structure",
                    "format": "poscar",
                    "structure_info": structure_info,
                }
            )
        ]
        return chunks


class AutoChunker(BaseChunker):
    """Auto-detect file type and pick the right chunker.
    
    Usage:
        chunker = AutoChunker()
        chunks = await chunker.chunk(text, filename="paper.md")
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    async def chunk(self, text: str, **kwargs) -> list[Chunk]:
        filename = kwargs.get("filename", "")
        
        # Detect by extension
        if filename.endswith(".cif"):
            chunker = StructureChunker()
        elif filename.endswith(".md") or filename.endswith(".markdown"):
            chunker = MarkdownChunker(self.chunk_size, self.overlap)
        elif filename.endswith(".poscar") or filename.endswith(".contcar") or filename.endswith(".vasp"):
            chunker = StructureChunker()
        else:
            # Detect by content
            if text.strip().startswith("data_") and "_cell_" in text:
                chunker = StructureChunker()
            elif re.search(r'^#{1,6}\s+', text, re.MULTILINE):
                chunker = MarkdownChunker(self.chunk_size, self.overlap)
            else:
                chunker = RecursiveCharacterChunker(self.chunk_size, self.overlap)
        
        chunks = await chunker.chunk(text, **kwargs)
        
        # Add source filename to metadata
        for chunk in chunks:
            chunk.metadata.setdefault("source", filename)
        
        return chunks


__all__ = [
    "Chunk",
    "BaseChunker",
    "RecursiveCharacterChunker",
    "MarkdownChunker",
    "StructureChunker",
    "AutoChunker",
]
