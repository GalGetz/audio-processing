from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF using pypdf.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    str
        Concatenated text from all pages (empty if the PDF is image-only/scanned).
    """
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(page_text)
    return "\n".join(parts)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = repo_root / "presentations"
    out_dir = pdf_dir / "_extracted_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in: {pdf_dir}")

    total = 0
    empty = 0
    for pdf_path in pdf_paths:
        text = extract_pdf_text(pdf_path)
        if not text.strip():
            empty += 1
        out_path = out_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        total += 1

    print(f"Extracted {total} PDFs to: {out_dir}")
    print(f"PDFs with no extracted text (likely scanned/image-only): {empty}/{total}")


if __name__ == "__main__":
    main()


