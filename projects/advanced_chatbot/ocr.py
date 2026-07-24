"""OCR providers for scanned PDFs / images → markdown text."""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod

from projects.advanced_chatbot.config import AdvancedChatConfig, load_config

OCR_PROMPT = (
    "Extract ALL readable text from this document image. "
    "Preserve headings, lists, and tables as Markdown. "
    "Return only the Markdown text, no commentary."
)


class OCRProvider(ABC):
    @abstractmethod
    def extract_markdown(self, *, content: bytes, filename: str, mime_type: str) -> str:
        raise NotImplementedError


class PassthroughTextOCR(OCRProvider):
    """Decode UTF-8 text files without a vision model."""

    def extract_markdown(self, *, content: bytes, filename: str, mime_type: str) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{filename} is not UTF-8 text. Enable OCR for images/PDFs."
            ) from exc


class PdfTextOCR(OCRProvider):
    """Extract embedded text from PDFs (no vision)."""

    def extract_markdown(self, *, content: bytes, filename: str, mime_type: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf for PDF text extraction: pip install pypdf") from exc

        reader = PdfReader(io.BytesIO(content))
        pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"## Page {index}\n\n{text}")
        joined = "\n\n".join(pages).strip()
        if not joined:
            raise ValueError(
                f"{filename} has no extractable text (likely a scan)."
            )
        return joined


class TesseractOCR(OCRProvider):
    """Local classic OCR — excellent for clean document screenshots on Mac/CPU."""

    def extract_markdown(self, *, content: bytes, filename: str, mime_type: str) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Install pytesseract + pillow: pip install pytesseract pillow"
            ) from exc

        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            try:
                return PdfTextOCR().extract_markdown(
                    content=content, filename=filename, mime_type=mime_type
                )
            except ValueError:
                content, mime_type = _pdf_first_page_png(content)

        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image).strip()
        if not text:
            raise RuntimeError(f"Tesseract found no text in {filename}")
        return text


class OllamaVisionOCR(OCRProvider):
    """Local multimodal OCR via Ollama (optional; quality varies by model)."""

    def __init__(self, config: AdvancedChatConfig | None = None) -> None:
        self.config = config or load_config()

    def extract_markdown(self, *, content: bytes, filename: str, mime_type: str) -> str:
        import httpx

        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            try:
                return PdfTextOCR().extract_markdown(
                    content=content, filename=filename, mime_type=mime_type
                )
            except ValueError:
                content, mime_type = _pdf_first_page_png(content)

        b64 = base64.b64encode(content).decode("ascii")
        model = self.config.ollama_vision_model
        url = self.config.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": OCR_PROMPT,
                    "images": [b64],
                }
            ],
        }
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=payload)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Ollama vision OCR failed ({response.status_code}): {response.text[:400]}. "
                    f"Pull the model: ollama pull {model}"
                )
            data = response.json()
        text = str((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise RuntimeError(f"Ollama vision returned empty OCR for {filename}")
        return text


class DeepSeekHTTPOCR(OCRProvider):
    """Self-hosted DeepSeek-OCR OpenAI-compatible endpoint (GPU recommended)."""

    def __init__(self, config: AdvancedChatConfig | None = None) -> None:
        self.config = config or load_config()
        if not self.config.deepseek_ocr_base_url:
            raise RuntimeError(
                "Set DEEPSEEK_OCR_BASE_URL to your DeepSeek-OCR OpenAI-compatible endpoint"
            )

    def extract_markdown(self, *, content: bytes, filename: str, mime_type: str) -> str:
        import httpx

        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            try:
                return PdfTextOCR().extract_markdown(
                    content=content, filename=filename, mime_type=mime_type
                )
            except ValueError:
                content, mime_type = _pdf_first_page_png(content)

        b64 = base64.b64encode(content).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        headers = {"Content-Type": "application/json"}
        if self.config.deepseek_ocr_api_key:
            headers["Authorization"] = f"Bearer {self.config.deepseek_ocr_api_key}"

        payload = {
            "model": self.config.deepseek_ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8192,
        }
        url = self.config.deepseek_ocr_base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OCR response: {data}") from exc


class AutoOCR(OCRProvider):
    """Try PDF text → Tesseract → Ollama vision → DeepSeek HTTP."""

    def __init__(self, config: AdvancedChatConfig | None = None) -> None:
        self.config = config or load_config()

    def extract_markdown(self, *, content: bytes, filename: str, mime_type: str) -> str:
        errors: list[str] = []
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            try:
                return PdfTextOCR().extract_markdown(
                    content=content, filename=filename, mime_type=mime_type
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"pdf_text: {exc}")

        for provider in (TesseractOCR(), OllamaVisionOCR(self.config)):
            try:
                return provider.extract_markdown(
                    content=content, filename=filename, mime_type=mime_type
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(provider).__name__}: {exc}")

        if self.config.deepseek_ocr_base_url:
            try:
                return DeepSeekHTTPOCR(self.config).extract_markdown(
                    content=content, filename=filename, mime_type=mime_type
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"deepseek: {exc}")

        raise RuntimeError(
            "All OCR providers failed. Install tesseract (`brew install tesseract`) "
            "or set DEEPSEEK_OCR_BASE_URL. Details: " + " | ".join(errors)
        )


def _pdf_first_page_png(content: bytes) -> tuple[bytes, str]:
    """Render first PDF page to PNG when the PDF is image-only."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError(
            "Scanned PDF needs pypdfium2 to render pages for vision OCR: "
            "pip install pypdfium2"
        ) from exc

    pdf = pdfium.PdfDocument(content)
    if len(pdf) == 0:
        raise ValueError("PDF has no pages")
    page = pdf[0]
    bitmap = page.render(scale=2)
    pil_image = bitmap.to_pil()
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def get_ocr_provider(config: AdvancedChatConfig | None = None) -> OCRProvider:
    cfg = config or load_config()
    provider = cfg.ocr_provider
    if provider in {"deepseek", "deepseek_http"}:
        return DeepSeekHTTPOCR(cfg)
    if provider in {"ollama", "ollama_vision"}:
        return OllamaVisionOCR(cfg)
    if provider in {"tesseract", "local"}:
        return TesseractOCR()
    if provider == "pdf_text":
        return PdfTextOCR()
    if provider == "auto":
        return AutoOCR(cfg)
    return PassthroughTextOCR()
