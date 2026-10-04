"""
compliance_watch.fetch_utils
=============================
負責「抓取來源頁面內容 + 判斷是否變動」的工具函式。

設計原則（務必保留）：
    - 任何一個來源抓取失敗，都不可以讓整個排程腳本崩潰。抓取失敗的
      來源要被明確標記為 `fetch_failed`，並進入「本週請人工檢查」清單，
      而不是被默默略過或假裝沒事。
    - 不對「首頁式」的模糊URL（例如只給網域根目錄，沒有指到實際子頁面）
      做自動雜湊比對——那種URL本來就無法代表任何特定頁面的內容，
      強行雜湊比對只會得到「首頁本身天天在變」的雜訊，沒有意義。
      這類來源一律標記為 `manual_only`，每週提醒人工自行查看。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

USER_AGENT = (
    "VeloGuardComplianceWatch/1.0 "
    "(+school-project compliance monitoring bot; contact: see GitHub repo)"
)
REQUEST_TIMEOUT_SEC = 25


@dataclass
class FetchResult:
    """單一來源的抓取結果。"""

    status: str  # "ok" | "fetch_failed" | "manual_only"
    content_hash: Optional[str] = None
    text_excerpt: Optional[str] = None  # 前2000字，供AI比對與人工檢查用
    full_text: Optional[str] = None
    detail: str = ""


def is_generic_homepage_url(url: str) -> bool:
    """判斷這個URL是否只是「網域首頁」等級的模糊連結，無法代表特定頁面。

    背景：知識庫的 `source_pages` 裡，有些荷蘭站的條目因為原始頁面是
    JS渲染的單頁應用（SPA），當初是人工複製貼上頁面文字，URL欄位只
    填了網域根目錄或「語言代碼首頁」（例如 "https://www.giant-bicycles.com/nl"，
    這裡的 "nl" 只是荷蘭語系首頁，不是指向特定子頁面），並非真正指向
    該子頁面的網址。這種URL做雜湊比對沒有意義。

    判斷規則：
        - 完全沒有路徑（網域根目錄），例如 "https://example.com" → True
        - 路徑只有一段，且該段看起來像語言/地區代碼（例如 "nl"、"de"、
          "en-us"）→ True（這是本地化首頁，不是特定子頁面）
        - 其他情況（路徑有實際子頁面slug，例如 "de/termsconditions"、
          "nl/veiligheid/e-bikes"）→ False
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path == "" and not parsed.query:
        return True
    segments = path.split("/")
    if len(segments) == 1 and re.fullmatch(r"[a-z]{2}(-[a-z]{2})?", segments[0], re.IGNORECASE):
        return True
    return False


def _strip_html_tags(html: str) -> str:
    """極簡的HTML去標籤器，避免額外引入bs4以外的重依賴。

    僅用於產生「純文字摘錄」給AI與人工閱讀比對，不追求完美的HTML解析，
    只要能把可讀文字抽出來即可。
    """
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_source(url: str) -> FetchResult:
    """抓取單一URL，回傳抓取結果（含內容雜湊，供之後比對是否變動）。

    Args:
        url: 來源頁面或檔案（含PDF）的URL。

    Returns:
        FetchResult。抓取失敗時 status="fetch_failed"，絕不拋出例外
        中斷呼叫端的流程（呼叫端只需要檢查 status 即可）。
    """
    if is_generic_homepage_url(url):
        return FetchResult(status="manual_only", detail="URL僅指向網域首頁，無法自動比對特定頁面內容")

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return FetchResult(status="fetch_failed", detail=f"HTTP請求失敗: {e}")

    content_type = resp.headers.get("Content-Type", "")

    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        # PDF：直接對原始位元組做雜湊，不嘗試解析文字內容
        # （文字擷取的正確性不影響「是否變動」的判斷，位元組雜湊足夠）
        content_hash = hashlib.sha256(resp.content).hexdigest()
        return FetchResult(
            status="ok",
            content_hash=content_hash,
            text_excerpt="(PDF檔案，未擷取文字內容，僅比對位元組雜湊)",
            full_text=None,
            detail=f"PDF, {len(resp.content)} bytes",
        )

    text = _strip_html_tags(resp.text)

    if len(text) < 80:
        # 內容異常短，很可能是被擋下（例如回傳一個「請啟用JavaScript」
        # 的空殼頁面），視同抓取失敗，避免拿這種假內容去騙AI重新生成chunk。
        return FetchResult(
            status="fetch_failed",
            detail=f"抓到的文字內容異常短（{len(text)}字），可能被阻擋或為JS渲染空殼頁面",
        )

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return FetchResult(
        status="ok",
        content_hash=content_hash,
        text_excerpt=text[:2000],
        full_text=text,
        detail=f"純文字, {len(text)}字",
    )
