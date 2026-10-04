"""
compliance_watch.check_sources
=============================
每週排程執行的主控腳本。流程：

    1. 讀取 veloguard_multicountry_chunks.json 的 source_pages（品牌政策、
       官方法律條文、第三方物流/電池法規），加上本檔內建的VAT稅率官方頁面清單。
    2. 對每個來源呼叫 fetch_utils.fetch_source() 抓取現在的內容並計算雜湊。
    3. 跟上次執行留下的 snapshot_store.json 比對雜湊，判斷是否有變動。
    4. 對「有變動且抓取成功」的來源，呼叫 ai_rechunk.draft_updated_chunk()
       請AI草擬新版chunk（草稿，非最終版本）。
    5. 產出兩份輸出：
        - compliance_watch/report.md      給人看的本週檢查報告
        - compliance_watch/proposed_chunks.json  給GitHub Action組PR用的機器可讀草稿
    6. 更新 snapshot_store.json（僅更新抓取成功的來源，失敗的來源保留舊雜湊，
       下次繼續重試比對，避免把「暫時抓取失敗」誤判為「內容變回原狀」）。
    7. 以退出碼/GITHUB_OUTPUT告知外部（GitHub Actions）這次是否有需要開PR的內容。

**重要：本腳本絕對不會直接修改 veloguard_multicountry_chunks.json。**
所有AI草擬的內容都只會寫進 proposed_chunks.json，必須經由人工在GitHub PR中
逐筆審核、確認無誤後才手動（或由審核者按下合併）真正更新知識庫檔案。
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from compliance_watch.fetch_utils import fetch_source
from compliance_watch.ai_rechunk import draft_updated_chunk

WATCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = WATCH_DIR.parent
KB_PATH = REPO_ROOT / "veloguard_multicountry_chunks.json"
SNAPSHOT_PATH = WATCH_DIR / "snapshot_store.json"
REPORT_PATH = WATCH_DIR / "report.md"
PROPOSED_CHUNKS_PATH = WATCH_DIR / "proposed_chunks.json"

# ---------------------------------------------------------------------------
# VAT稅率官方來源——這些不存在於veloguard_multicountry_chunks.json的
# source_pages結構中（該結構目前只涵蓋品牌政策/法律條文/第三方規範），
# 但使用者明確要求VAT稅率也要納入每週監控範圍，因此在此另外列出。
# 選用「官方稽徵機關」頁面，而非新聞或第三方懶人包，確保權威性。
# ---------------------------------------------------------------------------
VAT_RATE_SOURCES = [
    {
        "id": "VAT_DE_Official",
        "country": "Germany",
        "title": "Bundesministerium der Finanzen - Umsatzsteuer (德國聯邦財政部 增值稅稅率頁)",
        "url": "https://www.bundesfinanzministerium.de/Web/DE/Themen/Steuern/Steuerarten/Umsatzsteuer/umsatzsteuer.html",
        "source_type": "official_tax_rate",
    },
    {
        "id": "VAT_NL_Official",
        "country": "Netherlands",
        "title": "Belastingdienst - Btw-tarieven (荷蘭稽徵機關 BTW稅率頁)",
        "url": "https://www.belastingdienst.nl/wps/wcm/connect/bldcontentnl/belastingdienst/zakelijk/btw/tarieven_en_vrijstellingen/btw_tarieven",
        "source_type": "official_tax_rate",
    },
]


def _clean_url(raw_url: str) -> str:
    """從source_pages裡可能夾帶備註文字的URL欄位中，取出乾淨的URL。

    例如 "https://www.giant-bicycles.com/nl (下載版本，opgesteld oktober 2016)"
    這種欄位是人工登記時順手加註的，不是合法URL，直接拿去打requests.get()
    會因為含空白/括號而出錯。這裡只取第一個空白字元前的部分。
    """
    return raw_url.strip().split()[0] if raw_url.strip() else raw_url


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _iter_all_sources(kb: dict):
    """把 source_pages（依國家分組的list）攤平成單一序列，每筆補上country欄位；
    再串接VAT_RATE_SOURCES。統一格式方便後續迴圈處理。
    """
    source_pages = kb.get("source_pages", {})
    for country, pages in source_pages.items():
        for page in pages:
            entry = dict(page)
            entry.setdefault("country", country)
            yield entry
    for entry in VAT_RATE_SOURCES:
        yield dict(entry)


def _find_existing_chunk(kb: dict, source_id: str, country: str) -> Optional[dict]:
    """在既有chunks中找出對應這個來源頁面的chunk（用source_page欄位比對）。

    找不到就回傳None——這是正常情況（例如VAT稅率本身不是以chunk形式儲存，
    或這是全新的來源），呼叫端要能處理None，不能假設一定找得到。
    """
    for chunk in kb.get("chunks", []):
        if chunk.get("source_page") == source_id and chunk.get("country") == country:
            return chunk
    return None


def run(api_key: Optional[str] = None) -> bool:
    """執行一次完整的檢查流程。回傳True代表「這次有內容需要人工審核/開PR」。"""

    kb = _load_json(KB_PATH, {})
    if not kb:
        print(f"[錯誤] 找不到或無法讀取知識庫檔案: {KB_PATH}", file=sys.stderr)
        return False

    snapshots: dict = _load_json(SNAPSHOT_PATH, {})
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    changed_with_draft: list[dict] = []
    manual_review_needed: list[dict] = []
    fetch_failures: list[dict] = []
    unchanged_count = 0

    for source in _iter_all_sources(kb):
        source_id = source["id"]
        country = source.get("country", "?")
        url = _clean_url(source["url"])
        snapshot_key = f"{country}::{source_id}"

        result = fetch_source(url)

        if result.status == "manual_only":
            manual_review_needed.append(
                {
                    "source_id": source_id,
                    "country": country,
                    "url": url,
                    "reason": result.detail,
                }
            )
            continue

        if result.status == "fetch_failed":
            fetch_failures.append(
                {
                    "source_id": source_id,
                    "country": country,
                    "url": url,
                    "reason": result.detail,
                }
            )
            # 抓取失敗時保留舊雜湊，不覆寫snapshot——避免下次比對時
            # 誤以為「內容變回原狀」而漏掉真正的變動。
            continue

        prev_hash = snapshots.get(snapshot_key, {}).get("content_hash")
        if prev_hash == result.content_hash:
            unchanged_count += 1
            snapshots[snapshot_key] = {
                "content_hash": result.content_hash,
                "last_checked": now_iso,
                "last_changed": snapshots.get(snapshot_key, {}).get("last_changed"),
            }
            continue

        # 內容有變動（或這是第一次記錄此來源）
        is_first_time = prev_hash is None
        existing_chunk = _find_existing_chunk(kb, source_id, country)

        draft = draft_updated_chunk(
            existing_chunk=existing_chunk,
            new_full_text=result.full_text or "",
            source_meta=source,
            api_key=api_key,
        )

        snapshots[snapshot_key] = {
            "content_hash": result.content_hash,
            "last_checked": now_iso,
            "last_changed": now_iso,
        }

        changed_with_draft.append(
            {
                "source_id": source_id,
                "country": country,
                "url": url,
                "is_first_time_seen": is_first_time,
                "draft": asdict(draft),
            }
        )

    _save_json(SNAPSHOT_PATH, snapshots)
    _write_report(
        now_iso, changed_with_draft, manual_review_needed, fetch_failures, unchanged_count
    )
    _save_json(
        PROPOSED_CHUNKS_PATH,
        {
            "generated_at": now_iso,
            "note": "此檔案內容為AI草擬，一律待人工審核，絕不可直接覆蓋knowledge base",
            "proposed_changes": changed_with_draft,
        },
    )

    has_actionable_content = len(changed_with_draft) > 0
    print(f"檢查完成：{len(changed_with_draft)}筆變動需審核、"
          f"{len(manual_review_needed)}筆僅供人工查看、"
          f"{len(fetch_failures)}筆抓取失敗、{unchanged_count}筆無變動")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"changes_detected={'true' if has_actionable_content else 'false'}\n")

    return has_actionable_content


def _write_report(
    now_iso: str,
    changed_with_draft: list[dict],
    manual_review_needed: list[dict],
    fetch_failures: list[dict],
    unchanged_count: int,
) -> None:
    lines = [
        "# VeloGuard 合規來源每週檢查報告",
        "",
        f"執行時間（UTC）：{now_iso}",
        "",
        "> ⚠️ 本報告中所有「AI草擬」的內容都尚未經過人工審核，"
        "**絕對不可直接視為知識庫的最終正確版本**。請對照下方每一筆的"
        "「原始文字全文」與AI摘要，逐筆確認後才能採用。",
        "",
        f"- 偵測到需審核的變動：**{len(changed_with_draft)}** 筆",
        f"- 僅能人工查看（無法自動比對）的來源：{len(manual_review_needed)} 筆",
        f"- 這次抓取失敗的來源：{len(fetch_failures)} 筆",
        f"- 內容無變動：{unchanged_count} 筆",
        "",
    ]

    if changed_with_draft:
        lines.append("## 🔍 偵測到變動，需要人工審核")
        lines.append("")
        for item in changed_with_draft:
            draft = item["draft"]
            lines.append(f"### {item['country']} / `{item['source_id']}`")
            lines.append(f"- 來源網址：{item['url']}")
            if item["is_first_time_seen"]:
                lines.append("- ⚠️ 這是本系統第一次記錄此來源（可能是全新加入的監控項目）")
            if not draft["ok"]:
                lines.append(f"- ❌ AI草擬失敗：{draft['error_detail']}")
                lines.append("- **請人工直接查看來源網址並手動更新對應chunk。**")
            else:
                lines.append(f"- AI摘要的變動說明：{draft['change_summary']}")
                lines.append(
                    f"- AI判斷是否為實質內容變動：{'是' if draft['content_meaningfully_changed'] else '否（可能僅為排版/措辭調整）'}"
                )
                if draft.get("needs_human_judgment"):
                    lines.append(f"- ⚠️ AI標註需人工判斷之處：{draft['needs_human_judgment']}")
                lines.append(f"- AI信心說明：{draft['confidence_note']}")
                lines.append("- 草擬的新chunk內容請見 `proposed_chunks.json` 同批次資料，"
                              "並務必對照原始全文核對逐字內容是否被竄改。")
            lines.append("")

    if manual_review_needed:
        lines.append("## 👀 僅能人工查看的來源（本週建議手動檢查一次）")
        lines.append("")
        for item in manual_review_needed:
            lines.append(f"- {item['country']} / `{item['source_id']}`：{item['url']}　"
                          f"（原因：{item['reason']}）")
        lines.append("")

    if fetch_failures:
        lines.append("## ⚠️ 本次抓取失敗的來源")
        lines.append("")
        for item in fetch_failures:
            lines.append(f"- {item['country']} / `{item['source_id']}`：{item['url']}　"
                          f"（原因：{item['reason']}）")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    had_changes = run()
    sys.exit(0)
