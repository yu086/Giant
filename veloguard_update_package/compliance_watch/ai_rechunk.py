"""
compliance_watch.ai_rechunk
=============================
負責「當某個來源頁面內容有變動時，請AI（Claude API）幫忙草擬新版chunk」的模組。

**核心設計原則（這是整個模組存在的理由，請勿弱化）：**

    本專案先前在人工比對Gemini產出的條文時，發現過一次AI「靜默置換原文」
    的事故（AI在轉述法律條文時，擅自把爭議文字換成它自己覺得更合理的說法，
    而不是逐字保留原文）。這件事直接證明：**絕對不能讓AI自動生成的內容
    未經人工覆核就上線**，尤其是牽涉法律條文、品牌政策原文這種「一字之差
    就可能造成法律責任」的內容。

    因此本模組所有輸出都：
        1. 明確要求AI「逐字」保留原文摘錄（original_text_snippet），
           不可意譯、不可修飾、不可「幫忙修正看起來的錯字」。
        2. 一律標記 verification_status = "ai_drafted_pending_human_review"，
           永遠不會自動標記為已驗證。
        3. 附上完整的原始抓取文字（full_text_for_review）供人工比對，
           讓Reviewer可以自己核對AI是否老實逐字引用，不用只信任AI的話。
        4. 遇到解析失敗或AI回傳格式不對時，直接回報錯誤，絕不用猜測的內容
           填補，也絕不讓錯誤內容偽裝成正常輸出混進PR。

    這個模組的輸出**只能**進入「待人工審核」的草稿檔案，
    **絕對不能**被任何自動化流程直接寫回 veloguard_multicountry_chunks.json
    或直接部署上線。是否採用，必須由人工在GitHub PR中逐行確認。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False


MODEL_NAME = "claude-sonnet-4-5"  # 可依需要調整；建議使用穩定版本而非latest別名，避免行為漂移

# 系統提示：把「絕對逐字、禁止竄改」的規則放在最前面且重複強調，
# 因為這正是先前Gemini事故的核心教訓。
_SYSTEM_PROMPT = """你是VeloGuard跨境合規知識庫的「草稿助手」，職責是在偵測到來源頁面文字有變動時，
根據新抓取到的網頁純文字，草擬一筆更新後的知識庫chunk。

【最重要、不可違反的規則】
1. `original_text_snippet` 欄位必須是從提供的原始文字「逐字複製」的片段，
   一個字都不能改、不能意譯、不能「順一下語氣」、不能修正你以為的錯字或格式問題。
   如果原文有語法怪異、拼字看起來像錯誤、或條號寫法不一致，都要原封不動照抄。
2. 絕對禁止「靜默置換」原文——也就是絕對不能在你認為某段文字有爭議、不合理、
   或政治不正確時，自己換一個說法。你的工作是「如實記錄現況」，不是「修正現況」。
3. 如果你判斷這個頁面的變動只是排版/措辭上的細微調整，與現有chunk在法律意義上
   沒有實質差異，請明確在 `change_summary` 中說明「僅格式變動，無實質內容差異」，
   不要為了「有事可做」而誇大變動的重要性。
4. 如果原始文字裡有你無法確定如何分類的內容（例如看起來像新條款，但你不確定
   對應到哪個既有topic），請在 `needs_human_judgment` 欄位寫明原因，而不是自己
   硬套一個topic分類。
5. 你的輸出只是「草稿」，一定會有人工法律/合規人員逐字覆核後才會採用。
   所以你不需要、也不應該假裝有100%把握——如實標註你的信心程度與疑慮。

【輸出格式】
你必須只輸出一個JSON物件（不要有其他文字、不要用markdown code block包起來），格式如下：
{
  "change_summary": "這次頁面變動的簡短說明（繁體中文）",
  "content_meaningfully_changed": true或false,
  "proposed_chunk": {
    "chunk_id": "沿用原chunk_id（若是全新條款則標記為 NEW_<來源id>_<序號>）",
    "topic": "...",
    "product_scope": "...",
    "jurisdiction": "...",
    "source_type": "...",
    "source_page": "...",
    "official_article_ref": "...",
    "original_text_snippet": "從原文逐字複製，不可修改",
    "summary_zh": "繁體中文摘要說明這段規定的意思",
    "operational_rule": "給客服/營運人員的具體執行準則",
    "country": "..."
  },
  "needs_human_judgment": "若有不確定之處，說明原因；沒有則填 null",
  "confidence_note": "你對這份草稿的信心程度與理由（繁體中文，誠實表達，不要過度自信）"
}
"""


@dataclass
class RechunkDraft:
    """AI草擬結果（僅供人工審核，不可直接採用）。"""

    ok: bool
    chunk_id_hint: str
    change_summary: str = ""
    content_meaningfully_changed: bool = False
    proposed_chunk: Optional[dict] = None
    needs_human_judgment: Optional[str] = None
    confidence_note: str = ""
    verification_status: str = "ai_drafted_pending_human_review"
    full_text_for_review: str = ""
    error_detail: str = ""
    raw_model_output: str = ""


def _build_user_prompt(
    existing_chunk: Optional[dict],
    new_full_text: str,
    source_meta: dict,
) -> str:
    existing_chunk_json = (
        json.dumps(existing_chunk, ensure_ascii=False, indent=2)
        if existing_chunk
        else "（目前知識庫中沒有對應此來源的既有chunk，這可能是全新內容）"
    )
    # 只取前8000字給AI，避免超長頁面塞爆context；
    # 但完整全文仍會存進 full_text_for_review 供人工比對。
    truncated_text = new_full_text[:8000]
    truncation_note = "" if len(new_full_text) <= 8000 else "\n\n（注意：原文過長，此處僅顯示前8000字，人工審核時請查看完整原文）"

    return f"""【來源頁面資訊】
{json.dumps(source_meta, ensure_ascii=False, indent=2)}

【知識庫中既有的對應chunk（若有）】
{existing_chunk_json}

【本次重新抓取到的頁面純文字內容】
{truncated_text}{truncation_note}

請依照系統提示的規則與JSON格式，草擬更新後的chunk。"""


def draft_updated_chunk(
    existing_chunk: Optional[dict],
    new_full_text: str,
    source_meta: dict,
    api_key: Optional[str] = None,
) -> RechunkDraft:
    """呼叫Claude API，針對單一有變動的來源頁面草擬更新後的chunk。

    Args:
        existing_chunk: 知識庫中目前對應這個來源頁面的chunk（沒有則傳None）。
        new_full_text: 這次重新抓取到的頁面純文字全文（來自fetch_utils.fetch_source）。
        source_meta: 來源頁面的metadata（id、title、url、country等），會原封不動附給AI參考。
        api_key: Anthropic API金鑰；未提供時讀取環境變數 ANTHROPIC_API_KEY。

    Returns:
        RechunkDraft。任何失敗（缺套件、缺金鑰、API錯誤、JSON解析失敗）都會回傳
        ok=False並附上error_detail，絕不拋出例外中斷呼叫端流程，也絕不用猜測內容
        填補失敗的結果。
    """
    chunk_id_hint = (existing_chunk or {}).get("chunk_id") or source_meta.get("id", "UNKNOWN")

    if not _ANTHROPIC_AVAILABLE:
        return RechunkDraft(
            ok=False,
            chunk_id_hint=chunk_id_hint,
            full_text_for_review=new_full_text,
            error_detail="缺少 anthropic 套件，請先執行: pip install anthropic",
        )

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        return RechunkDraft(
            ok=False,
            chunk_id_hint=chunk_id_hint,
            full_text_for_review=new_full_text,
            error_detail="找不到 ANTHROPIC_API_KEY（環境變數未設定，或未以api_key參數傳入）",
        )

    client = anthropic.Anthropic(api_key=resolved_key)
    user_prompt = _build_user_prompt(existing_chunk, new_full_text, source_meta)

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:  # noqa: BLE001 - 任何API層錯誤都要優雅回報，不可讓排程腳本崩潰
        return RechunkDraft(
            ok=False,
            chunk_id_hint=chunk_id_hint,
            full_text_for_review=new_full_text,
            error_detail=f"呼叫Anthropic API失敗: {e}",
        )

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return RechunkDraft(
            ok=False,
            chunk_id_hint=chunk_id_hint,
            full_text_for_review=new_full_text,
            raw_model_output=raw_text,
            error_detail=f"AI回傳內容不是合法JSON，需要人工檢查原始輸出: {e}",
        )

    proposed_chunk = parsed.get("proposed_chunk")
    if not isinstance(proposed_chunk, dict):
        return RechunkDraft(
            ok=False,
            chunk_id_hint=chunk_id_hint,
            full_text_for_review=new_full_text,
            raw_model_output=raw_text,
            error_detail="AI回傳的JSON缺少合法的 proposed_chunk 物件",
        )

    return RechunkDraft(
        ok=True,
        chunk_id_hint=proposed_chunk.get("chunk_id", chunk_id_hint),
        change_summary=parsed.get("change_summary", ""),
        content_meaningfully_changed=bool(parsed.get("content_meaningfully_changed", False)),
        proposed_chunk=proposed_chunk,
        needs_human_judgment=parsed.get("needs_human_judgment"),
        confidence_note=parsed.get("confidence_note", ""),
        verification_status="ai_drafted_pending_human_review",
        full_text_for_review=new_full_text,
        raw_model_output=raw_text,
    )
