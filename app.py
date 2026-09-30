"""
VeloGuard 跨境電商法規合規決策系統 - Streamlit 視覺化前端介面
=====================================================================

執行方式：
    streamlit run app.py

前置需求：
    1. 與本檔案同目錄下需有 `veloguard_hybrid_retriever.py`
       （混合檢索模組，本檔案直接匯入其中的
       `VeloGuardHybridRetriever` / `load_retriever_with_auto_fallback`）。
    2. 與本檔案同目錄下需有知識庫檔案
       `veloguard_multicountry_chunks.json`（德國+荷蘭合併版）。
       若檔名或路徑不同，請修改下方 `KNOWLEDGE_BASE_PATH`。

功能總覽：
    - 側邊欄：目標市場切換（DE/NL）、品類過濾、alpha權重滑桿、Top-K選擇、
      跨境退貨稅費試算工具。
    - 主查詢區：預設情境快捷按鈕 + 自然語言查詢輸入框。
    - 檢索結果卡片：一般條款卡、內部矛盾提示卡、
      「法律覆蓋（Override）」衝突警示卡（品牌字面規定 vs 法定強制標準對比）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# 路徑設定：確保無論從哪個工作目錄執行 `streamlit run app.py`，
# 都能正確匯入同目錄下的 veloguard_hybrid_retriever 模組。
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from veloguard_hybrid_retriever import (  # noqa: E402
    RetrievalResult,
    load_retriever_with_auto_fallback,
)

KNOWLEDGE_BASE_PATH = APP_DIR / "veloguard_multicountry_chunks.json"


# ---------------------------------------------------------------------------
# 靜態設定：市場資訊、品類選項、中文標籤對照表、預設查詢情境
# ---------------------------------------------------------------------------

# 各市場基本資訊：國家代碼、VAT/BTW稅率、稅名、顯示用旗幟標籤
MARKETS: dict[str, dict[str, Any]] = {
    "🇩🇪 德國 (Germany - 19% MwSt)": {
        "code": "DE",
        "vat_rate": 0.19,
        "vat_label": "MwSt（德國增值稅）",
        "depreciation_label": "Wertersatz（價值賠償）",
    },
    "🇳🇱 荷蘭 (Netherlands - 21% BTW)": {
        "code": "NL",
        "vat_rate": 0.21,
        "vat_label": "BTW（荷蘭增值稅）",
        "depreciation_label": "Waardevermindering（價值減損）",
    },
}

# 品類篩選選項 -> 傳給 retriever.search(product_type=...) 的值
# 備註：知識庫中的 product_scope 實際只有 "All" / "E_Bike" / "Custom_Bike" /
# "Accessories" 四種值，一般整車的退貨規則多半標記為 "All"（適用所有品類）。
# 這裡讓「整車 (Bikes)」對應一個知識庫中不存在的品類代碼 "Bikes"：
# 依路由過濾邏輯（product_scope=="All" 或 product_scope==product_type），
# 這樣篩選出的正好是「泛用（All）條款」，同時排除E-Bike/客製化/配件的
# 專屬條款，符合「只看一般整車規則」的篩選意圖。
PRODUCT_CATEGORIES: dict[str, Optional[str]] = {
    "全部品類 (All)": None,
    "整車 (Bikes)": "Bikes",
    "電動輔助自行車 (E_Bike)": "E_Bike",
    "客製化車款 (Custom_Bike)": "Custom_Bike",
    "配件 (Accessories)": "Accessories",
}

TOPIC_LABELS: dict[str, str] = {
    "withdrawal_period": "撤回權 / 退貨期限",
    "value_depreciation": "價值減損 / 折舊費用",
    "return_exclusions": "退貨排除條款",
    "return_logistics_cost": "退貨物流與費用",
}

JURISDICTION_LABELS: dict[str, str] = {
    "DE_EU": "🇩🇪 德國",
    "NL_EU": "🇳🇱 荷蘭",
    "EU": "🇪🇺 全歐盟共通",
    "COMMON_EU": "🇪🇺 全歐盟共通",
}

# 預設情境快捷按鈕：(按鈕顯示文字, 帶入查詢框的完整自然語言問句)
PRESET_QUERIES: list[dict[str, str]] = [
    {
        "label": "🚴 3500€ E-Bike 落地磨損退貨",
        "query": "3500歐元的E-Bike已經落地騎乘試用，消費者現在要求退貨，能扣多少折舊費？",
    },
    {
        "label": "🎨 客製化塗裝車退貨",
        "query": "消費者訂購了客製化烤漆塗裝的自行車，現在想要退貨，適用撤回權嗎？",
    },
    {
        "label": "🔋 鋰電池能否自行寄送",
        "query": "E-Bike電池已經從車上拆下，消費者想自己用一般快遞寄回，可以嗎？",
    },
]


# ---------------------------------------------------------------------------
# 檢索器載入（使用 st.cache_resource 避免每次互動都重新初始化模型）
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="正在載入 VeloGuard 檢索引擎（首次啟動較久，請稍候）...")
def get_retriever():
    """載入並快取 VeloGuardHybridRetriever 執行個體。

    使用 `load_retriever_with_auto_fallback`：會先嘗試下載/載入
    SentenceTransformer 多語言模型，若在時限內連不上（例如目前網路
    白名單不含 huggingface.co），會自動改用離線的 LocalLSAEncoder
    （TF-IDF+LSA），確保介面在任何網路環境下都能正常啟動與示範。

    Returns:
        (retriever, embed_mode) 二元組；embed_mode 為
        "sentence_transformer" 或 "local_lsa_fallback"。
    """
    return load_retriever_with_auto_fallback(str(KNOWLEDGE_BASE_PATH), verbose=False)


# ---------------------------------------------------------------------------
# 結果卡片渲染
# ---------------------------------------------------------------------------

def render_score_row(r: RetrievalResult) -> None:
    """以 st.metric 三欄呈現BM25 / 語意 / 綜合分數，方便理解排序依據。"""
    c1, c2, c3 = st.columns(3)
    c1.metric("BM25 關鍵字分數", f"{r.bm25_score_norm:.2f}")
    c2.metric("語意向量分數", f"{r.dense_score_norm:.2f}")
    c3.metric("綜合分數 (Final)", f"{r.final_score:.2f}")


def is_override_conflict_chunk(c: dict[str, Any]) -> bool:
    """判斷chunk是否為「已確認的法律覆蓋（Override）」衝突類型。"""
    return c.get("applied_rule") == "legal_minimum_standard" and bool(c.get("legal_minimum_standard"))


def is_internal_inconsistency_chunk(c: dict[str, Any]) -> bool:
    """判斷chunk是否為「內部資料矛盾/待確認」類型（有conflict_flag但非override）。"""
    return bool(c.get("conflict_flag")) and not is_override_conflict_chunk(c)


def render_override_conflict_card(c: dict[str, Any], result: Optional[RetrievalResult] = None) -> None:
    """渲染「法律覆蓋（Override）」衝突警示卡片。

    適用對象：chunk 帶有完整override結構欄位
    （brand_stated_rule / legal_minimum_standard / applied_rule），
    例如 Giant_NL_AGB_004——品牌條款字面規定與當地法定強制標準有明確落差，
    系統判斷應以法律標準覆蓋品牌字面規定執行。

    Args:
        c: 原始chunk字典（可直接來自知識庫，不一定要經過檢索排序）。
        result: 若此卡片是某次查詢排序結果的一部分，傳入對應的
            RetrievalResult 以顯示BM25/語意/綜合分數；若是「固定顯示、
            不受排序影響」的提醒區塊呼叫，則留空不顯示分數列。
    """
    with st.container(border=True):
        # 把chunk_id/條款依據獨立放在最上面當作明顯的標題列，
        # 不要埋在st.error警示文字中間，讓使用者第一眼就能定位是哪一筆。
        topic_label = TOPIC_LABELS.get(c.get("topic"), c.get("topic", ""))
        st.markdown(f"### 🚨 {topic_label}　｜　`{c['chunk_id']}`")
        st.caption(f"條款依據：{c.get('official_article_ref', '')}")
        st.error(
            "此條款「品牌字面規定」與「當地法定強制標準」不一致，"
            "系統已依法規標準覆蓋品牌字面規定，禁止依原文字面直接執行。"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ❌ 品牌字面規定")
            st.markdown(f":red[{c.get('brand_stated_rule', c.get('summary_zh', ''))}]")
        with col2:
            st.markdown("#### ⚖️ 法定強制標準")
            st.markdown(f":green[{c.get('legal_minimum_standard', '（未提供詳細法規標準文字）')}]")

        st.info(f"👉 **客服標準操作守則**：{c.get('operational_rule', '')}")

        with st.expander("🔍 查看覆蓋原因、驗證來源與後續建議行動"):
            if c.get("override_reason"):
                st.markdown(f"**覆蓋原因：** {c['override_reason']}")
            if c.get("verification_status"):
                st.markdown(f"**驗證狀態：** {c['verification_status']}")
            if c.get("verification_sources"):
                sources = c["verification_sources"]
                if isinstance(sources, list):
                    sources = "、".join(sources)
                st.markdown(f"**驗證來源：** {sources}")
            if c.get("action_required"):
                st.markdown(f"**建議後續行動：** {c['action_required']}")
            if c.get("original_text_snippet"):
                st.caption(f"原文摘錄：「{c['original_text_snippet']}」")

        if result is not None:
            render_score_row(result)


def render_internal_inconsistency_card(c: dict[str, Any], result: Optional[RetrievalResult] = None) -> None:
    """渲染「內部資料矛盾/待確認」提示卡片（黃色警示）。

    適用對象：chunk 帶有 `conflict_flag` 但沒有完整override結構
    （例如兩個官方頁面說法不一致、原文表述有歧義等），性質上是「待內部
    確認」而非「已確認的法律違規」，因此用較輕量的黃色警示呈現，
    不做品牌規定vs法律標準的二元對比。

    Args:
        c: 原始chunk字典。
        result: 同 `render_override_conflict_card`，可選的排序結果分數。
    """
    with st.container(border=True):
        # 同override卡片，chunk_id/條款依據獨立放最上面當標題列。
        topic_label = TOPIC_LABELS.get(c.get("topic"), c.get("topic", ""))
        st.markdown(f"### ⚠️ {topic_label}　｜　`{c['chunk_id']}`")
        st.caption(f"條款依據：{c.get('official_article_ref', '')}")
        st.warning(f"資料一致性提示：{c.get('conflict_flag', '')}")
        st.markdown(f"**摘要：** {c.get('summary_zh', '')}")
        st.info(f"👉 **客服標準操作守則**：{c.get('operational_rule', '')}")
        if result is not None:
            render_score_row(result)


def render_normal_card(r: RetrievalResult, rank: int) -> None:
    """渲染一般（無衝突標記）檢索結果卡片。"""
    with st.container(border=True):
        topic_label = TOPIC_LABELS.get(r.topic, r.topic)
        jur_label = JURISDICTION_LABELS.get(r.jurisdiction, r.jurisdiction)
        st.markdown(
            f"**#{rank}　{topic_label}** ｜ {jur_label} ｜ 適用範圍: `{r.product_scope}`"
        )
        st.caption(f"條款依據：{r.official_article_ref}")
        st.markdown(f"**摘要：** {r.summary_zh}")
        st.markdown(f"👉 **客服標準操作守則：** {r.operational_rule}")
        if r.original_text_snippet:
            with st.expander("查看原文摘錄"):
                st.caption(r.original_text_snippet)
        render_score_row(r)


def render_result_card(r: RetrievalResult, rank: int) -> None:
    """依chunk是否帶有衝突/覆蓋標記，分派到對應的卡片渲染函式。"""
    c = r.raw_chunk
    if is_override_conflict_chunk(c):
        render_override_conflict_card(c, result=r)
    elif is_internal_inconsistency_chunk(c):
        render_internal_inconsistency_card(c, result=r)
    else:
        render_normal_card(r, rank)


# ---------------------------------------------------------------------------
# 固定顯示的「已知法規重點提醒」面板：不受Top-K/排序影響
# ---------------------------------------------------------------------------

def render_known_conflicts_panel(retriever, params: dict[str, Any], shown_chunk_ids: set[str]) -> None:
    """顯示目前市場+品類篩選範圍內，所有已知的衝突/覆蓋chunk。

    背景：檢索結果的排序取決於BM25/語意分數，同一個chunk換一種問法，
    排名可能大幅浮動——例如 `Giant_NL_AGB_004` 這個重要的法律覆蓋警示，
    若使用者查詢字句與其摘要文字關鍵字重疊不高，就可能被擠出Top-K之外，
    導致這麼重要的合規提醒「使用者剛好沒看到」。這在合規決策系統裡是不
    可接受的風險，因此另外開一個「固定提醒區塊」：只要目前選擇的市場+
    品類範圍內存在已知的衝突/覆蓋chunk，一律顯示，不受任何查詢或排序
    參數影響。

    為避免與下方查詢結果重複顯示造成混亂，若某個chunk已經出現在最近一次
    查詢結果中（且該次查詢的市場/品類與目前選擇相同），這裡就不重複列出。

    Args:
        retriever: 已載入的 VeloGuardHybridRetriever。
        params: `render_sidebar()` 回傳的目前參數（市場代碼、品類篩選）。
        shown_chunk_ids: 已經在下方查詢結果卡片顯示過的chunk_id集合。
    """
    candidates = retriever._route_filter(params["country_code"], params["product_type"])
    overrides = [c for c in candidates if is_override_conflict_chunk(c) and c["chunk_id"] not in shown_chunk_ids]
    inconsistencies = [
        c for c in candidates if is_internal_inconsistency_chunk(c) and c["chunk_id"] not in shown_chunk_ids
    ]

    if not overrides and not inconsistencies:
        return

    total = len(overrides) + len(inconsistencies)
    with st.expander(
        f"🛡️ 目前篩選範圍內的已知法規重點提醒（共 {total} 筆，不受Top-K/排序影響，固定顯示）",
        expanded=bool(overrides),
    ):
        st.caption(
            "以下項目只要符合目前選擇的市場與品類，就一律顯示，"
            "不會因為查詢字句或Top-K設定不同而被排除在外。"
        )
        for c in overrides:
            render_override_conflict_card(c)
        for c in inconsistencies:
            render_internal_inconsistency_card(c)


# ---------------------------------------------------------------------------
# 側邊欄：市場/品類/檢索參數 + 跨境退貨稅費試算工具
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    """渲染側邊欄所有控制元件，回傳使用者目前選擇的參數字典。"""
    st.sidebar.title("🚲 VeloGuard 設定")

    market_label = st.sidebar.selectbox("🌍 目標市場", list(MARKETS.keys()))
    market = MARKETS[market_label]

    category_label = st.sidebar.selectbox("📦 產品品類過濾", list(PRODUCT_CATEGORIES.keys()))
    product_type = PRODUCT_CATEGORIES[category_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 進階檢索參數")
    alpha = st.sidebar.slider(
        "語意 vs 關鍵字權重 (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="alpha=0 完全依賴BM25關鍵字比對；alpha=1 完全依賴語意向量相似度。",
    )
    top_k = st.sidebar.slider("Top-K 回傳筆數", min_value=1, max_value=10, value=5)

    st.sidebar.markdown("---")
    render_vat_calculator(market)

    return {
        "market_label": market_label,
        "country_code": market["code"],
        "vat_rate": market["vat_rate"],
        "vat_label": market["vat_label"],
        "depreciation_label": market["depreciation_label"],
        "product_type": product_type,
        "alpha": alpha,
        "top_k": top_k,
    }


def render_vat_calculator(market: dict[str, Any]) -> None:
    """側邊欄的跨境退貨稅費試算小工具（展開式 expander）。"""
    with st.sidebar.expander("💶 跨境退貨稅費試算工具", expanded=False):
        st.caption(
            "本工具僅為退款金額試算輔助，非正式法律或稅務意見；"
            "實際退款金額請以Giant官方財務/法務流程認定為準。"
        )

        order_amount = st.number_input(
            "訂單含稅總金額 (EUR)", min_value=0.0, value=1000.0, step=50.0
        )
        depreciation_pct = st.slider(
            "技師評估之磨損扣除比例 (%)", min_value=0, max_value=100, value=20
        )

        vat_rate = market["vat_rate"]

        # 計算邏輯：
        # 1. 先算出「原訂單」的不含稅淨額與VAT，供使用者理解原始金額結構。
        # 2. 折舊扣除金額以「訂單含稅總金額」為基礎計算（Wertersatz/
        #    Waardevermindering 實務上多以消費者實際支付的含稅價格為基準）。
        # 3. 最終退款金額 = 含稅總額 - 折舊扣除金額；並進一步拆解出
        #    「退款中的VAT部分」，因為部分退款時，其對應的VAT也應等比例退還。
        net_amount_original = order_amount / (1 + vat_rate)
        depreciation_amount = order_amount * (depreciation_pct / 100)
        refund_gross = order_amount - depreciation_amount
        refund_net = refund_gross / (1 + vat_rate)
        refund_vat = refund_gross - refund_net

        st.markdown(f"**當前市場：** {market['code']}（VAT/BTW = {vat_rate:.0%}）")

        c1, c2 = st.columns(2)
        c1.metric("原訂單不含稅淨額", f"€{net_amount_original:,.2f}")
        c2.metric(market["vat_label"], f"€{order_amount - net_amount_original:,.2f}")

        c3, c4 = st.columns(2)
        c3.metric(
            f"折舊扣除（{market['depreciation_label']}）",
            f"-€{depreciation_amount:,.2f}",
        )
        c4.metric("退款中對應VAT金額", f"€{refund_vat:,.2f}")

        st.metric("💰 最終應退還買家款項", f"€{refund_gross:,.2f}")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="VeloGuard 跨境合規決策系統",
        page_icon="🚲",
        layout="wide",
    )

    params = render_sidebar()

    retriever, embed_mode = get_retriever()

    st.title("🚲 VeloGuard 跨境合規決策系統")
    st.caption(
        f"目前市場：{params['market_label']}　｜　"
        f"知識庫chunk總數：{len(retriever.chunks)}"
    )

    if embed_mode == "local_lsa_fallback":
        st.warning(
            "⚠️ 語意引擎目前為**離線替代模式**（TF-IDF+LSA）："
            "偵測到目前網路環境無法連線 Hugging Face Hub 下載多語言模型，"
            "已自動退回本地語意編碼器以維持系統可用性。語意檢索品質較預訓練"
            "Transformer模型略低，正式上線建議在可連線的環境重新啟動。",
            icon="⚠️",
        )
    elif embed_mode == "bm25_only_fallback":
        st.warning(
            "⚠️ 語意引擎目前為**純關鍵字模式**（僅BM25）："
            "sentence-transformers 與 scikit-learn 均無法在目前環境使用"
            "（常見於過新的Python版本，相依套件尚無對應安裝包），語意向量"
            "分數暫時恆為0。建議將側邊欄的 alpha 調低（趨近0），並儘量使用"
            "與知識庫原文相近的關鍵字查詢；長期建議改用Python 3.11/3.12"
            "等較穩定版本重新安裝。",
            icon="⚠️",
        )
    else:
        st.success("✅ 語意引擎：SentenceTransformer 多語言模型（paraphrase-multilingual-MiniLM-L12-v2）")

    st.markdown("---")

    # -----------------------------------------------------------------
    # 查詢區：預設情境快捷按鈕 + 自然語言輸入框
    # 頁面順序：先查詢框 → 再已知法規重點提醒 → 最後才是完整搜尋結果。
    # -----------------------------------------------------------------
    st.subheader("🔎 合規檢索查詢")

    if "query_text" not in st.session_state:
        st.session_state.query_text = PRESET_QUERIES[0]["query"]

    st.caption("快速情境（點擊自動帶入查詢框）：")
    preset_cols = st.columns(len(PRESET_QUERIES))
    for col, preset in zip(preset_cols, PRESET_QUERIES):
        if col.button(preset["label"], use_container_width=True):
            st.session_state.query_text = preset["query"]

    st.text_area("自然語言查詢", key="query_text", height=90)

    search_clicked = st.button("🚀 開始合規檢索", type="primary")

    # -----------------------------------------------------------------
    # 執行檢索並顯示結果（存入session_state，避免頁面互動時結果消失）
    # -----------------------------------------------------------------
    if search_clicked:
        with st.spinner("正在執行混合檢索（BM25 + 語意向量）..."):
            results = retriever.search(
                query=st.session_state.query_text,
                target_country=params["country_code"],
                product_type=params["product_type"],
                top_k=params["top_k"],
                alpha=params["alpha"],
            )
        st.session_state.last_results = results
        st.session_state.last_query_meta = params

    # 現在session_state已經是最新狀態（若剛執行過檢索），才計算「查詢
    # 結果已顯示過哪些chunk_id」，用來讓下面的提醒面板不與結果重複。
    #
    # 同時判斷：上一次查詢結果的市場/品類，是否還跟目前側邊欄選擇一致。
    # 若使用者換了市場或品類卻還沒按「開始合規檢索」，舊結果其實是對應
    # 「另一組設定」算出來的，繼續原樣顯示容易誤導（也會跟下面新面板
    # 重複），所以改成提示使用者重新查詢，而不是留著過期的結果。
    results: list[RetrievalResult] = st.session_state.get("last_results", [])
    results_meta = st.session_state.get("last_query_meta", {})
    results_match_current_params = (
        results_meta.get("country_code") == params["country_code"]
        and results_meta.get("product_type") == params["product_type"]
    )
    shown_chunk_ids = {r.chunk_id for r in results} if results_match_current_params else set()

    # -----------------------------------------------------------------
    # 已知法規重點提醒面板：只要符合目前市場/品類就一律顯示，
    # 不受下方查詢結果的Top-K或排序影響。放在查詢框之後、完整結果
    # 列表之前，執行到這裡時session_state已經是最新的，不需要再用
    # placeholder延後填入。
    # -----------------------------------------------------------------
    render_known_conflicts_panel(retriever, params, shown_chunk_ids)

    st.markdown("---")

    if not results:
        st.info("尚無檢索結果。請輸入查詢內容並點擊「開始合規檢索」。")
        return

    if not results_match_current_params:
        st.info(
            f"⚠️ 目前顯示的是「{results_meta.get('market_label', '之前')}」設定下的查詢結果，"
            "與目前側邊欄選擇的市場/品類不同。請重新點擊「開始合規檢索」以更新結果。"
        )
        return

    meta = results_meta
    st.subheader(
        f"📋 檢索結果（{meta['market_label']} ｜ 共 {len(results)} 筆，"
        f"依綜合分數排序）"
    )

    conflict_count = sum(
        1
        for r in results
        if r.raw_chunk.get("applied_rule") == "legal_minimum_standard"
        or r.raw_chunk.get("conflict_flag")
    )
    if conflict_count:
        st.caption(f"⚠️ 本次結果中有 {conflict_count} 筆帶有法規衝突/內部一致性標記，已於下方特別標示。")

    for rank, r in enumerate(results, start=1):
        render_result_card(r, rank)


if __name__ == "__main__":
    main()
