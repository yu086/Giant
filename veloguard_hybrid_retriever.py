"""
VeloGuard 跨境電商法規合規決策系統 - 混合檢索（Hybrid Search）模組
=====================================================================

安裝所需套件：
    pip install rank-bm25 sentence-transformers numpy

模組功能：
    對知識庫（knowledge_base_de_nl.json）進行「國家/品類路由過濾」
    + 「BM25關鍵字檢索」+「多語言語意向量檢索」的混合式RAG檢索管線。

作者備註：
    本模組僅負責「檢索（Retrieval）」，不含「生成（Generation）」步驟；
    取回的Top-K chunk建議後續交由LLM組織成自然語言回答，或直接
    將 operational_rule / summary_zh 欄位呈現給客服人員參考。

離線/受限網路環境備註：
    正式版語意向量使用 sentence-transformers 的
    paraphrase-multilingual-MiniLM-L12-v2，第一次執行時會從
    Hugging Face Hub 下載權重檔（約470MB）。若執行環境的網路白名單
    不包含 huggingface.co（例如企業內網、封閉的CI/沙盒環境，只開放
    pypi.org / npmjs.org / github.com 等套件登錄檔網域），則無法完成
    這次下載——這類環境目前沒有合法的GitHub鏡像可以取代Hugging Face
    Hub上的這個模型（各套件登錄檔網域是模型下載的必要路徑，而非可
    任意替換的鏡像來源）。
    本模組另外提供 `LocalLSAEncoder`（見下方），以 scikit-learn 的
    TF-IDF + TruncatedSVD（潛在語意分析）在「本地知識庫語料」上即時
    訓練出語意向量，不需下載任何外部權重檔，可在完全離線或網路受限
    的環境下驗證整條檢索管線邏輯。其語意品質不如預訓練的多語言
    Transformer模型，僅建議作為「網路受限環境的替代/測試方案」，
    正式上線仍建議改用 SentenceTransformer。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

# sentence-transformers（連帶torch）安裝失敗時，不讓整個模組直接掛掉——
# 例如在極新的Python版本上，torch有時還沒推出對應的安裝包，會導致
# `pip install sentence-transformers` 失敗或import錯誤。這裡改成
# 「盡量載入，載入不到就記錄下來」，讓LocalLSAEncoder離線備援方案
# 仍然可以正常運作，不會被這個匯入錯誤拖累。
try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# 資料結構定義
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """單一檢索結果，包含原始chunk內容與各階段分數，方便除錯與呈現。"""

    chunk_id: str
    topic: str
    jurisdiction: str
    product_scope: str
    official_article_ref: str
    summary_zh: str
    operational_rule: str
    original_text_snippet: str
    bm25_score_raw: float
    bm25_score_norm: float
    dense_score_raw: float
    dense_score_norm: float
    final_score: float
    raw_chunk: dict[str, Any] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """轉為結構化字典，方便序列化成JSON回傳給前端或LLM。"""
        return {
            "chunk_id": self.chunk_id,
            "topic": self.topic,
            "jurisdiction": self.jurisdiction,
            "product_scope": self.product_scope,
            "official_article_ref": self.official_article_ref,
            "summary_zh": self.summary_zh,
            "operational_rule": self.operational_rule,
            "original_text_snippet": self.original_text_snippet,
            "scores": {
                "bm25_raw": round(self.bm25_score_raw, 4),
                "bm25_normalized": round(self.bm25_score_norm, 4),
                "dense_raw": round(self.dense_score_raw, 4),
                "dense_normalized": round(self.dense_score_norm, 4),
                "final_score": round(self.final_score, 4),
            },
        }


# ---------------------------------------------------------------------------
# 主檢索器類別
# ---------------------------------------------------------------------------

class VeloGuardHybridRetriever:
    """VeloGuard 混合檢索器（BM25 稀疏檢索 + 多語言語意向量密集檢索）。

    使用流程：
        1. 以 `knowledge_base_path` 初始化，載入JSON知識庫並建立索引。
        2. 呼叫 `.search(query, target_country, product_type, top_k)` 取得結果。

    Attributes:
        chunks: 完整載入的知識庫chunk列表（未過濾）。
        embed_model: 用於語意向量檢索的 SentenceTransformer 模型實例。
        _bm25_cache: 依 (country, product_type) 組合快取BM25索引，避免重複斷詞建索引。
    """

    # 多語言輕量模型：支援中文查詢比對德文/荷蘭文原文，且模型體積小、推論速度快，
    # 適合放進沒有GPU的後端服務或本地開發環境。
    DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    # 國家代碼 -> 允許通過路由過濾的 jurisdiction 集合
    # 注意：真實知識庫（veloguard_multicountry_chunks.json）中，適用全歐盟的
    # 共通條款（如14天撤回權、鋰電池跨境運輸規範）使用的欄位值是 "EU"，
    # 而不是本模組Mock Data範例中示範用的 "COMMON_EU"；兩者都收進允許清單，
    # 避免因資料產製腳本用詞不同而讓共通條款被路由過濾器誤篩掉。
    _JURISDICTION_MAP: dict[str, set[str]] = {
        "DE": {"DE_EU", "EU", "COMMON_EU"},
        "NL": {"NL_EU", "EU", "COMMON_EU"},
    }

    def __init__(
        self,
        knowledge_base_path: str,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        embed_model: Optional[SentenceTransformer] = None,
    ) -> None:
        """初始化檢索器：載入知識庫、建立語意向量模型與嵌入快取。

        Args:
            knowledge_base_path: `knowledge_base_de_nl.json` 的檔案路徑。
            embed_model_name: SentenceTransformer 模型名稱，預設為多語言輕量模型。
            embed_model: 若已在外部載入模型實例，可直接傳入以避免重複載入
                （例如測試時注入 Mock 模型，或注入 `LocalLSAEncoder` 離線方案）。

        Raises:
            RuntimeError: 當未傳入 `embed_model`、且執行環境未安裝
                sentence-transformers套件時（例如torch尚未支援目前的
                Python版本）。此時應改用
                `load_retriever_with_auto_fallback()`，或自行建立
                `LocalLSAEncoder` 後傳入 `embed_model` 參數。
        """
        self.chunks: list[dict[str, Any]] = self._load_knowledge_base(knowledge_base_path)

        if embed_model is not None:
            self.embed_model = embed_model
        elif _SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embed_model = SentenceTransformer(embed_model_name)
        else:
            raise RuntimeError(
                "未安裝 sentence-transformers（或其相依套件torch無法在目前"
                "Python版本上安裝），且未傳入 embed_model。請改用 "
                "load_retriever_with_auto_fallback()（會自動切換至離線"
                "LocalLSAEncoder），或自行建立 LocalLSAEncoder 實例後透過 "
                "embed_model 參數傳入。"
            )

        # 快取：每個chunk的「檢索用文本」與其embedding，避免每次查詢都重算
        self._chunk_texts: dict[str, str] = {
            c["chunk_id"]: self._build_retrieval_text(c) for c in self.chunks
        }
        self._chunk_embeddings: dict[str, np.ndarray] = self._encode_all_chunks()

    # ------------------------------------------------------------------
    # 資料載入與前處理
    # ------------------------------------------------------------------

    @staticmethod
    def _load_knowledge_base(path: str) -> list[dict[str, Any]]:
        """讀取知識庫JSON檔案，回傳chunk物件列表。

        Args:
            path: JSON檔案路徑。

        Returns:
            chunk字典的列表。

        Raises:
            ValueError: 當必要欄位缺失時，提早失敗並給出明確錯誤訊息，
                避免髒資料靜默流入檢索管線。
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        chunks = data["chunks"] if isinstance(data, dict) and "chunks" in data else data

        required_fields = {
            "chunk_id",
            "topic",
            "jurisdiction",
            "product_scope",
            "official_article_ref",
            "summary_zh",
            "operational_rule",
        }
        for c in chunks:
            missing = required_fields - c.keys()
            if missing:
                raise ValueError(
                    f"Chunk '{c.get('chunk_id', '<unknown>')}' 缺少必要欄位: {missing}"
                )
            # original_text_snippet 允許缺省，補空字串避免後續 KeyError
            c.setdefault("original_text_snippet", "")

        return chunks

    @staticmethod
    def _build_retrieval_text(chunk: dict[str, Any]) -> str:
        """組合一個chunk用於檢索比對的合成文本。

        將中文摘要、操作規則與原文摘錄串在一起，讓BM25與語意向量
        都能同時利用「中文查詢意圖」與「原文關鍵字」進行匹配，
        這對跨語言（中文Query比對德/荷文原文）場景尤其重要。

        Args:
            chunk: 單一chunk字典。

        Returns:
            合成後的檢索文本字串。
        """
        parts = [
            chunk.get("topic", ""),
            chunk.get("product_scope", ""),
            chunk.get("summary_zh", ""),
            chunk.get("operational_rule", ""),
            chunk.get("original_text_snippet", ""),
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """簡易多語言分詞器，供BM25使用。

        說明：
            正式產品建議中文段落改用 jieba 分詞，德文/荷蘭文部分用
            空白分詞即可（皆為空白分隔語言）。此處為求模組零額外
            依賴，採用「中文逐字切 + 非中文按空白/標點切」的簡化策略，
            足以支撐BM25對關鍵詞（如"E-Bike"、"落地"、"磨損"）的匹配。

        Args:
            text: 欲斷詞的原始字串。

        Returns:
            斷詞後的token列表（全部小寫）。
        """
        text = text.lower()
        # 先抓出英數字詞（含連字號，如 e-bike）
        latin_tokens = re.findall(r"[a-z0-9][a-z0-9\-]*", text)
        # 移除英數字後，剩餘的中文/其他文字逐字切分
        remainder = re.sub(r"[a-z0-9][a-z0-9\-]*", " ", text)
        cjk_tokens = [ch for ch in remainder if ch.strip() and not ch.isspace()]
        return latin_tokens + cjk_tokens

    def _encode_all_chunks(self) -> dict[str, np.ndarray]:
        """對所有chunk的檢索文本一次性做批次embedding編碼。

        Returns:
            chunk_id -> 正規化後的embedding向量（np.ndarray）。
        """
        chunk_ids = list(self._chunk_texts.keys())
        texts = [self._chunk_texts[cid] for cid in chunk_ids]
        embeddings = self.embed_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return dict(zip(chunk_ids, embeddings))

    # ------------------------------------------------------------------
    # Step 1: 國家與品類路由過濾
    # ------------------------------------------------------------------

    def _route_filter(
        self, target_country: str, product_type: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """依國家與（可選的）品類，過濾出候選chunk子集。

        Args:
            target_country: "DE" 或 "NL"。
            product_type: 品類篩選條件，若提供則僅保留
                `product_scope == "All"` 或 `product_scope == product_type`
                的chunk；若為 None 則不做品類過濾。

        Returns:
            通過路由過濾的候選chunk列表。

        Raises:
            ValueError: 當 target_country 不在支援清單內時。
        """
        target_country = target_country.upper()
        if target_country not in self._JURISDICTION_MAP:
            raise ValueError(
                f"不支援的國家代碼: '{target_country}'，"
                f"目前僅支援 {list(self._JURISDICTION_MAP.keys())}"
            )

        allowed_jurisdictions = self._JURISDICTION_MAP[target_country]
        candidates = [c for c in self.chunks if c["jurisdiction"] in allowed_jurisdictions]

        if product_type is not None:
            candidates = [
                c
                for c in candidates
                if c["product_scope"] == "All" or c["product_scope"] == product_type
            ]

        return candidates

    # ------------------------------------------------------------------
    # Step 2: BM25 稀疏檢索
    # ------------------------------------------------------------------

    def _bm25_scores(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> dict[str, float]:
        """對候選集合計算BM25分數。

        Args:
            query: 使用者查詢字串。
            candidates: 已經過路由過濾的候選chunk列表。

        Returns:
            chunk_id -> BM25原始分數 的對應字典。
        """
        if not candidates:
            return {}

        corpus = [self._tokenize(self._chunk_texts[c["chunk_id"]]) for c in candidates]
        bm25 = BM25Okapi(corpus)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        return {c["chunk_id"]: float(s) for c, s in zip(candidates, scores)}

    # ------------------------------------------------------------------
    # Step 3: 語意向量檢索
    # ------------------------------------------------------------------

    def _dense_scores(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> dict[str, float]:
        """對候選集合計算語意向量（Cosine相似度）分數。

        Args:
            query: 使用者查詢字串。
            candidates: 已經過路由過濾的候選chunk列表。

        Returns:
            chunk_id -> Cosine相似度原始分數 的對應字典。
        """
        if not candidates:
            return {}

        query_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        scores: dict[str, float] = {}
        for c in candidates:
            chunk_vec = self._chunk_embeddings[c["chunk_id"]]
            # 兩向量皆已 L2 正規化，內積即為 Cosine 相似度
            cosine_sim = float(np.dot(query_vec, chunk_vec))
            scores[c["chunk_id"]] = cosine_sim

        return scores

    # ------------------------------------------------------------------
    # Step 4: 分數正規化與加權融合
    # ------------------------------------------------------------------

    @staticmethod
    def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
        """對一組分數做 Min-Max 正規化至 [0, 1]。

        Args:
            scores: chunk_id -> 原始分數。

        Returns:
            chunk_id -> 正規化後分數。若所有分數相同（含只有一筆資料
            或分數全為0），回傳全部為 0.0，避免除以零。
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_v, max_v = min(values), max(values)

        if max_v - min_v < 1e-12:
            return {k: 0.0 for k in scores}

        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    def _fuse_scores(
        self,
        bm25_raw: dict[str, float],
        dense_raw: dict[str, float],
        alpha: float,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """將BM25與Dense分數正規化後，依 alpha 加權融合。

        Args:
            bm25_raw: chunk_id -> BM25原始分數。
            dense_raw: chunk_id -> Dense原始分數。
            alpha: 語意向量分數的權重，範圍應在 [0, 1]，
                `Final_Score = alpha * Dense_Score + (1 - alpha) * BM25_Score`。

        Returns:
            三元組 (bm25_normalized, dense_normalized, final_scores)，
            皆為 chunk_id -> 分數 的字典。
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必須介於 0 到 1 之間，收到: {alpha}")

        bm25_norm = self._min_max_normalize(bm25_raw)
        dense_norm = self._min_max_normalize(dense_raw)

        chunk_ids = set(bm25_norm) | set(dense_norm)
        final_scores = {
            cid: alpha * dense_norm.get(cid, 0.0) + (1 - alpha) * bm25_norm.get(cid, 0.0)
            for cid in chunk_ids
        }

        return bm25_norm, dense_norm, final_scores

    # ------------------------------------------------------------------
    # Step 5: 對外主入口 - 組裝Top-K結果
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        target_country: str,
        product_type: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[RetrievalResult]:
        """執行完整的混合檢索管線，回傳 Top-K 結果。

        Args:
            query: 使用者的自然語言查詢（可為中文）。
            target_country: 目標市場國家代碼，"DE" 或 "NL"。
            product_type: 品類篩選條件（如 "E_Bike"），預設不篩選。
            top_k: 回傳結果數量上限，預設 5。
            alpha: Dense分數權重，`Final = alpha*Dense + (1-alpha)*BM25`，預設 0.5。

        Returns:
            依 final_score 由高到低排序的 `RetrievalResult` 列表，
            長度最多為 top_k（候選數不足時會少於 top_k）。
        """
        # Step 1: 路由過濾
        candidates = self._route_filter(target_country, product_type)
        if not candidates:
            return []

        # Step 2 & 3: 分別計算稀疏與密集分數
        bm25_raw = self._bm25_scores(query, candidates)
        dense_raw = self._dense_scores(query, candidates)

        # Step 4: 正規化與加權融合
        bm25_norm, dense_norm, final_scores = self._fuse_scores(bm25_raw, dense_raw, alpha)

        # Step 5: 排序組裝
        ranked_ids = sorted(final_scores, key=lambda cid: final_scores[cid], reverse=True)[:top_k]
        candidates_by_id = {c["chunk_id"]: c for c in candidates}

        results: list[RetrievalResult] = []
        for cid in ranked_ids:
            c = candidates_by_id[cid]
            results.append(
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    topic=c["topic"],
                    jurisdiction=c["jurisdiction"],
                    product_scope=c["product_scope"],
                    official_article_ref=c["official_article_ref"],
                    summary_zh=c["summary_zh"],
                    operational_rule=c["operational_rule"],
                    original_text_snippet=c.get("original_text_snippet", ""),
                    bm25_score_raw=bm25_raw.get(cid, 0.0),
                    bm25_score_norm=bm25_norm.get(cid, 0.0),
                    dense_score_raw=dense_raw.get(cid, 0.0),
                    dense_score_norm=dense_norm.get(cid, 0.0),
                    final_score=final_scores[cid],
                    raw_chunk=c,
                )
            )

        return results


# ---------------------------------------------------------------------------
# 離線替代方案：不需下載任何外部權重檔的語意編碼器
# ---------------------------------------------------------------------------

class LocalLSAEncoder:
    """以 TF-IDF + TruncatedSVD（LSA，潛在語意分析）實作的離線語意編碼器。

    用途：
        當執行環境無法連線 huggingface.co（例如網路白名單僅開放
        pypi.org / npmjs.org / github.com 等套件登錄檔網域的沙盒或
        企業內網環境）、因而無法下載 SentenceTransformer 預訓練權重時，
        可改用這個類別作為 `VeloGuardHybridRetriever(embed_model=...)`
        的替代品，滿足相同的 `.encode(texts, convert_to_numpy=True,
        normalize_embeddings=True)` 介面。

    原理：
        1. 對「知識庫全體chunk文本」做 TF-IDF 向量化，建立詞彙表；
        2. 用 TruncatedSVD 將高維稀疏TF-IDF向量降維至低維稠密向量
           （潛在語意空間），讓同義詞/共現詞在向量空間中位置接近；
        3. 查詢字串使用同一組已訓練好的 vectorizer/svd 轉換，
           確保查詢向量與chunk向量落在同一語意空間中，可直接算cosine。

    限制（務必告知使用者/寫入報告的方法論章節）：
        - 語意品質高度依賴知識庫本身的語料規模與詞彙重疊度，
          知識庫越小、語意泛化能力越弱（不像預訓練Transformer
          模型看過大規模語料，能理解「退貨」與「退款」的語意關聯）；
        - 依賴分詞結果（沿用 `VeloGuardHybridRetriever._tokenize`），
          對中文僅做逐字切分，語意粒度較粗；
        - 僅建議作為網路受限環境的「離線驗證/展示方案」，
          正式上線仍應改用 SentenceTransformer 多語言模型。

    使用方式：
        encoder = LocalLSAEncoder(corpus_texts=[...知識庫所有chunk文本...])
        retriever = VeloGuardHybridRetriever(
            knowledge_base_path="knowledge_base_de_nl.json",
            embed_model=encoder,
        )
    """

    def __init__(self, corpus_texts: list[str], n_components: int = 64) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        if not corpus_texts:
            raise ValueError("corpus_texts 不可為空，需要知識庫語料才能訓練LSA空間。")

        # n_components 不可超過語料筆數或詞彙表大小，否則 TruncatedSVD 會報錯
        effective_dim = max(1, min(n_components, len(corpus_texts) - 1 or 1))

        self._tokenize = VeloGuardHybridRetriever._tokenize
        self._vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            token_pattern=None,
            lowercase=False,  # _tokenize 內部已處理小寫
        )
        tfidf_matrix = self._vectorizer.fit_transform(corpus_texts)

        self._svd = TruncatedSVD(n_components=effective_dim, random_state=42)
        self._svd.fit(tfidf_matrix)

    def encode(
        self, texts: list[str], convert_to_numpy: bool = True, normalize_embeddings: bool = True
    ) -> np.ndarray:
        """將輸入文本轉換為LSA語意向量（介面與SentenceTransformer.encode相容）。"""
        tfidf_vecs = self._vectorizer.transform(texts)
        dense_vecs = self._svd.transform(tfidf_vecs)
        dense_vecs = np.asarray(dense_vecs, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense_vecs = dense_vecs / norms

        return dense_vecs


class NullEncoder:
    """最後備援：完全不做語意編碼，一律回傳零向量。

    用途：
        當 sentence-transformers（連帶torch）與 scikit-learn 都無法在
        目前環境安裝成功時（例如極新的Python版本，兩者的相依套件都還
        沒推出對應的安裝包），系統仍應能啟動——只是退化為「純BM25關鍵字
        檢索」，語意向量分數一律為0（等同該次查詢alpha的語意部分完全不
        起作用）。這比整個系統直接啟動失敗好。

    注意：
        使用此編碼器時，建議前端將 alpha 提示使用者調低（甚至設為0），
        因為dense分數恆為0，過高的alpha會讓大量候選並列同分，排序主要
        仍靠BM25決定。
    """

    def encode(
        self, texts: list[str], convert_to_numpy: bool = True, normalize_embeddings: bool = True
    ) -> np.ndarray:
        return np.zeros((len(texts), 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# 通用載入工具：自動偵測網路狀況，於SentenceTransformer與離線LSA間切換
# ---------------------------------------------------------------------------

def _try_load_sentence_transformer(timeout_sec: float = 8.0) -> Optional["SentenceTransformer"]:
    """在有限時間內嘗試下載/載入SentenceTransformer，逾時或失敗回傳None。

    背景：huggingface_hub 在連線被拒（如 403）時可能仍會依內建重試/
    backoff機制多次嘗試，導致明顯延遲；這裡用執行緒+timeout強制
    限制等待時間，避免網路受限環境下使用者等待過久才看到降級訊息。

    若 sentence-transformers 套件本身就沒裝成功（`_SENTENCE_TRANSFORMERS_
    AVAILABLE` 為False，例如torch尚未推出支援目前Python版本的安裝包），
    直接回傳None，不浪費時間嘗試。
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    import concurrent.futures

    # 注意：不用 `with` context manager，避免離開區塊時
    # ThreadPoolExecutor.shutdown(wait=True) 阻塞等待背景執行緒
    # （下載重試）跑完，讓timeout形同虛設；背景執行緒逾時後
    # 若仍在重試，會隨行程結束自然終止，不影響後續流程。
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(SentenceTransformer, VeloGuardHybridRetriever.DEFAULT_EMBED_MODEL)
    try:
        result = future.result(timeout=timeout_sec)
        pool.shutdown(wait=False)
        return result
    except Exception:
        pool.shutdown(wait=False)
        return None


def load_retriever_with_auto_fallback(
    knowledge_base_path: str, timeout_sec: float = 8.0, verbose: bool = True
) -> tuple[VeloGuardHybridRetriever, str]:
    """載入知識庫並建立檢索器，自動在「真實多語言模型」與「離線LSA替代方案」間切換。

    此函式是 `app.py`（Streamlit前端）與本模組 `__main__` 測試區塊共用的
    載入邏輯：先嘗試下載/載入 SentenceTransformer 多語言模型；若在時限內
    失敗（例如網路白名單不含 huggingface.co），則改用 `LocalLSAEncoder`
    在知識庫語料上即時訓練離線語意向量，確保系統在任何網路環境下都能
    正常啟動。

    Args:
        knowledge_base_path: 知識庫JSON檔案路徑（如 knowledge_base_de_nl.json）。
        timeout_sec: 嘗試下載SentenceTransformer的等待秒數上限。
        verbose: 是否印出載入狀態訊息（Streamlit介面通常改用UI元件顯示，
            可設為False避免污染終端機log）。

    Returns:
        (retriever, embed_mode) 二元組，embed_mode 為
        "sentence_transformer"、"local_lsa_fallback" 或
        "bm25_only_fallback"（sklearn也無法使用時的最後備援），
        方便前端顯示目前使用的語意引擎狀態徽章。
    """
    if verbose:
        print("正在載入 VeloGuardHybridRetriever（首次執行需下載多語言embedding模型）...")

    st_model = _try_load_sentence_transformer(timeout_sec=timeout_sec)

    if st_model is not None:
        retriever = VeloGuardHybridRetriever(
            knowledge_base_path=knowledge_base_path, embed_model=st_model
        )
        return retriever, "sentence_transformer"

    if verbose:
        print(
            "[警告] 無法在時限內下載 SentenceTransformer 權重"
            "（可能是網路白名單不含 huggingface.co，或套件未安裝成功），"
            "改用離線 LocalLSAEncoder（TF-IDF+LSA）作為替代方案..."
        )

    chunks = VeloGuardHybridRetriever._load_knowledge_base(knowledge_base_path)
    corpus_texts = [VeloGuardHybridRetriever._build_retrieval_text(c) for c in chunks]

    try:
        fallback_encoder = LocalLSAEncoder(corpus_texts=corpus_texts)
        embed_mode = "local_lsa_fallback"
    except ImportError:
        # scikit-learn 也無法使用（例如過新的Python版本尚無對應wheel）：
        # 退到最後備援，語意分數恆為0，僅靠BM25關鍵字檢索運作。
        if verbose:
            print(
                "[警告] scikit-learn 亦無法使用，退回 NullEncoder"
                "（純BM25關鍵字檢索，語意分數恆為0）..."
            )
        fallback_encoder = NullEncoder()
        embed_mode = "bm25_only_fallback"

    retriever = VeloGuardHybridRetriever(
        knowledge_base_path=knowledge_base_path,
        embed_model=fallback_encoder,
    )
    return retriever, embed_mode


# ---------------------------------------------------------------------------
# 測試範例（內建 Mock Data，示範查詢「3500歐元E-Bike落地磨損退貨」）
# ---------------------------------------------------------------------------

def _build_mock_knowledge_base() -> list[dict[str, Any]]:
    """建立模擬假資料，涵蓋德國/荷蘭/共通EU三種管轄範圍，
    用於在沒有真實 knowledge_base_de_nl.json 檔案時也能跑通測試。
    """
    return [
        {
            "chunk_id": "Giant_DE_AGB_007",
            "topic": "value_depreciation",
            "jurisdiction": "DE_EU",
            "product_scope": "All",
            "official_article_ref": "AGB § 3 Widerrufsbelehrung Abs. (6)",
            "summary_zh": "撤回權允許消費者比照實體店面檢視商品；若使用超出檢視範圍導致價值減損或毀損，業者將收取合理的價值賠償（Wertersatz）。",
            "operational_rule": "客服判斷退貨商品的使用痕跡是否超出檢視/簡短試用合理範圍，超出則依減損程度收取合理價值賠償。",
            "original_text_snippet": "Solltest du die Ware... benutzen und hierdurch eine Verschlechterung... eintreten, werden wir Dir hierfür einen angemessenen Wertersatz in Rechnung stellen.",
        },
        {
            "chunk_id": "Giant_DE_Return_014",
            "topic": "return_logistics_cost",
            "jurisdiction": "DE_EU",
            "product_scope": "E_Bike",
            "official_article_ref": "Widerrufsrecht-Seite, 末段",
            "summary_zh": "自行車與E-Bike應交給經銷商，由經銷商專業安全送回GIANT，避免消費者自行拆卸電池寄送。",
            "operational_rule": "客服應引導E-Bike退貨消費者將車輛整體交給經銷商處理，不建議自行拆卸電池郵寄。",
            "original_text_snippet": "Übergebe Fahrräder und E-Bikes bitte an Deinen ausgewählten GIANT Vertragshändler...",
        },
        {
            "chunk_id": "Giant_NL_AGB_004",
            "topic": "value_depreciation",
            "jurisdiction": "NL_EU",
            "product_scope": "All",
            "official_article_ref": "Algemene Voorwaarden Artikel 7.4",
            "summary_zh": "Giant條款字面規定商品一旦使用撤回權即完全失效，惟依荷蘭法律最低標準（BW 6:230s），僅需負擔價值減損賠償，撤回權依然有效，系統採用法律標準而非品牌字面規定。",
            "operational_rule": "荷蘭消費者退回已使用（含試乘）之E-Bike，客服不得以已使用為由拒絕退貨，應由技師檢測磨損程度扣除合理折舊費用，其餘金額全額退還。",
            "original_text_snippet": "De Consument mag het Product niet gebruiken. In geval een Product in gebruik is genomen, vervalt het Herroepingsrecht.",
        },
        {
            "chunk_id": "Giant_NL_Return_004",
            "topic": "return_logistics_cost",
            "jurisdiction": "NL_EU",
            "product_scope": "All",
            "official_article_ref": "Retouren & Annuleren 頁面",
            "summary_zh": "透過官網訂購、經銷商交付的自行車，須先取得客服確認才能退還至經銷商，經銷商確認退貨後數個工作天內退款。",
            "operational_rule": "客服須先主動確認並回覆消費者後，消費者才能將車送到經銷商。",
            "original_text_snippet": "Na bevestiging vanuit onze customer service kun je de fiets... retourneren bij de betreffende Giant retailer.",
        },
        {
            "chunk_id": "COMMON_EU_Withdrawal_Base",
            "topic": "withdrawal_period",
            "jurisdiction": "COMMON_EU",
            "product_scope": "All",
            "official_article_ref": "EU Consumer Rights Directive 2011/83/EU Art. 9",
            "summary_zh": "歐盟消費者權利指令規定，消費者享有自收貨起14天的無條件撤回權，此為歐盟境內所有會員國的共通最低標準。",
            "operational_rule": "客服可將此條款作為所有歐盟國家撤回期限計算的共通基準，各國落地法規細節仍以當地條款為準。",
            "original_text_snippet": "The consumer shall have a period of 14 days to withdraw from a distance... contract, without giving any reason.",
        },
        {
            "chunk_id": "Giant_DE_AGB_006",
            "topic": "return_exclusions",
            "jurisdiction": "DE_EU",
            "product_scope": "Custom_Bike",
            "official_article_ref": "AGB Artikel 7.8",
            "summary_zh": "依消費者規格客製、非預先製造之商品不適用撤回權。",
            "operational_rule": "客服判斷是否為客製化訂單，若符合定義應告知消費者不適用撤回權。",
            "original_text_snippet": "Das Widerrufsrecht besteht nicht bei Verträgen zur Lieferung von Waren, die nicht vorgefertigt sind...",
        },
    ]


def _print_results(country_label: str, results: list[RetrievalResult]) -> None:
    """輔助函式：以易讀格式印出檢索結果。"""
    print(f"\n{'=' * 70}\n【{country_label}】檢索結果 Top-{len(results)}\n{'=' * 70}")
    if not results:
        print("（無符合路由過濾條件的結果）")
        return

    for rank, r in enumerate(results, start=1):
        print(
            f"\n#{rank}  chunk_id={r.chunk_id}  "
            f"final_score={r.final_score:.4f}  "
            f"(bm25_norm={r.bm25_score_norm:.4f}, dense_norm={r.dense_score_norm:.4f})"
        )
        print(f"    主題: {r.topic} | 適用範圍: {r.product_scope} | 管轄: {r.jurisdiction}")
        print(f"    條款依據: {r.official_article_ref}")
        print(f"    摘要: {r.summary_zh}")
        print(f"    客服操作守則: {r.operational_rule}")


if __name__ == "__main__":
    import tempfile
    import os

    # 將 Mock Data 寫成暫存 JSON 檔案，模擬真實 knowledge_base_de_nl.json 的載入流程
    mock_data = {"chunks": _build_mock_knowledge_base()}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp_f:
        json.dump(mock_data, tmp_f, ensure_ascii=False)
        tmp_path = tmp_f.name

    try:
        retriever, embed_mode = load_retriever_with_auto_fallback(tmp_path)
        print(f"（本次使用的語意引擎：{embed_mode}）")

        query = "3500歐元的E-Bike已經落地騎乘試用，消費者現在要求退貨，能扣多少折舊費？"

        # 查詢德國市場
        de_results = retriever.search(
            query=query, target_country="DE", product_type="E_Bike", top_k=3, alpha=0.5
        )
        _print_results("德國 DE / E_Bike", de_results)

        # 查詢荷蘭市場
        nl_results = retriever.search(
            query=query, target_country="NL", product_type="E_Bike", top_k=3, alpha=0.5
        )
        _print_results("荷蘭 NL / E_Bike", nl_results)

        # 額外示範：不限品類，比較 alpha 權重對排序的影響
        print(f"\n{'=' * 70}\n附加測試：alpha=0.8（更偏重語意向量）\n{'=' * 70}")
        nl_results_alpha08 = retriever.search(
            query=query, target_country="NL", top_k=3, alpha=0.8
        )
        _print_results("荷蘭 NL / 不限品類 / alpha=0.8", nl_results_alpha08)

    finally:
        os.unlink(tmp_path)
