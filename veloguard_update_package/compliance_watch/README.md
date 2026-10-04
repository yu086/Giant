# compliance_watch — VeloGuard 知識庫每週自動監控系統

## 為什麼需要這個系統

VeloGuard 部署到公開網址後，知識庫裡的內容——VAT/BTW稅率、DHL物流與鋰電池運輸規範、
德國/荷蘭官方法律條文（BGB、BW）、Giant自己的AGB/退貨政策——都可能隨時間變動。
如果網址是公開的，卻用著半年前甚至更早的規則回答，等於在對外提供**過時且可能錯誤**
的合規資訊，這比完全沒有系統風險更高。

同時，這個專案在早期開發階段就已經實際發生過一次「AI靜默置換原文」的事故
（使用Gemini交叉查證時，AI在轉述有爭議的法律條文時，擅自把文字換成自己覺得更合理
的說法）。這件事直接決定了本系統的核心限制：

> **AI只能負責「初篩+草擬」，絕對不能有任何內容未經人工確認就自動上線。**

## 整體架構

```
GitHub Actions（每週一次排程，或手動觸發）
    │
    ▼
compliance_watch/check_sources.py   ← 主控腳本
    │
    ├─ fetch_utils.fetch_source()   對每個來源頁面抓取內容 + 算雜湊
    │                                （抓取失敗 → 標記fetch_failed，絕不中斷、
    │                                  絕不假裝內容沒變）
    │
    ├─ 比對 snapshot_store.json 裡存的上次雜湊
    │   ├─ 雜湊相同 → 無變動，跳過
    │   └─ 雜湊不同 → 有變動，往下一步
    │
    ├─ ai_rechunk.draft_updated_chunk()   請Claude API草擬新版chunk
    │                                      （強制逐字保留原文，禁止竄改；
    │                                        永遠標記為 pending_human_review）
    │
    └─ 產出：
        ├─ report.md              人類可讀的本週檢查報告
        ├─ proposed_chunks.json   機器可讀的草稿，供PR審核用
        └─ snapshot_store.json    更新後的雜湊記錄
    │
    ▼
若有偵測到變動 → 自動開一個GitHub Pull Request（絕不直接合併）
    │
    ▼
人工審核PR：對照原文、確認AI沒有竄改內容、確認分類正確
    │
    ▼
人工按下Merge → main分支更新 → Streamlit Community Cloud 自動重新部署
```

## 各檔案說明

| 檔案 | 用途 |
|---|---|
| `fetch_utils.py` | 抓取單一URL的內容並計算SHA-256雜湊，用於偵測「內容是否變動」。會優雅處理抓取失敗、識別出只是網域首頁的模糊URL（標記為`manual_only`）、對PDF直接做位元組雜湊。 |
| `ai_rechunk.py` | 呼叫Anthropic API，針對「偵測到變動」的來源草擬新版chunk。System prompt明確要求逐字保留原文、禁止意譯或竄改，輸出一律標記`ai_drafted_pending_human_review`。 |
| `check_sources.py` | 主控腳本：讀取來源清單、逐一檢查、呼叫AI草擬、產出報告與草稿檔案、更新雜湊快照。**絕對不會修改`veloguard_multicountry_chunks.json`本身。** |
| `snapshot_store.json` | 記錄每個來源上次抓到的內容雜湊與檢查時間，用於判斷「這次跟上次比有沒有變」。 |
| `report.md` / `proposed_chunks.json` | 每次執行後產生（不需要手動維護，會被覆寫）。 |
| `../.github/workflows/weekly-compliance-check.yml` | GitHub Actions排程設定，每週一UTC 03:00自動執行一次，也可以在GitHub網頁上手動點擊「Run workflow」立刻執行。 |

## 監控範圍

- **Giant品牌政策**：德國/荷蘭的AGB、退貨頁、撤回權頁面
- **官方法律條文**：荷蘭民法典（BW Boek 6）
- **官方監管機關指引**：荷蘭 ACM（Autoriteit Consument & Markt）
- **第三方物流/危險品規範**：DHL Paket、鋰電池UN3480運輸規範（GWP指南）
- **VAT/BTW稅率**：德國聯邦財政部、荷蘭稽徵機關（Belastingdienst）的官方稅率頁面
  （這兩個來源目前是寫在`check_sources.py`裡的`VAT_RATE_SOURCES`常數，因為
  `veloguard_multicountry_chunks.json`原本的`source_pages`結構並沒有收錄VAT官方頁面）

有4個荷蘭來源（`AGB_NL`、`ReturnAnnuleren`、`Retourvoorwaarden`、`Bezorging`）目前在
知識庫裡登記的URL只是網域首頁（因為原始頁面是JS渲染的單頁應用），系統會將它們標記為
`manual_only`，每週在報告中提醒需要人工自行查看，而不是產生沒有意義的雜湊比對噪音。

## 設定步驟（第一次使用前）

1. **設定Anthropic API金鑰**：到GitHub repo的
   `Settings → Secrets and variables → Actions → New repository secret`，
   新增一個名稱為 `ANTHROPIC_API_KEY` 的secret，值填入你的Anthropic API金鑰。
   **絕對不要**把金鑰直接寫進任何程式碼或yml檔案裡。
2. 確認 `.github/workflows/weekly-compliance-check.yml` 已經一併推送到`main`分支——
   GitHub會自動辨識並依排程執行，不需要額外設定。
3. 如果想立刻測試一次，不用等到下週一：到GitHub repo頁面的
   `Actions → 每週合規來源自動檢查 → Run workflow` 手動觸發。

## 每週人工審核SOP（收到PR之後要做什麼）

1. 打開PR，先看 `compliance_watch/report.md`，了解這次哪些來源有變動。
2. 對每一筆變動，打開 `compliance_watch/proposed_chunks.json`：
   - 對照 `draft.full_text_for_review`（這次抓到的完整原文）與
     `draft.proposed_chunk.original_text_snippet`（AI摘錄的片段），
     確認AI真的是「逐字複製」，沒有換掉任何一個字。
   - 看 `draft.needs_human_judgment`，如果AI寫了東西在這裡，代表AI自己也不確定，
     務必親自判斷。
   - 看 `draft.confidence_note`，了解AI對這份草稿的信心程度。
3. 確認無誤後，**手動**把確認過的內容整合進 `veloguard_multicountry_chunks.json`
   （目前設計上刻意不讓自動化流程碰這個檔案，這一步必須是人工動作）。
4. 一切確認後，合併PR。Streamlit Community Cloud會自動重新部署最新版本。
5. 如果AI草擬失敗（`draft.ok == false`），報告裡會直接寫「請人工直接查看來源網址
   並手動更新對應chunk」——這種情況不需要也不應該再重試AI，直接人工處理即可。

## 已知限制（誠實列出，供專案報告使用）

- GitHub Actions排程的觸發時間是「大約」而非精確到秒，實際執行可能延後數分鐘到數十分鐘。
- 若來源網站當週剛好暫時性故障或封鎖，本系統會如實回報`fetch_failed`並保留舊雜湊，
  下週會繼續嘗試比對，不會漏掉真正的變動，但也代表短暫的網站問題不會被誤判為內容改變。
- AI草擬品質取決於原始頁面文字的清晰度；若頁面是圖片化的條文或高度JS互動介面，
  抓到的純文字可能不完整，這種情況`fetch_utils`會盡量以「文字過短」偵測並標記為
  `fetch_failed`，但無法保證100%攔截所有此類邊界情況。
- 目前VAT官方頁面來源是手動選定的兩個政府網站，並非窮舉所有可能改稅的官方公告管道
  （例如個別的立法草案新聞稿），屬於「合理範圍內的官方來源」而非「絕對完整覆蓋」。
