# Image Tagger 變更紀錄

## v1.4.0 (2025-02-17)

### 新功能
- **xAI (Grok) API 支援** — 新增 xAI Grok Vision 後端
  - 預設模型：`grok-4`, `grok-4-fast`, `grok-4.1`, `grok-4.1-fast`, `grok-3`
  - 可自行輸入任意模型名稱
- **OpenRouter API 支援** — 新增 OpenRouter 統一 API 後端
  - 預設模型：`qwen/qwen-2.5-vl-72b-instruct`, `x-ai/grok-4`, `mistralai/pixtral-large-latest` 等
  - 可自行輸入任意 OpenRouter 模型名稱
- **統一 OpenAI 兼容後端** — xAI 和 OpenRouter 共用 `openai_compatible_api.py`

### UI 改進
- Model Type 從 Radio Button 改為 4 選項下拉選單
- **修復 Generation Settings 重啟後被重設的 bug** — slider 標籤值現在正確恢復
- **新增 Quick Save 按鈕** — 直接覆蓋當前模板，無需輸入名稱
- **System Prompt 文字區塊加大** — 從 80px 改為 200px，並自動展開

### 依賴
- 新增 `openai>=1.0.0` 套件依賴

---

## v1.3.0 (2025-02-05)

### 新功能
- **Qwen3VL 模型支援** - 新增 Qwen3-VL 架構支援
  - 需要安裝 JamePeng 的 llama-cpp-python fork
  - 預編譯 wheel 下載：https://github.com/JamePeng/llama-cpp-python/releases/
  - 支援 CUDA 12.4 ~ 13.0
  - GUI 新增 VLM Type 下拉選單 (Qwen3VL / LLaVA)

### 改進
- 本地模型設定區新增模型類型選擇
- VLM 類型設定會自動儲存/載入

---

## v1.2.0 (2025-02-05)

### Bug 修復
- **修復提示詞系統** - 移除硬編碼的提示詞，現在完全使用模板中的 System Prompt
  - 之前無論如何設定 System Prompt，輸出都是固定的 Danbooru tag 格式
  - 現在模板的提示詞會直接作為主要指令發送給模型

### 改進
- **精簡預設模板** - 從 8 個減少為 2 個：
  - **Danbooru Tag** - 針對 LoRA/Fine-tuning 訓練優化的標籤格式
  - **Natural Caption** - 詳細的自然語言描述格式
- 更新模板提示詞內容，包含完整的分析指南與格式規則

---

## v1.1.0 (2025-02-05)

### 新功能

#### 三欄式 GUI 介面
- **左側：圖片瀏覽器**
  - 顯示資料夾內所有圖片的縮圖
  - 每張圖片有勾選框，可選擇要打標的圖片
  - 狀態指示器：✓ (已打標) / ○ (未打標)
  - Select All / Clear 快速選擇按鈕

- **中央：設定面板**
  - 模型類型選擇 (Local VLM / Gemini API)
  - 生成參數調整 (Temperature, Top-K, Top-P, Min-P, Repeat Penalty)
  - 提示詞模板選擇與管理

- **右側：標籤編輯器**
  - 圖片預覽
  - 對應 .txt 檔案內容顯示/編輯
  - Save / Revert 按鈕
  - 打標過程即時更新

#### 設定持久化
- 所有設定自動儲存至 `%APPDATA%\ImageTagger\config.json`
- 包含：資料夾路徑、模型設定、API Key、滑桿參數、視窗大小
- 下次啟動時自動載入上次設定

#### 系統提示詞模板
- 每個模板包含格式類型 (Tag/Caption) 與系統提示詞
- 支援自訂模板儲存與刪除

### 改進
- 移除獨立的 Output Format 選項，整合至模板系統
- 預設溫度從 0.7 改為 0.4
- Gemini 模型更新至 2025 版本 (gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash)
- 標籤格式改為 Danbooru 風格 (無底線，使用空格)

---

## v1.0.0 (初始版本)

### 功能
- 基本圖片打標功能
- 支援 Local VLM (.gguf) 與 Gemini API
- Captioning / Tag 格式輸出
- 基本 GUI 介面
