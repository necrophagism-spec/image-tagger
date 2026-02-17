# Image Tagger — 專案總結

## 專案概述

**Image Tagger** 是一個基於 Python 的圖片自動打標工具，針對 **AI 繪圖訓練資料集製作** 設計。支援批次處理資料夾中的圖片，為每張圖片產生 Danbooru 風格標籤或自然語言描述，並儲存為同名 `.txt` 檔案。

---

## 核心功能

### 🖼️ 四後端推理引擎

| 後端 | 說明 | 狀態 |
|------|------|------|
| **Local VLM** | 使用 `llama-cpp-python` 離線推理 | ✅ 完成 |
| **Gemini API** | 使用 Google Gemini 雲端推理 | ✅ 完成 |
| **xAI (Grok)** | 使用 xAI Grok Vision API | ✅ 完成 |
| **OpenRouter** | 統一 API 聞道，支援 50+ 模型 | ✅ 完成 |

### 🤖 支援的模型

**本地模型 (GGUF):**

| 架構 | 說明 | 依賴 |
|------|------|------|
| **LLaVA** | 傳統 LLaVA 1.5 模型 | 標準 `llama-cpp-python` |
| **Qwen3VL** | Qwen3-VL 視覺語言模型 | [JamePeng fork](https://github.com/JamePeng/llama-cpp-python/releases/) |

**API 模型 (預設列表，可自行輸入):**

| 後端 | 預設模型 |
|------|----------|
| Gemini | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash` |
| xAI | `grok-4`, `grok-4-fast`, `grok-4.1`, `grok-4.1-fast`, `grok-3` |
| OpenRouter | `qwen/qwen-2.5-vl-72b-instruct`, `x-ai/grok-4`, `mistralai/pixtral-large-latest` 等 |

### 🎨 三欄式 GUI 介面

```
┌──────────────┬──────────────────┬──────────────────┐
│   圖片瀏覽器   │     設定面板      │    標籤編輯器    │
│              │                  │                  │
│  ☑ image1.jpg │  模型選擇         │  [圖片預覽]      │
│  ☑ image2.png │  VLM Type 選擇    │                  │
│  ☐ image3.webp│  生成參數調整      │  tags / caption  │
│              │  提示詞模板        │                  │
│  [全選] [清除] │  [開始] [停止]     │  [儲存] [還原]   │
└──────────────┴──────────────────┴──────────────────┘
```

### 📝 提示詞模板系統

兩個預設模板：

| 模板 | 用途 |
|------|------|
| **Danbooru Tag** | 產生 LoRA/Fine-tuning 訓練用的分類標籤 |
| **Natural Caption** | 產生詳細自然語言圖片描述 |

支援自訂模板的新增、修改與刪除。

### ⚙️ 生成參數

- Temperature、Top-K、Top-P、Min-P、Repeat Penalty
- 所有參數透過 GUI 滑桿即時調整
- **重開程式後自動恢復上次設定**

### 💾 設定持久化

所有設定自動儲存至 `%APPDATA%\ImageTagger\config.json`，包含：資料夾路徑、模型設定、API Key（Base64 混淆）、生成參數、視窗大小。

---

## 專案架構

```
Antigravity/
├── main.py                       # 程式進入點
├── requirements.txt              # Python 依賴
├── 啟動 Image Tagger.bat          # 快速啟動批次檔
│
├── core/                         # 核心邏輯
│   ├── config_manager.py         # 設定管理 (JSON 持久化)
│   ├── gemini_api.py             # Gemini API 後端
│   ├── openai_compatible_api.py   # xAI / OpenRouter 後端 (統一 OpenAI 後端)
│   ├── image_processor.py        # 圖片檔案操作
│   ├── local_vlm.py              # 本地 VLM 推理 (LLaVA + Qwen3VL)
│   ├── prompt_templates.py       # 提示詞模板管理
│   └── tagger.py                 # 統一打標介面
│
├── gui/
│   └── app.py                    # CustomTkinter GUI
│
└── CHANGELOG.md
```

---

## 開發進度

### ✅ 已完成

- [x] v1.0.0 — 基本圖片打標功能 (Local VLM + Gemini API)
- [x] v1.1.0 — 三欄式 GUI 介面、設定持久化、提示詞模板
- [x] v1.2.0 — 修復提示詞系統、精簡預設模板至 2 個
- [x] v1.3.0 — Qwen3VL 模型架構支援 (GUI + 後端)
- [x] v1.4.0 — xAI (Grok) + OpenRouter API 支援、UI 優化
- [x] JamePeng wheel 下載與安裝 (`cp314`, CUDA 13.0)

### ⚠️ 待確認

- [ ] CUDA Toolkit 13.0 安裝完成後，驗證 `Qwen3VLChatHandler` import
- [ ] 使用實際 Qwen3-VL GGUF 模型進行端對端測試
- [ ] 使用實際 LLaVA GGUF 模型進行端對端測試

### 💡 未來可考慮

- [ ] 批次處理進度條改進
- [ ] 多語言介面支援
- [ ] 更多 VLM 架構支援

---

## 技術依賴

| 套件 | 用途 | 版本 |
|------|------|------|
| `customtkinter` | GUI 框架 | latest |
| `Pillow` | 圖片處理 | latest |
| `google-genai` | Gemini API | latest |
| `openai` | xAI / OpenRouter API | latest |
| `llama-cpp-python` | 本地推理 (JamePeng fork) | 0.3.23+cu130 |

**Python 版本**：3.14  
**CUDA**：需要 CUDA Toolkit 13.0（驅動 13.1 相容）

---

## 快速啟動

```powershell
# 直接啟動
python main.py

# 或使用批次檔
.\啟動 Image Tagger.bat
```
