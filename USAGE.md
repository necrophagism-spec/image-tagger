# Image Tagger 使用說明

## 系統需求

- Python 3.10+
- Windows / macOS / Linux

## 安裝

```bash
cd c:\Users\USER\Desktop\Antigravity
pip install -r requirements.txt
```

### 依賴套件
- `customtkinter` - 現代化 GUI 框架
- `Pillow` - 圖片處理
- `google-genai` - Gemini API
- `llama-cpp-python` - 本地 VLM (可選)

---

## 啟動程式

```bash
python main.py
```

---

## 操作流程

### 1. 選擇圖片資料夾

1. 點擊上方 **Browse** 按鈕
2. 選擇包含圖片的資料夾
3. 左側會顯示所有圖片縮圖

### 2. 選擇要打標的圖片

- **全選**：點擊 **Select All** 按鈕
- **清除**：點擊 **Clear** 按鈕
- **個別選擇**：點擊每張圖片旁的勾選框

### 3. 設定模型

#### 使用 Gemini API (推薦)
1. 選擇 **Gemini API** 選項
2. 輸入您的 [Google AI Studio API Key](https://aistudio.google.com/apikey)
3. 選擇模型 (建議 gemini-2.5-flash)

#### 使用 Local VLM
1. 選擇 **Local VLM** 選項
2. 點擊 **...** 選擇 .gguf 模型檔案
3. 選擇對應的 mmproj.gguf 投影器
4. 點擊 **Load Model** 載入模型

### 4. 選擇模板

從 **Prompt Template** 下拉選單選擇：

| 模板 | 輸出格式 | 說明 |
|------|---------|------|
| Danbooru Tags | Tag | 通用標籤格式 |
| Anime Tags | Tag | 動漫角色標籤 |
| Photo Tags | Tag | 攝影標籤 |
| NSFW Tags | Tag | 成人內容標籤 |
| Natural Caption | Caption | 自然語言描述 |
| Anime Caption | Caption | 動漫風格描述 |
| Photo Caption | Caption | 攝影風格描述 |
| NSFW Caption | Caption | 成人內容描述 |

### 5. 調整參數 (可選)

- **Temperature**: 控制輸出隨機性 (0-2)
- **Top-K**: 取樣候選數量 (1-100)
- **Top-P**: 核心取樣閾值 (0-1)
- **Min-P**: 最小機率閾值 (僅 Local VLM)
- **Repeat Penalty**: 重複懲罰 (僅 Local VLM)

### 6. 開始打標

1. 點擊 **▶ Start Tagging** 開始處理
2. 下方進度條顯示處理進度
3. 右側編輯器即時顯示結果
4. 完成後會顯示處理數量

### 7. 編輯標籤

1. 點擊左側縮圖選擇圖片
2. 右側編輯器顯示對應 .txt 內容
3. 手動編輯文字
4. 點擊 **💾 Save** 儲存
5. 點擊 **↩ Revert** 還原

---

## 自訂模板

### 儲存新模板
1. 在 System Prompt 欄位輸入提示詞
2. 點擊 **Save As...** 按鈕
3. 輸入模板名稱

### 刪除模板
1. 選擇要刪除的模板
2. 點擊 **Delete** 按鈕
3. 確認刪除 (預設模板無法刪除)

---

## 設定檔位置

| 檔案 | 位置 |
|------|------|
| 設定檔 | `%APPDATA%\ImageTagger\config.json` |
| 模板檔 | `%APPDATA%\ImageTagger\prompt_templates.json` |

---

## 輸出格式

### Tag 格式
```
1girl, long hair, blonde hair, blue eyes, school uniform, smile, looking at viewer
```

### Caption 格式
```
A young woman with long blonde hair and bright blue eyes stands in a sunlit classroom. She wears a traditional school uniform consisting of a white blouse and pleated navy skirt. Her expression is warm and inviting as she looks directly at the viewer with a gentle smile.
```

---

## 快捷鍵

目前無快捷鍵支援，所有操作透過 GUI 完成。

---

## 疑難排解

### Gemini API 錯誤
- 確認 API Key 正確
- 確認網路連線正常
- 檢查 API 配額

### 本地模型載入失敗
- 確認模型架構支援 (目前支援 LLaVA 系列)
- Qwen3-VL 等新架構可能尚未支援
- 確認 mmproj 檔案與模型匹配

### GUI 顯示異常
- 嘗試調整視窗大小
- 重新啟動程式
