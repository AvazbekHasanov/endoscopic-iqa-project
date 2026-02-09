# Endoskopik Rasm Sifati Baholash (IQA) Loyihasi - Uzbek Tahlili

## 📖 Loyija Haqida

**Endoscopic Image Quality Assessment (IQA) Proyekti** - bu endoskopik rasmlarning sifatini real-time rejimida baholash uchun yaratilgan to'liq sistema. U klassik kompyuter ko'rish (computer vision) texnikalarisini va sun'iy neural tarmoqlarni (deep learning) birlashtirib, jarroh-ayniyotchilarning diagnostik aniqligini ta'minlaydi.

### Asosiy Xususiyatlari:
- ✅ **Hech qanday Taqdim Kerak Emas** - Standart rasm bilan taqqoslash shart emas
- ✅ **Real-vaqt Ishlashi** - Har bir kadr 100ms dan tez qayta ishlanadi
- ✅ **Klinik Qo'llanilish** - Jarrohxonada bevosita ishlashi mumkin
- ✅ **Hybrid Yondashuv** - Klassik + Deep Learning metrikalar
- ✅ **API Tayyorligi** - Tibbiy tizimlar bilan integratsiya
- ✅ **Interactive Demo** - Streamlit interfeysi orqali test qilish

---

## 🏗️ Loyijaning Arxitekturasi

Proyekt uch asosiy qismdan iborat:

```
┌─────────────────────────────────────────────────┐
│    ENDOSCOPIC IMAGE QUALITY ASSESSMENT SYSTEM   │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. TRADITIONAL METRICS (Traditional IQA)       │
│     • Laplacian Variance (Blur Detection)       │
│     • Gradient Energy (Sharpness)               │
│     • RMS Contrast                              │
│     • Entropy                                   │
│     • Noise Estimation                          │
│     • Tenengrad                                 │
│     • MSCN Std                                  │
│                                                 │
│  2. DEEP LEARNING MODEL (CNN)                   │
│     • Multi-scale Feature Extraction            │
│     • Spatial Attention Mechanisms              │
│     • Feature Fusion                            │
│     • Lightweight Design (<50MB)                │
│                                                 │
│  3. INFERENCE SYSTEM (Deployment)               │
│     • Real-time Prediction                      │
│     • Video Processing                          │
│     • FastAPI Server                            │
│     • Streamlit Interface                       │
└─────────────────────────────────────────────────┘
```

---

## 📊 KLASSIK METRIKALAR VA ULARNING MA'NOLARI

### 1. **Laplacian Variance: 549.52**

#### 🔍 Nima?
Rasmning **blur** darajasini o'lchash uchun ishlatiladi. Bu metrika rasmning necha darajada tez (sharp) ekanligini bildiradi.

#### 📈 Qiymat Qayerdan Keladi?

```
BOSHLANG'ICH RASM
      ↓
   Laplacian Operatori Qo'llash
   (Matematik Filter: [-1 -1 -1]
                      [-1  8 -1]
                      [-1 -1 -1])
      ↓
Har bir Pixel Qiymatining Tahlili
      ↓
Dispersiya (Variance) Hisoblash
      ↓
NATIJA: 549.52
```

#### 💡 Matematik Formula:
```
Laplacian = ∂²I/∂x² + ∂²I/∂y²
Variance = Σ(Laplacian - Mean)² / N
```

#### 📋 Qiymat Talqini:

| Range | Image Status | Rating |
|--------|------------|---------|
| **< 100** | Severely blurred | ❌ Not diagnostic |
| **100-300** | Blurred | ⚠️ Poor quality |
| **300-600** | Good | ✅ Diagnostic quality |
| **> 600** | Very sharp | 🌟 Ideal quality |

#### 🎯 Endoskopiyada Ma'nosi:
**549.52** - Bu rasm aniqlik bo'yicha **yaxshi**. Jarroh tarkibiy qism va patologiyalarni aniq ko'rishi mumkin.

#### 📍 Kod Joyi:
```
models/traditional/traditional_iqa.py:65-79
```

---

### 2. **RMS Contrast: 0.2203**

#### 🔍 Nima?
Rasmda **brightness** va **darkness** ning farqi - ya'ni rasmning qancha "separated" ekanligini o'lchaydi.

#### 📈 Qiymat Qayerdan Keladi?

```
RASM PIKSELLARINI OLAMIZ
      ↓
GRAYSCALE (To'q-oq) QA'TLANMAGA AYLANTIRAMIZ
      ↓
O'rtacha Pixel Qiymatini Topamiz (Average = 128)
      ↓
Har bir pikselning o'rtachadan FARQINI topamiz
      ↓
Farqlarni KVADRATGA ko'taramiz
      ↓
O'rtachani olamiz
      ↓
ILDIZINI CHIQARAMIZ
      ↓
NATIJA: 0.2203
```

#### 💡 Matematik Formula:
```
RMS_Contrast = √(Σ(Pixel - Mean)² / N)
```

#### 📋 Qiymat Talqini:

| Value | Meaning | Diagnostics |
|--------|---------|------------|
| **0.0-0.1** | Very low contrast | Poor visualization |
| **0.1-0.3** | Low contrast | ⚠️ Moderate concern |
| **0.3-0.5** | Medium contrast | ✅ Acceptable |
| **0.5+** | High contrast | 🌟 Ideal |

#### 🎯 Endoskopiyada Ma'nosi:
**0.2203** - RMS Kontrast **pastroq** bo'lgan. Bu rasmda:
- Ranglar juda o'xshash
- Tarkibiy qismlar farq qilib ko'rinmadi
- **Yaxshilash talab etiladi** - svetlantirish yoki kontrast sozlamasi

#### 🔧 Yaxshilash Usullari:
```python
# Kontrast moslashtirgichlari
1. Histogram equalization
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Gamma correction
4. Exposure adjustment
```

#### 📍 Kod Joyi:
```
models/traditional/traditional_iqa.py:105-125
```

---

### 3. **Noise Estimate: 5.9156**

#### 🔍 Nima?
Rasmda qanchalik **random noise** borligini aniqlaydi. Noise - bu rasmga kelinmagan pixels.

#### 📈 Qiymat Qayerdan Keladi?

```
RASM KICHIK BO'LIMLARGA BO'LINADI
      ↓
LOKAL STANDART DEVIATION HISOBLANADI
      ↓
GRADIENT-BASED NOISE ESTIMATION
      ↓
LAPLACIAN PYRAMIDS ISHLATILADI
      ↓
NATIJA: 5.9156
```

#### 💡 Turli Noise Turlari:

```
1. GAUSSIAN NOISE - Random pixels (sensordan keladi)
2. POISSON NOISE - Light intensity kam bo'lsa kelib chiqadi
3. SALT-PEPPER NOISE - Qora va oq nuqtalar paydo bo'ladi
4. COLOR NOISE - Rangda tebranishlar
```

#### 📋 Qiymat Talqini:

| Value | Noise Level | Image Quality |
|--------|-----------------|-----------|
| **0-2** | Very low | 🌟 Clean, professional |
| **2-5** | Low-medium | ✅ Good |
| **5-10** | Medium | ⚠️ Acceptable |
| **>10** | High | ❌ Poor quality |

#### 🎯 Endoskopiyada Ma'nosi:
**5.9156** - Bu **normal darajada noise**. Rangli JPEG compressed rasmda tipik qiymat. Jarroh hali ham diagnostik aniqlikni saqlaydi.

#### 🔧 Noise Kamaytiirish Usullari:
```python
1. Bilateral Filter - Structure-ni qoldirmay noise-ni kamaytiradi
2. Non-local Means Denoising - Rasm strukturasini saqlaydi
3. Morphological Operations - Kichik noise-larni yo'q qiladi
```

#### 📍 Kod Joyi:
```
models/traditional/traditional_iqa.py:175-205
```

---

### 4. **MSCN Std (Multi-Scale Contrast Normalized Std): 0.6366**

#### 🔍 Nima?
**Multi-Scale Contrast Normalization** qo'llanilgan rasmning **standard deviation** (tarqoqligi). Rasmning turli masshtablarda necha darajada detail-ga ega ekanligini o'lchaydi.

#### 📈 Qiymat Qayerdan Keladi?

```
RASM TURLI MASSHTABLARDA TAHLIL QILINADI
      ↓
LOKAL KONTRAST NORMALIZATSIYA QILINADI
      ↓
GAUSSIAN PYRAMIDS ISHLATILADI
      ↓
STANDART DEVIATION HISOBLANADI
      ↓
NATIJA: 0.6366
```

#### 💡 Matematik Protsess:
```
1. Gaussian filters bilan blur qilamiz (turli radius)
2. Lokal contrast = Original - Blurred
3. Normalization: Contrast / (Mean + 1)
4. Std = √(Σ(MSCN - Mean)² / N)
```

#### 📋 Qiymat Talqini:

| Value | Details | Image Type |
|--------|---------|----------|
| **< 0.3** | Very low | Smooth, no artifacts |
| **0.3-0.7** | Balanced | ✅ Natural image |
| **0.7-1.5** | High | Rich details |
| **> 1.5** | Very high | Noisy images |

#### 🎯 Endoskopiyada Ma'nosi:
**0.6366** - **Ideal balans**. Rasmda:
- Diagnostik detallar ko'rinadi
- Shovqin ortiqcha emas
- Jarroh jarohatning tarkibiy qisimlarini aniq ko'radi

#### 📍 Kod Joyi:
```
models/traditional/traditional_iqa.py:225-270
```

---

### 5. **Gradient Energy: 7992.04**

#### 🔍 Nima?
Rasmda qanchalik **sharp transitions** (o'tkir o'zgarishlar) (edges) borligini o'lchaydi. Gradient - bu pixel qiymatlarining keskin o'zgarishi.

#### 📈 Qiymat Qayerdan Keladi?

```
RASM TAHLILI
      ↓
SOBEL OPERATOR QOLLANILADI
      ↓
X DIRECTION GRADIENTINI TOPAMIZ (Gx)
      ↓
Y DIRECTION GRADIENTINI TOPAMIZ (Gy)
      ↓
MAGNITUDE: √(Gx² + Gy²)
      ↓
ENERGY = Σ(MAGNITUDE²)
      ↓
NATIJA: 7992.04
```

#### 💡 Sobel Operator:
```
Gx Kernel:        Gy Kernel:
[-1  0  1]        [-1 -2 -1]
[-2  0  2]        [ 0  0  0]
[-1  0  1]        [ 1  2  1]
```

#### 📋 Qiymat Talqini:

| Value | Sharpness | Edges | Diagnostics |
|--------|----------|---------|------------|
| **< 500** | Very low | None | ❌ Blurred |
| **500-2000** | Low | Few | ⚠️ Unacceptable |
| **2000-8000** | Medium | Many | ✅ Good |
| **> 8000** | High | Very many | 🌟 Ideal |

#### 🎯 Endoskopiyada Ma'nosi:
**7992.04** - Bu **yuqori qiymat**, rasmda:
- Jarohat kenarları aniq ko'rinadi
- Tarkibiy qisimlar farq qilib ko'rinadi
- Jarroh anatomiyani aniq ko'radi
- **Diagnostik sifat yuqori**

#### 📍 Kod Joyi:
```
models/traditional/traditional_iqa.py:81-104
```

---

### 6. **Entropy (Entropiya): 6.8474**

#### 🔍 Nima?
Information Theory asosida, rasmda qanchalik **"disorder"** yoki **"information content"** borligini o'lchaydi.

#### 📈 Qiymat Qayerdan Keladi?

```
RASM HISTOGRAMI HISOBLANADI
      ↓
RANG TAQSIMOTINI TOPAMIZ
      ↓
PROBABILITY FUNKSIYASI (PROBABILITY)
      ↓
ENTROPY FORMULASI QOLLANILADI:
H = -Σ(p(x) * log₂(p(x)))
      ↓
NATIJA: 6.8474
```

#### 💡 Entropy Formulasi:
```
H = -Σ P(x) × log₂(P(x))

Bu yerda:
P(x) - Har bir rang qiymatining ehtimoli
log₂ - 2 asosli logarifm
Σ - Barcha rang qiymatlarida yig'indi
```

#### 📋 Qiymat Talqini:

| Value | Information | Image Type |
|--------|----------|--------------|
| **0-2** | Very low | Single color image |
| **2-4** | Low | Very limited |
| **4-6** | Medium | ⚠️ Limited information |
| **6-8** | High | ✅ Rich information |
| **8 (max)** | Maximum | All colors equally probable |

#### 🎯 Endoskopiyada Ma'nosi:
**6.8474** - Bu **yaxshi ma'lumot boyitligi**:
- Rasmda ko'p rang variatsiyasi
- Diagnostik detallar saqlanib qolgan
- Jarroh rasm-dan to'liq ma'lumot oladi
- **Sifat yaxshi**

#### 💡 Entropy va Noise Munosabati:
```
LOW ENTROPY        → Toza rasmlar (bitta rang)
MEDIUM ENTROPY     → Natural rasmlar  
HIGH ENTROPY       → Ko'p shovqin bilan rasmlar
```

#### 📍 Kod Joyi:
```
models/traditional/traditional_iqa.py:127-145
```

---

### 7. **Tenengrad: 7627.44**

#### 🔍 Nima?
Gradient Energy ga o'xshash, lekin **Laplacian kernel** ishlatadi. Rasmning **sharpness va edges** aniqligi.

#### 📈 Qiymat Qayerdan Keladi?

```
RASM TAHLILI
      ↓
LAPLACIAN KERNEL QOLLANILADI
      ↓
KENARLAR ANIQLANDI
      ↓
MAGNITUDE HISOBLANADI
      ↓
KVADRATGA KO'TARAMIZ
      ↓
SUMMA OLAMIZ
      ↓
NATIJA: 7627.44
```

#### 💡 Tenengrad Operatori:
```
Laplacian Kernel:
[0 -1  0]
[-1  4 -1]
[0 -1  0]

Tenengrad = Σ(Laplacian(I)²)
```

#### 📋 Qiymat Talqini:

| Value | Sharpness | Edges | Image Quality |
|--------|----------|---------|-----------|
| **< 1000** | Very low | None | ❌ Poor |
| **1000-4000** | Low | Few | ⚠️ Questionable |
| **4000-8000** | High | Many | ✅ Good |
| **> 8000** | Very high | Clear | 🌟 Ideal |

#### 🎯 Endoskopiyada Ma'nosi:
**7627.44** - Gradient Energy (7992) bilan deyarli bir xil:
- Rasmda **ko'p o'tkir kenarlar**
- Tarkibiy qisimlar aniq farq qilib ko'rinadi
- Jarroh patologiyalarni aniq ko'radi
- **Diagnostik sifat jo'da yuqori**

#### 🔬 Gradient Energy vs Tenengrad Farqi:
```
GRADIENT ENERGY = Sobel asosida (X va Y gradientlar)
TENENGRAD = Laplacian asosida (Ikkinchi tartib hosilasi)

Ikkalasi ham sharpness o'lchaydi, lekin:
• Tenengrad - edges-ni aniqroq aniqlaydi
• Gradient Energy - umumiy o'zgarishlarni o'lchaydi
```

#### 📍 Kod Joyi:
```
models/traditional/traditional_iqa.py:147-173
```

---

## 📊 BARCHA METRIKALARNING UMUMIY TAHLILI

### Qiymatlar Jadvali

| Metrika | Qiymat | Baholash | Rasm Holati |
|---------|--------|----------|-----------|
| **Quality Score** | **0.781** | 👍 Yaxshi | Diagnostik sifat yetarli |
| **Laplacian** | **549.52** | 👍 Yuqori | Sharpness yaxshi |
| **RMS Contrast** | **0.2203** | ⚠️ Pastroq | Contrast yetarli emas |
| **Noise** | **5.9156** | 👍 Qabul | Noise normal darajada |
| **MSCN Std** | **0.6366** | 👍 Balans | Details va sharpness balans |
| **Gradient Energy** | **7992.04** | 👍 Yuqori | Ko'p o'tkir detallar |
| **Entropy** | **6.8474** | 👍 Yaxshi | Boyit ma'lumot |
| **Tenengrad** | **7627.44** | 👍 Yuqori | O'tkir edges |

### 🎯 Umumiy Xulosa:

```
┌────────────────────────────────────────────┐
│   RASM SIFAT BAHO: 0.781 (YAXSHI)          │
├────────────────────────────────────────────┤
│ ✅ Sharpness - YUQORI                      │
│ ✅ Details - YAXSHI                        │
│ ⚠️ Contrast - ORTA                         │
│ ✅ Noise - NORMAL                          │
│ ✅ Edges - ANIQ                            │
│ ✅ Information - BOYIT                     │
├────────────────────────────────────────────┤
│ DIAGNOSTIK MUMKINCHILIK: ✅ YETARLI        │
│ JARROHLIK FOYDALANISH: ✅ MUMKIN           │
└────────────────────────────────────────────┘
```

---

## 🔧 OGOHLANTIRISH VA XATOLAR (WARNINGS)

### ⚠️ 1️⃣ UserWarning: `use_column_width` Parametri Deprecated Qilingan

#### Ogohlantirish Matni:
```
The `use_column_width` parameter has been deprecated and will be removed 
in a future release. Please utilize the `use_container_width` parameter instead.
```

#### 🔍 Ogohlantirish Nima Ekanligini Tushuntirish:

**Deprecated** - Bu so'z "eski, foydalanishdan chiqarilgan" demani bildiradi.

```
┌─ OLD METHOD (Deprecated) ──────────────────┐
│ st.image(image, use_column_width=True)      │
│                                             │
│ Bu parametr endi ishlatilmaydigan olddi     │
│ Future versiyalarida o'chiriladi            │
└─────────────────────────────────────────────┘

┌─ NEW METHOD (Recommended) ────────────────┐
│ st.image(image, use_container_width=True)  │
│                                            │
│ Bu yangi usul, batafsil va yaxshi          │
│ Bu parametrni ishlatish talab qilinadi     │
└────────────────────────────────────────────┘
```

#### 📍 Ogohlantirish Joyi:
```
File: /models/deep_learning/__init__.py:58
Error Type: UserWarning
Status: Deprecated Parameter
Time: 6-02-09 08:30:07.729
```

#### 🔧 Xatoni Tuzatish

Qaysi faylni tekshiramiz:

```python
# Eski kod
st.image(image, use_column_width=True)

# Yangi kod (to'g'ri)
st.image(image, use_container_width=True)
```

#### 📝 Batafsil Tushuntirish:

| Parametr | Eski | Yangi | Ma'nosi |
|----------|------|-------|---------|
| **use_column_width** | ✅ Eski (Deprecated) | ❌ Ishlatma | Column width-ga moslash |
| **use_container_width** | ❌ Yo'q | ✅ Yangi (Recommended) | Container width-ga moslash |

#### 💡 Farqi:
```
use_column_width = Faqat stulbeccni en'i bo'yicha
use_container_width = Butun konteyner (oyna) en'iga moslanadi
```

### ⚠️ 2️⃣ Kiritilish Ogohlantirishi: Float32 Conversion

#### Ogohlantirish Matni:
```
Converted deep-learning model parameters to float32 to avoid dtype mismatch.
```

#### 🎯 Bu Ogohlantirish Nima Ekanligini Tushuntirish:

**dtype (data type)** - Ma'lumotning tipi. Masalan: float32, float64, int32 va hokazo.

#### 📍 Ogohlantirish Joyi:
```
File: /models/deep_learning/__init__.py:58
Warning Type: UserWarning
Status: Information (Xavfli emas, shunchaki ma'lumot)
Time: 6-02-09 08:30:07.729
```

#### 🔍 Kod Tahlili:

```python
# models/deep_learning/__init__.py: 48-58 qatorlar

if _orig_get_model is not None:
    def get_model(*args, _force_float=True, **kwargs):
        """
        Calls the discovered get_model and coerces returned torch.nn.Module to float() if present.
        _force_float (bool): if True, call model.float() to ensure float32 parameters/tensors.
        """
        model = _orig_get_model(*args, **kwargs)
        if _force_float:
            try:
                import torch
                if isinstance(model, torch.nn.Module):
                    model = model.float()  # ← float32 ga aylantirish
                    warnings.warn(
                        "Converted deep-learning model parameters to float32 to avoid dtype mismatch.",
                        UserWarning
                    )
```

#### 💡 Tafsilotli Tushuntirish:

```
┌─ MASALA ─────────────────────────────────────┐
│ 1. Model float64 tipida saqlangan            │
│ 2. GPU/Inference float32 tipida kutgan       │
│ 3. Farq bor = "dtype mismatch"               │
│ 4. Natija: Hisob xatosi yoki sekinchilik    │
└──────────────────────────────────────────────┘

┌─ YECHIM ─────────────────────────────────────┐
│ 1. Model parametrlarini float32 ga aylantirdik
│ 2. Endi moslashdi (match)                    │
│ 3. Inference tez ishlaydi                    │
│ 4. Hech qanday muammo qolmadi                │
└──────────────────────────────────────────────┘
```

#### 📊 Data Tipi Jadvali:

| Type | Size | Precision | Speed | Usage |
|-----|-------|--------|--------|---------|
| **float32** | 4 bytes | 7 decimals | 🚀 FASTER | Deep Learning (GPU) |
| **float64** | 8 bytes | 15 decimals | ⚠️ Slower | Scientific computing |
| **int32** | 4 bytes | Integer | 🚀 Fast | Common operations |
| **int64** | 8 bytes | Integer | ⚠️ Slow | Big numbers |

#### 💡 Nima Uchun Float32 Talab Qilinadi?

```
1. TEZLIK ────────────→ Float32 qayta ishlanish 2x tezroq
2. XOTIRA ────────────→ Float32 kam xotira ishlatadi (50% kam)
3. ANIQLIK ───────────→ Deep Learning uchun yetarli aniqlik
4. GPU OPTIMIZATSIYA ─→ CUDA/cuDNN float32 ni tezroq qayta ishlaydi
5. NVIDIA HARDWARE ───→ Tensor Core float32 ni tezroq qayta ishlaydi
```

#### ⚙️ Float32 Konversiyaning Taglibotlari:

```python
# Model float64 da saqlangan
model = model.double()  # float64 (8 bayt)

# Inference uchun float32 ga aylantirish
model = model.float()   # float32 (4 bayt)

# Natija: Muvaffaqiyatli inference
output = model(input)   # Xatosiz ishlaydi
```

#### 🔬 Tekshiruv Kodi:

```python
import torch

# Model yarating
model = torch.nn.Linear(10, 1)

# Default dtype-ni ko'ring
print(model.weight.dtype)  # → torch.float32

# Float64 ga o'tkazing
model = model.double()
print(model.weight.dtype)  # → torch.float64

# Qaytadan float32 ga o'tkazing
model = model.float()
print(model.weight.dtype)  # → torch.float32 ✅
```

#### ✅ Float32 Conversion Issues Resolved:

| Issue | Reason | Solution | Result |
|--------|-------|--------|--------|
| **dtype mismatch** | float64 vs float32 | Conversion | ✅ No errors |
| **Slow performance** | float64 processing | Use float32 | ✅ 2x faster |
| **Memory overhead** | float64 (8 bytes) | Use float32 (4 bytes) | ✅ 50% less |
| **GPU memory** | Excessive usage | Convert to float32 | ✅ Smaller model |

#### 🎯 Xulosa:

Bu ogohlantirish **xavfli emas**, aksincha **foydali ma'lumot**:
- ✅ Model to'g'ri o'zgartirildi
- ✅ Float32 -> optimal tur
- ✅ Ishlashi tez va to'g'ri
- ✅ Xotira kamroq ishlatiladi
- ✅ GPU acceleration ishlaydi
- ✅ Qayta ishlanish normal davom etadi

#### 🔧 Ogohlantirish Tekshiruvi:

```bash
# Terminal da loglarni ko'ring
# Ogohlantirish chiqsa, bu normalni:
# "Converted deep-learning model parameters to float32 to avoid dtype mismatch."

# Bu xatoni tugilishi emas, tashxisiy ma'lumot
# "UserWarning" - Faqat bilish uchun
```

---

## 📂 FAYL TUZILISHI

```
endoscopic-iqa-project/
│
├── 📋 data/                          # Ma'lumot yuklash va qayta ishlash
│   ├── dataset_loader.py            # Datasets yuklash
│   ├── synthetic_degradation.py     # Suniy degradatsiya
│   ├── preprocessing.py             # Tayyorlash (resize, normalize)
│   ├── augmentation.py              # Augmentatsiya (flip, rotate)
│   └── datasets/                    # Actual ma'lumot
│
├── 🤖 models/                        # Modellar
│   ├── traditional/                 # Klassik metrikalar
│   │   └── traditional_iqa.py      # 7 ta klassik metrika
│   ├── deep_learning/               # Deep Learning modeli
│   │   ├── iqa_model.py            # CNN arxitektura
│   │   ├── feature_fusion.py       # Detallarni birlashtirish
│   │   └── attention.py            # Attention mehanizmi
│   ├── pretrained/                 # Oldin o'rgatilgan modellar
│   └── utils.py                    # Utility funksiyalar
│
├── 📚 training/                      # O'rgatish jarayoni
│   ├── train.py                     # O'rgatish skripti
│   ├── trainer.py                   # O'rgatish klassi
│   ├── losses.py                    # Loss funksiyalar
│   └── utils/                       # Utility-lar
│
├── 📊 evaluation/                    # Baholash
│   ├── metrics.py                   # PLCC, SRCC, RMSE, MAE
│   ├── evaluator.py                 # Baholash klassi
│   └── visualization/               # Grafiklar va chartlar
│
├── 🚀 inference/                     # Foydalanish va demo
│   ├── predictor.py                 # Prediction klassi
│   ├── real_time_demo.py            # Streamlit demo
│   └── api/                         # FastAPI server
│
├── ⚙️ configs/                       # Konfiguratsiya fayllar
│   ├── demo_config.yaml             # Demo sozlamalari
│   └── training_config.yaml         # O'rgatish sozlamalari
│
└── 📝 docs/                          # Dokumentatsiya
    ├── api_reference.md             # API hujjatlari
    ├── installation.md              # O'rnatish qo'llanmasi
    └── usage.md                     # Foydalanish qo'llanmasi
```

---

## 🚀 ISHLATISH MISOLLARI

### 1️⃣ Klassik Metrikalar Bilan

```python
from models.traditional.traditional_iqa import TraditionalIQA
import cv2

# Klassik IQA-ni boshlang
iqa = TraditionalIQA()

# Rasmni yuklang
image = cv2.imread('endoscopy_image.jpg')

# Barcha metrikalarni hisoblang
metrics = iqa.compute_all_metrics(image)

# Natijalarni chiqarish
print(f"Laplacian Variance: {metrics['laplacian_variance']:.2f}")
print(f"Gradient Energy: {metrics['gradient_energy']:.2f}")
print(f"Quality Score: {metrics['quality_score']:.3f}")
```

### 2️⃣ Deep Learning Modeli Bilan

```python
from inference.predictor import IQAPredictor

# Modeli yuklang
predictor = IQAPredictor(model_path='models/pretrained/best_model.pth')

# Sifat baho berish
score = predictor.predict('endoscopy_image.jpg')

print(f"Sifat Baho: {score:.3f}")
if score > 0.8:
    print("✅ Yaxshi sifat - Jarrohlik foydalanish mumkin")
elif score > 0.6:
    print("⚠️ O'rtacha sifat - Ehtiyotkorlik bilan foydalaning")
else:
    print("❌ Zaif sifat - O'ng-toshlandi")
```

### 3️⃣ Video Streaming Bilan Real-time

```python
from inference.predictor import IQAPredictor
import cv2

# Modeli yuklang
predictor = IQAPredictor(model_path='models/pretrained/best_model.pth')

# Video-ni oching (kamera yoki fayl)
cap = cv2.VideoCapture(0)  # 0 = built-in kamera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Real-time sifat baholash
    score = predictor.predict(frame)
    
    # Natijani ekranga chiqarish
    color = (0, 255, 0) if score > 0.7 else (0, 165, 255) if score > 0.5 else (0, 0, 255)
    cv2.putText(frame, f'Quality: {score:.3f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Endoscopic IQA', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 💻 O'RNATISH QO'LLANMASI

### 1️⃣ Repository-ni klonlang:
```bash
git clone https://github.com/AvazbekHasanov/endoscopic-iqa-project.git
cd endoscopic-iqa-project
```

### 2️⃣ Virtual Environment yarating:
```bash
python -m venv venv
source venv/bin/activate          # Mac/Linux
# yoki
venv\Scripts\activate              # Windows
```

### 3️⃣ Requirements o'rnatish:
```bash
pip install -r requirements.txt
```

### 4️⃣ Paketni o'rnatish:
```bash
pip install -e .
```

---

## 🧪 TEST VA DEMO

### Streamlit Demo:
```bash
streamlit run inference/real_time_demo.py
```

### FastAPI Server:
```bash
uvicorn inference.api.app:app --reload
```

### Unit Testlar:
```bash
pytest tests/
```

---

## 📈 PERFORMANCE BENCHMARK

| Metric | Value |
|--------|--------|
| **GPU Speed** | <100ms per frame |
| **CPU Speed** | <300ms per frame |
| **Model Size** | ~20MB (compressed) |
| **RAM Usage** | <2GB |
| **PLCC Correlation** | >0.85 |
| **SRCC Correlation** | >0.82 |

---

## 🎓 ILMIY XAVOLA

Agar siz ushbu ishlashni ilmiy maqolangizda qo'llanib, shunday kutubxana qo'shishingiz talab:

```bibtex
@software{endoscopic_iqa_2024,
  title={Endoscopic Image Quality Assessment System},
  author={Hasanov, Avazbek},
  year={2024},
  url={https://github.com/AvazbekHasanov/endoscopic-iqa-project}
}
```

---

## 📞 SAVOL VA JAVOBLAR

### S: Klassik metrikalar vs Deep Learning - Qay biri yaxshi?
**J:** Kettasi birga: Klassik metrikalar tez va tushunarli, Deep Learning - ko'proq aniqlik beradi.

### S: Rasm sifat baho 0-1 oralig'ida nima degani?
**J:** 0 = Zaif sifat, 1 = Ideal sifat. 0.7+ = Diagnostik sifat yetarli.

### S: Real-time video processing uchun GPU kerak?
**J:** Talab emas, lekin tavsiya qilinadi. CPU ishlatishi mumkin, lekin sekinroq (300ms vs 100ms).

### S: Qanday endoskopik rasm turlari qo'llaniladi?
**J:** GI tract, ENT, Urology, Gynecology va boshqa endoskopik sohalar.

---

## 📜 LITSENZIYA

MIT License - Xatosiz foydalanish mumkin.

---

## 🙏 MINNATDORCHILIK

Ushbu proyekt quyidagi tadqiqot ishlari asosida yaratilgan:
- Medical Image Quality Assessment Research
- No-Reference IQA Techniques
- Clinical Deployment Guidelines

---

## ✍️ Yaratuvchi

**Avazbek Hasanov**
- GitHub: @AvazbekHasanov
- Email: hasanov.avazbek@example.com

---

**Keyingi yangilanish:** Fevral 9, 2024

*Ushbu dokumentasiya eng so'nggi tarkibga mos keladi. Savollar uchun GitHub Issues-ga yozing.*
































