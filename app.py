"""
=============================================================================
  Yapay Zeka Destekli Fatura İşleme (OCR & Veri Çıkarımı) SaaS — MVP
=============================================================================
  Teknoloji Yığını:
    • Arayüz          : Streamlit
    • Veri İşleme      : Pandas + openpyxl
    • Görsel İşleme    : pdf2image + Pillow
    • Yapay Zeka (OCR) : Google Gemini 1.5 Flash (multimodal)
=============================================================================
"""

# ── Standart Kütüphaneler ─────────────────────────────────────────────────
import os
import io
import json
import re
from typing import Any

# ── Üçüncü Parti Kütüphaneler ────────────────────────────────────────────
import streamlit as st
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# pdf2image, poppler bağımlılığına sahiptir.
# Windows'ta poppler kurulumu aşağıda açıklanmıştır.
from pdf2image import convert_from_bytes

# ── Ortam Değişkenlerini Yükle ────────────────────────────────────────────
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# 1) SAYFA YAPILANDIRMASI
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Fatura İşleme",
    page_icon="🧾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Özel CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Ana başlık */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Metrik kartları */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f1ff 100%);
        border: 1px solid #e0e0ff;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(102,126,234,0.08);
    }

    /* DataFrame tablosu */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e0e0ff;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Upload alanı */
    section[data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 14px;
        padding: 10px;
        background: #fafaff;
    }

    /* İndir butonu */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.6rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(102,126,234,0.35);
    }

    /* Expander */
    details[data-testid="stExpander"] {
        border: 1px solid #e0e0ff;
        border-radius: 10px;
    }

    /* Ayırıcı çizgi */
    hr { border-color: #e0e0ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# 2) YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════

def get_gemini_model() -> genai.GenerativeModel:
    """
    Gemini API istemcisini yapılandırır ve modeli döndürür.
    API anahtarı önce Streamlit secrets, sonra .env dosyasından aranır.
    """
    # Streamlit Cloud'da secrets desteği, yerelde .env desteği
    api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error(
            "⚠️ **GEMINI_API_KEY** bulunamadı! Lütfen `.env` dosyanıza veya "
            "Streamlit secrets'a API anahtarınızı ekleyin."
        )
        st.stop()

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def pdf_to_image(pdf_bytes: bytes) -> Image.Image:
    """
    PDF dosyasının ilk sayfasını PIL Image nesnesine dönüştürür.
    Arka planda poppler kullanılır.
    """
    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200)
    return images[0]


def extract_invoice_data(model: genai.GenerativeModel, image: Image.Image) -> dict[str, Any]:
    """
    Görseli Gemini modeline gönderir ve fatura verilerini JSON olarak çıkarır.
    
    Dönen JSON yapısı:
      {
        "firma_adi": str,
        "vergi_no": str,
        "tarih": str,
        "fatura_no": str,
        "ara_toplam": float,
        "kdv_toplam": float,
        "genel_toplam": float,
        "urunler": [
          {"urun_adi": str, "miktar": int/float, "birim_fiyat": float, "toplam": float}
        ]
      }
    """
    prompt = (
        "Sen uzman bir muhasebe asistanısın. "
        "Gönderdiğim faturadaki bilgileri analiz et ve BİREBİR şu JSON formatında, "
        "başka hiçbir açıklama yapmadan dön:\n\n"
        '{"firma_adi": "...", "vergi_no": "...", "tarih": "...", "fatura_no": "...", '
        '"ara_toplam": 0.0, "kdv_toplam": 0.0, "genel_toplam": 0.0, '
        '"urunler": [{"urun_adi": "...", "miktar": 0, "birim_fiyat": 0.0, "toplam": 0.0}]}'
        "\n\nÖnemli kurallar:\n"
        "- Sadece geçerli JSON döndür, başka metin yazma.\n"
        "- Sayısal değerleri ondalık sayı olarak yaz (string değil).\n"
        "- Eğer bir bilgiyi bulamıyorsan ilgili alanı boş string veya 0.0 olarak bırak.\n"
        "- JSON anahtarlarını aynen koru, değiştirme."
    )

    response = model.generate_content([prompt, image])
    raw_text = response.text.strip()

    # Gemini bazen ```json ... ``` bloğu içinde döndürebilir — temizle
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        st.error("❌ Yapay zekadan gelen yanıt geçerli bir JSON değil. Ham yanıt aşağıda:")
        st.code(raw_text, language="json")
        st.stop()

    return data


def build_products_dataframe(products: list[dict]) -> pd.DataFrame:
    """
    Ürün listesini düzenli bir Pandas DataFrame'e dönüştürür.
    Sütun başlıklarını Türkçeleştirir.
    """
    df = pd.DataFrame(products)

    column_map = {
        "urun_adi": "Ürün Adı",
        "miktar": "Miktar",
        "birim_fiyat": "Birim Fiyat (₺)",
        "toplam": "Toplam (₺)",
    }
    df.rename(columns=column_map, inplace=True)

    # Sayısal sütunları float'a zorla
    for col in ["Miktar", "Birim Fiyat (₺)", "Toplam (₺)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Pandas DataFrame'i bellekte bir Excel (.xlsx) dosyasına dönüştürür
    ve byte dizisi olarak döndürür (indirme için).
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Fatura Ürünleri")
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# 3) ANA ARAYÜZ
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Streamlit uygulamasının giriş noktası."""

    # ── Başlık ────────────────────────────────────────────────────────────
    st.markdown('<p class="main-title">🧾 AI Fatura İşleme</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Faturanızı yükleyin — yapay zeka verileri sizin için çıkarsın.</p>',
        unsafe_allow_html=True,
    )

    # ── Nasıl Çalışır? (Bilgi notu) ──────────────────────────────────────
    with st.expander("ℹ️ Nasıl Çalışır?", expanded=False):
        st.markdown(
            """
            | Adım | Açıklama |
            |:----:|:---------|
            | **1. Yükle** | PDF, PNG veya JPG formatında fatura dosyanızı yükleyin. |
            | **2. İşle** | Yapay zeka (Gemini 1.5 Flash) faturadaki tüm bilgileri otomatik olarak çıkarır. |
            | **3. İndir** | Çıkarılan verileri tablo olarak görüntüleyin ve Excel dosyası olarak indirin. |
            """
        )

    st.divider()

    # ── Dosya Yükleme ────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📎 Fatura dosyanızı yükleyin",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Desteklenen formatlar: PDF, PNG, JPG",
    )

    if uploaded_file is None:
        st.info("👆 Başlamak için bir fatura dosyası yükleyin.")
        return

    # ── Ön İşleme — Görseli Hazırla ──────────────────────────────────────
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.type  # e.g., "application/pdf", "image/png"

    image: Image.Image | None = None

    if "pdf" in file_type:
        with st.spinner("📄 PDF görsele dönüştürülüyor…"):
            try:
                image = pdf_to_image(file_bytes)
            except Exception as exc:
                st.error(
                    f"PDF dönüştürme hatası: `{exc}`\n\n"
                    "**Çözüm:** Poppler'ın sisteminizde kurulu olduğundan emin olun."
                )
                st.stop()
    else:
        image = Image.open(io.BytesIO(file_bytes))

    # Yüklenen görseli küçük bir ön izleme ile göster
    with st.expander("🖼️ Yüklenen Fatura Ön İzlemesi", expanded=True):
        st.image(image, use_container_width=True)

    st.divider()

    # ── LLM ile Veri Çıkarımı ────────────────────────────────────────────
    with st.spinner("🤖 Yapay zeka faturayı analiz ediyor… Lütfen bekleyin."):
        model = get_gemini_model()
        invoice_data = extract_invoice_data(model, image)

    st.success("✅ Fatura başarıyla analiz edildi!")

    # ── Genel Fatura Bilgileri (Metrikler) ────────────────────────────────
    st.subheader("📋 Fatura Genel Bilgileri")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏢 Firma Adı", invoice_data.get("firma_adi", "—"))
        st.metric("📅 Tarih", invoice_data.get("tarih", "—"))
    with col2:
        st.metric("🔢 Vergi No", invoice_data.get("vergi_no", "—"))
        st.metric("🧾 Fatura No", invoice_data.get("fatura_no", "—"))

    st.divider()

    # ── Finansal Özet ─────────────────────────────────────────────────────
    st.subheader("💰 Finansal Özet")

    fin_col1, fin_col2, fin_col3 = st.columns(3)
    with fin_col1:
        st.metric("Ara Toplam", f"₺{invoice_data.get('ara_toplam', 0):,.2f}")
    with fin_col2:
        st.metric("KDV Toplam", f"₺{invoice_data.get('kdv_toplam', 0):,.2f}")
    with fin_col3:
        st.metric("Genel Toplam", f"₺{invoice_data.get('genel_toplam', 0):,.2f}")

    st.divider()

    # ── Ürün Tablosu ──────────────────────────────────────────────────────
    st.subheader("📦 Ürün / Hizmet Detayları")

    products = invoice_data.get("urunler", [])
    if products:
        df = build_products_dataframe(products)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Excel İndirme ────────────────────────────────────────────────
        excel_bytes = dataframe_to_excel_bytes(df)
        st.download_button(
            label="📥 Excel Olarak İndir",
            data=excel_bytes,
            file_name=f"fatura_{invoice_data.get('fatura_no', 'veri')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.warning("⚠️ Faturada ürün/hizmet kalemi bulunamadı.")

    # ── Ham JSON (geliştiriciler için) ────────────────────────────────────
    with st.expander("🔍 Ham JSON Yanıtı (Geliştirici)", expanded=False):
        st.json(invoice_data)


# ═══════════════════════════════════════════════════════════════════════════
# 4) UYGULAMA GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
