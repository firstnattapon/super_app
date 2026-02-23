import streamlit as st
import json
import io
from PIL import Image

st.set_page_config(page_title="🧪 Test Gemini Slip Reader", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    .main { background: #0f0f1a; }
    .stApp { background: #0f0f1a; color: #e0e0e0; }
    h1 { color: #7ee8fa; font-family: monospace; }
    .result-box {
        background: #1a1a2e;
        border: 1px solid #7ee8fa44;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        font-family: monospace;
    }
    .success { border-color: #00ff88; color: #00ff88; }
    .error   { border-color: #ff4444; color: #ff4444; }
    .info    { border-color: #7ee8fa; color: #7ee8fa; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Gemini Slip Reader — Test 1010 Bypass")
st.caption("จำลองเหตุการณ์ SlipOK ตอบ 1010 (ธนาคารดีเลย์) แล้วให้ AI อ่านสลิปแทน")

st.divider()

# API Key input
api_key = st.text_input("🔑 GENAI_API_KEY", type="password", placeholder="AIzaSy...")

# Image upload
uploaded = st.file_uploader("📎 อัปโหลดรูปสลิป", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="สลิปที่อัปโหลด", width=300)

st.divider()

if st.button("🚀 เทส Gemini อ่านสลิป (จำลอง 1010)", use_container_width=True, type="primary"):
    if not api_key:
        st.error("❌ กรุณาใส่ GENAI_API_KEY")
        st.stop()
    if not uploaded:
        st.error("❌ กรุณาอัปโหลดรูปสลิป")
        st.stop()

    with st.spinner("🤖 Gemini กำลังอ่านสลิป..."):
        try:
            from google import genai
            from google.genai import types
            from google.genai.types import GenerateContentConfig, HttpOptions

            # Init client — timeout=30 (fix จริง)
            client = genai.Client(
                api_key=api_key,
                http_options=HttpOptions(timeout=30),
            )

            # Optimize image
            img = Image.open(uploaded)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

            st.markdown(f'<div class="result-box info">📐 ขนาดรูปหลัง optimize: {len(img_bytes):,} bytes ({img.width}×{img.height}px)</div>', unsafe_allow_html=True)

            prompt = """
            You are a system to extract data from Thai bank slips.
            Analyze this image.
            1. "amount": The transfer amount (number only, float). Ignore balance available.
            2. "trans_ref": The transaction reference number.

            Return strictly JSON: {"amount": float, "trans_ref": string}
            """

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            raw = response.text.strip()

            # Clean JSON
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            result = json.loads(raw)
            amount = result.get("amount")
            trans_ref = result.get("trans_ref")

            st.markdown(f"""
            <div class="result-box success">
            ✅ Gemini อ่านสลิปสำเร็จ!<br><br>
            💰 amount: <b>{amount}</b><br>
            🔖 trans_ref: <b>{trans_ref}</b>
            </div>
            """, unsafe_allow_html=True)

            # Map check
            MACHINE_MAPPING = {"20.0": "20", "20": "20", "30.0": "30", "30": "30",
                               "30.01": "301", "40.0": "40", "40": "40", "50.0": "50", "50": "50"}

            amt_str = str(amount)
            machine = MACHINE_MAPPING.get(amt_str)
            if not machine:
                try:
                    f = float(amount)
                    if f.is_integer():
                        machine = MACHINE_MAPPING.get(str(int(f)))
                except:
                    pass

            if machine:
                firebase_path = f"{machine}/payment_commands"
                st.markdown(f"""
                <div class="result-box success">
                🎯 Firebase path: <b>{firebase_path}</b><br>
                📤 จะ push: status=work, method=ai_fallback, amount={amount}
                </div>
                """, unsafe_allow_html=True)
                st.success(f"🤖(AI) ได้รับยอด {amount} บาท\n*******เริ่มทำงาน*******")
            else:
                st.markdown(f"""
                <div class="result-box error">
                ⚠️ ยอดเงิน {amount} บาท ไม่ตรงกับราคาเครื่อง<br>
                (ต้องเป็น 20, 30, 40, 50 บาท)
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="result-box error">❌ Error: {e}</div>', unsafe_allow_html=True)

st.divider()
st.caption("ทดสอบ: timeout=30s | model=gemini-2.0-flash | จำลอง SlipOK 1010 bypass")
