# สรุปหลักการและ Key Messages — `app.py` (Chain System: Main Engine)

> แหล่งข้อมูล: `app.py` (Streamlit, ~1,436 บรรทัด) + `flywheels.py` (engine คำนวณ)

---

## 1. หลักการ (Core Principle)

**"Chain System" = เชื่อมกำไรจากทุก Flywheel เข้าเป็นลูกโซ่** — กำไรจากขั้นหนึ่งกลายเป็น "เชื้อเพลิง" ของขั้นถัดไป วนเป็นวงจร ทั้งขาขึ้นและขาลง

| ช่วง | กลไก |
|---|---|
| 📈 **ขาขึ้น** | กำไร Shannon (+ Harvest) → จ่ายค่า Put Hedge → ส่วนเหลือ = **Surplus** → Scale Up `fix_c` โดยไม่เพิ่มความเสี่ยงของทุนเดิม (**Free Risk**) |
| 📉 **ขาลง** | Put (Hedge) ระเบิดกำไร → เข้า **Pool CF** → นำไป **Deploy** (เมื่อ Regime กลับตัว) + กันส่วนหนึ่งเป็น **EV Reserve** |

พื้นฐานทางทฤษฎี: **Shannon's Demon** (Rebalancing คงมูลค่าคงที่ `fix_c` เพื่อเก็บกำไรจากความผันผวน) + **Put Option hedge** (ประกันขาลง / Convexity) + **Ev (Extrinsic Value)** เป็นต้นทุนเวลาที่ระบบต้อง "ชนะ"

---

## 2. สมการหลัก

| สมการ | ความหมาย |
|---|---|
| `Shannon = fix_c × ln(P_new / P_old)` | กำไรจาก Rebalancing ในรอบนั้น (รายได้ตั้งต้น) |
| `Hedge Cost = qty × BS_Put(S=t, K=0.9t, T=1, r=4%, σ=50%)`, `qty = (fix_c / t) × Hedge×` | ค่าเบี้ยประกัน Put OTM 10% (σ ซ่อนไว้คงที่ 0.5) |
| `Surplus = Shannon − Hedge Cost` | กำไรสุทธิหลังจ่ายประกัน |
| `Scale Up = max(0, Surplus)` → `fix_c_new = fix_c + Scale Up` | เฉพาะ Surplus บวกเท่านั้นที่ขยายหน้าตัก (ติดลบ = ไม่ลด fix_c) |
| `b_new = b_old + c·ln(P/t_old) − c'·ln(P/t_new)` | **Continuous Rollover** — Baseline ต่อเนื่องเมื่อเปลี่ยน `fix_c` และ re-center ราคา `t = P_new` |
| `Ev = Premium − Intrinsic Value` | ต้นทุนเวลาของ LEAPS (Burn) |
| `Ev Efficiency = (Lock P&L + Surplus IV) / Cumulative Ev` | **> 1.0 = Sustainable** (กำไรระบบชนะค่าเช่าเวลา) · **< 1.0 = Bleeding** (ยังต้องพึ่งทิศทางตลาด) |

---

## 3. Key Messages

1. **กำไรต้องเป็นเชื้อเพลิง ไม่ใช่ปลายทาง** — ผลตอบแทนทุกรอบถูกส่งต่อเป็นทุน/ประกันรอบถัดไป
2. **Scale Up จาก Surplus เท่านั้น = Free Risk** — ขยาย `fix_c` ด้วยกำไรที่ผ่านการหักค่าประกันแล้ว ไม่ใช้เงินต้นเพิ่ม
3. **Hedge คือต้นทุนที่ต้องจ่ายก่อนเสมอ** — Surplus คำนวณหลังหัก Hedge Cost; Over-hedge (Hedge× > 1) เร่ง Convexity เมื่อเกิดวิกฤต
4. **Baseline ต่อเนื่อง (Continuous Rollover)** — re-center ราคาแต่ไม่ทำให้ประวัติกำไรขาดตอน
5. **Pool CF = War Chest** — แหล่งพักเงินรวมศูนย์ (จาก Funding, Harvest, Extract Baseline) ก่อน Deploy กลับเข้า Baseline หรือกันเป็น EV Reserve
6. **Ev Burn คือ "ค่าเช่าเวลา"** — เป้าหมายคือให้ Ev Efficiency ≥ 1.0x
7. **Human-in-the-loop** — Preview → แก้ตัวเลขได้ → Commit (ไม่ commit อัตโนมัติ)

---

## 4. โครงสร้างแอป (3 แท็บ)

| แท็บ | หน้าที่ |
|---|---|
| ⚡ **Engine & History** | แถบ Global metrics (Pool CF, EV Reserve, Fix_C, Net) · Watchlist · **Chain Round** (ป้อน `P New` + `Hedge×` → 🔍 Preview → ✅ Commit) · แท็บย่อย: History / Overview (Waterfall) / Treasury / Pool CF / Deploy |
| 🔗 **Payoff Profile** | Simulator โครงสร้างผลตอบแทน log 11 เส้น (Shannon 1/2, Net P/L, Call/Put intrinsic, Long/Short, Benchmark) + **Sankey** แสดง Capital Flow; sync ค่าจาก Preview |
| ➕ **Manage Data** | เพิ่ม Ticker · แก้ Original (P0, fix_c, baseline) · Export/Import JSON (ไฟล์/URL, ต้อง confirm เพราะเขียนทับ) · ลบทั้งหมด |

### Flow การทำงานหนึ่งรอบ
```
ป้อน P New + Hedge×  →  run_chain_round()  →  Preview (แก้ไขได้)
   Shannon → − Hedge Cost → Surplus → Scale Up → fix_c ใหม่, b ใหม่, t = P_new
→ COMMIT → commit_round(): บันทึก rounds[], อัปเดต current_state, สะสม cumulative_ev, save
```

### การเคลื่อนไหวของเงิน (Treasury)
- **Funding / Harvest** → เพิ่ม Pool CF
- **Extract Baseline** → ลด Baseline, เพิ่ม Pool CF
- **Deploy** → ลด Pool CF, เพิ่ม Baseline ของ Ticker
- **Allocate** → Pool CF → EV Reserve · **Record Flow** → ปรับ EV Reserve (Income/Expense)
- ทุกรายการถูกบันทึกใน `treasury_history` (audit trail)

---

## 5. จุดเทคนิคที่น่าสังเกต

- ดึงราคา Yahoo ผ่าน **Chart API + urllib** (ไม่ใช้ yfinance/curl_cffi) เพื่อเลี่ยง segfault บน Streamlit Cloud; cache 60 วินาที รวมผลล้มเหลวกัน retry storm; ลอง `.BK` อัตโนมัติสำหรับ ticker ≤ 5 ตัวอักษร
- Black-Scholes เขียนเอง (ไม่พึ่ง SciPy) — `_norm_cdf` ใน `flywheels.py`
- `baseline` ของ Ticker ใหม่เริ่มที่ **0.0 เสมอ** เพื่อกัน Extract ดึงเงินที่ไม่เคยมีจริง
- ใช้ `_flash` queue + `st.toast` เพื่อให้ feedback รอดข้าม `st.rerun()`
- ข้อมูลเก็บเป็น JSON (`trading_data.json`) พร้อม auto-backup / atomic write / auto-repair `current_state` จาก `rounds[-1]`

---

## 6. ข้อควรระวัง (ข้อสังเกตจากโค้ด)

- `σ = 0.5`, strike = 90%, T = 1 ปี **ฮาร์ดโค้ด** — ค่าเบี้ยจริงในตลาดอาจต่างจากโมเดล (แก้ได้ใน Preview ก่อน Commit)
- `Scale Up = max(0, Surplus)` — รอบที่ขาดทุนไม่ลด `fix_c` (ขาดทุนถูกสะท้อนผ่าน baseline/Net แทน)
- Import ข้อมูล = **เขียนทับทั้งหมด** (มี confirm step แต่ไม่มี undo นอกจากไฟล์ backup)
- สรุปนี้เป็นการอธิบายตรรกะของโค้ด ไม่ใช่คำแนะนำการลงทุน
