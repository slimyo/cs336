# scripts/analyze_snapshot.py
import pickle, sys

path = sys.argv[1] if len(sys.argv) > 1 else "snapshot.pickle"
snap = pickle.load(open(path, "rb"))

def is_active(b):
    return b.get("state") == "active_allocated"   # 精确匹配

def frames_of(x):
    return x.get("frames") or []

def show_frames(x, n=6):
    fr = frames_of(x)
    if not fr:
        print("      (无堆栈)")
        return
    for f in reversed(fr[-n:]):
        print(f"      {f.get('filename','?')}:{f.get('line')}  in {f.get('name','?')}")

# ── 1) 当前存活的分配（segments）──────────────────────────
rows = []
for seg in snap.get("segments", []):
    for b in seg.get("blocks", []):
        if is_active(b):
            rows.append((b["size"], seg.get("device"), b))

rows.sort(key=lambda x: -x[0])
print(f"== 存活分配: {len(rows)} 个, "
      f"{sum(s for s,_,_ in rows)/2**30:.2f} GiB ==\n")
for i, (size, dev, b) in enumerate(rows[:10]):
    addr = b.get("address", b.get("addr"))
    req = b.get("requested_size", size)
    print(f"#{i:02d}  size={size/2**20:9.2f} MiB  requested={req/2**20:9.2f} MiB"
          f"  dev={dev}  addr={addr:#x}")
    show_frames(b)
    print()


# ── 2) 历史最大 alloc（device_traces，含已释放的瞬态张量）──
events = []
for tr in snap.get("device_traces", []):
    for e in tr:
        if str(e.get("action", e.get("phase", ""))).lower() in ("alloc", "allocate"):
            events.append(e)

events.sort(key=lambda e: -e.get("size", 0))
print(f"== 历史最大分配 (traces, 含已释放): 共 {len(events)} 次 alloc ==\n")
for i, e in enumerate(events[:10]):
    print(f"#{i:02d}  {e['size']/2**20:9.2f} MiB")
    show_frames(e, n=5)
    print()
