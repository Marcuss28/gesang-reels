---
name: "gesang-reels"
description: "Erkennt aus Suno-Vocal-Stems die punktgenauen Gesangseinsätze eines Songs und schneidet daraus 9:16-Reels, die exakt am Gesang starten. Nutzen, wenn Marcus Reels/Shorts mit exaktem Gesangseinstieg oder Gesangs-Zeitstempel aus Stems braucht."
---

# Gesang-Reels: punktgenaue Einstiege

Erkennt aus Suno-Stems, wann Gesang einsetzt/aufhört, und schneidet daraus Reels,
die exakt am Gesang starten. Kommunikation immer auf Deutsch, „Du", knapp.

## Wann verwenden
- Marcus will Reels/Shorts aus einem Song, die genau auf einem Gesangseinsatz starten.
- Oder er braucht nur die Gesangs-Zeitstempel (SRT/CSV) aus den Stems.

## Was ich brauche
1. **Vocal-Stems aus Suno** – mindestens `0 Lead Vocals.wav`, ideal zusätzlich `1 Backing Vocals.wav`.
   (In Suno: Song → „…"-Menü → „Stems". Der API-Connector kostet extra Guthaben – NICHT nutzen.)
2. **Longform-Video** (MP4) des Songs, falls Reels geschnitten werden sollen.
3. Optional Parameter – sonst Standard verwenden.

## Standard-Parameter (Marcus)
- Reel-Länge: **45 s**
- Format: **vertikal 9:16**, Ausgabe **1080×1920**, **30 fps**
- Vorlauf vor dem Gesang: **0,3 s** (damit die erste Silbe nicht abgeschnitten wird)
- Erst **ein Test-Reel**, prüfen lassen, dann Rest.

## Schritt 1 – Gesangserkennung
Skript nach `/tmp/stem_detect.py` schreiben und ausführen. Pfade zu Lead-/Backing-Stem anpassen.
Saubere Stems → einfache RMS-Schwelle (−32 dB unter Peak) ist zuverlässig.

```python
import numpy as np, librosa
SR=22050; HOP=512; FRAME=2048
LEAD="<pfad>/0 Lead Vocals.wav"
BACK="<pfad>/1 Backing Vocals.wav"   # optional, sonst weglassen
OUT="/sessions/.../outputs/"          # aktuelles outputs-Verzeichnis
def rms_db(p):
    y,sr=librosa.load(p,sr=SR,mono=True)
    r=librosa.feature.rms(y=y,frame_length=FRAME,hop_length=HOP)[0]
    t=librosa.frames_to_time(np.arange(len(r)),sr=sr,hop_length=HOP)
    return t,20*np.log10(r+1e-9),len(y)/sr
def smooth(x,sec): 
    w=max(1,int(sec/(HOP/SR))); return np.convolve(x,np.ones(w)/w,mode="same")
def segs(db,t,thr_db=-32,bridge=0.35,minlen=0.35):
    thr=np.percentile(db,99.5)+thr_db
    act=smooth((db>thr).astype(float),0.15)>0.5
    out=[];st=None
    for i,a in enumerate(act):
        if a and st is None: st=t[i]
        elif not a and st is not None: out.append([st,t[i]]);st=None
    if st is not None: out.append([st,t[-1]])
    m=[]
    for s in out:
        if m and s[0]-m[-1][1]<=bridge: m[-1][1]=s[1]
        else: m.append(list(s))
    return [s for s in m if s[1]-s[0]>=minlen]
tL,dbL,dur=rms_db(LEAD); segL=segs(dbL,tL)
try: tB,dbB,_=rms_db(BACK); segB=segs(dbB,tB)
except Exception: segB=[]
comb=[]
for s in sorted(segL+segB):
    if comb and s[0]-comb[-1][1]<=0.35: comb[-1][1]=max(comb[-1][1],s[1])
    else: comb.append(list(s))
def mmss(t): mm=int(t//60);return f"{mm}:{t-60*mm:05.2f}"
def srt_t(t):
    h=int(t//3600);m=int((t%3600)//60);s=int(t%60);ms=int((t-int(t))*1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
gaps=[];prev=0.0
for s in comb:
    if s[0]-prev>=3.0: gaps.append([prev,s[0]])
    prev=s[1]
if dur-prev>=3.0: gaps.append([prev,dur])
open(OUT+"gesang_stems.srt","w").write("".join(f"{i}\n{srt_t(s[0])} --> {srt_t(s[1])}\nGesang\n\n" for i,s in enumerate(comb,1)))
with open(OUT+"gesang_stems.txt","w") as f:
    f.write(f"Gesamtdauer: {mmss(dur)}\n\nGESANG GESAMT:\n")
    for s in comb: f.write(f"  {mmss(s[0])} - {mmss(s[1])}  ({s[1]-s[0]:.1f}s)\n")
    f.write("\nnur LEAD:\n")
    for s in segL: f.write(f"  {mmss(s[0])} - {mmss(s[1])}\n")
    f.write("\nINSTRUMENTAL (>=3s):\n")
    for g in gaps: f.write(f"  {mmss(g[0])} - {mmss(g[1])}\n")
print("Erster Gesangseinsatz:", mmss(comb[0][0]) if comb else "-")
for s in comb: print(f"{mmss(s[0])} - {mmss(s[1])}  ({s[1]-s[0]:.1f}s)")
```

Voraussetzung einmalig: `pip install --break-system-packages -q librosa soundfile` (läuft in ~1 Shell-Aufruf).
Der erste Wert von `comb` ist der erste Gesangseinsatz.

## Schritt 2 – Reel schneiden
Für jeden gewünschten Einsatz `START` (Sekunden aus Schritt 1, minus 0,3 s Vorlauf):

```bash
ffmpeg -y -i "<longform>.mp4" -ss <START-0.3> -t 45 \
  -vf "crop=trunc(ih*9/16/2)*2:ih,scale=1080:1920,setsar=1,fps=30" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 192k \
  "Reel_<name>_9x16_45s.mp4"
```

- Mittiger Zuschnitt; bei Bedarf horizontal verschieben via `crop=...:...:x_offset:0`.
- Optional Fades an den Kanten: `,fade=t=in:st=0:d=0.4,fade=t=out:st=44.6:d=0.4` an den Videofilter anhängen und `-af "afade=t=in:st=0:d=0.4,afade=t=out:st=44.6:d=0.4"`.

## Ausgeben
Ergebnisse mit `present_files` teilen (SRT/CSV/TXT und die Reel-MP4s). Kurz zusammenfassen,
wo der erste Gesangseinsatz liegt und welche instrumentalen Passagen sich als Einstieg eignen.

## Grenzen / Hinweise
- Lange Hall-/Outro-Blöcke können die Schwelle überdehnen → bei Verdacht `thr_db` auf −28 verschärfen.
- Longform-Videos von Suno sind oft quasi statische Standbilder (niedrige fps) → Reel-Bild bewegt sich kaum.
- Wenn nur eine kombinierte Vocal-Datei vorliegt, `BACK` weglassen.
```

