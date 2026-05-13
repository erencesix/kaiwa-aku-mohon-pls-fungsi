import 'dotenv/config';
import express from 'express';
import cors from 'cors';

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Health check — Render pings this to know the server is alive
app.get('/health', (_, res) => res.json({ ok: true }));

// POST /transcribe/start
// Receives raw audio bytes, uploads to AssemblyAI, returns jobId.
// Express has no timeout — will wait as long as AssemblyAI needs.
app.post('/transcribe/start', express.raw({ type: '*/*', limit: '500mb' }), async (req, res) => {
  const key = process.env.ASSEMBLYAI_KEY;
  if (!key) return res.status(500).json({ error: 'ASSEMBLYAI_KEY not set.' });
  if (!req.body?.length) return res.status(400).json({ error: 'No audio received.' });

  try {
    // Upload raw bytes to AssemblyAI
    const upload = await fetch('https://api.assemblyai.com/v2/upload', {
      method: 'POST',
      headers: {
        'Authorization': key,
        'Content-Type': 'application/octet-stream',
      },
      body: req.body,
    });

    if (!upload.ok) {
      const t = await upload.text();
      return res.status(502).json({ error: `AssemblyAI upload failed: ${t.slice(0, 200)}` });
    }

    const { upload_url } = await upload.json();

    // Submit transcription job
    const job = await fetch('https://api.assemblyai.com/v2/transcript', {
      method: 'POST',
      headers: { 'Authorization': key, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_url: upload_url,
        language_detection: true,
        punctuate: true,
        format_text: true,
        disfluencies: false,
      }),
    });

    if (!job.ok) {
      const t = await job.text();
      return res.status(502).json({ error: `AssemblyAI job failed: ${t.slice(0, 200)}` });
    }

    const { id: jobId } = await job.json();
    res.json({ jobId });

  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /transcribe/poll?jobId=xxx
app.get('/transcribe/poll', async (req, res) => {
  const key   = process.env.ASSEMBLYAI_KEY;
  const jobId = req.query.jobId;
  if (!key)   return res.status(500).json({ error: 'ASSEMBLYAI_KEY not set.' });
  if (!jobId) return res.status(400).json({ error: 'Missing jobId.' });

  try {
    const r    = await fetch(`https://api.assemblyai.com/v2/transcript/${jobId}`, {
      headers: { 'Authorization': key },
    });
    const data = await r.json();

    if (data.status === 'error') {
      return res.json({ status: 'error', error: data.error });
    }

    if (data.status !== 'completed') {
      return res.json({ status: 'processing' });
    }

    // Build 5-second segments from word timestamps
    const words = data.words || [];
    const segs  = [];
    let ss = null, se = 0, sw = [];

    for (const w of words) {
      if (ss === null) ss = w.start / 1000;
      se = w.end / 1000;
      sw.push(w.text);
      if (se - ss >= 5 || w === words[words.length - 1]) {
        segs.push({ start: ss, end: se, text: sw.join(' ') });
        ss = null; se = 0; sw = [];
      }
    }
    if (!segs.length && data.text) {
      segs.push({ start: 0, end: 0, text: data.text });
    }

    res.json({
      status: 'done',
      result: {
        text:     data.text || '',
        language: data.language_code || null,
        segments: segs,
      },
    });

  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /analyze
// Streams GPT-4o analysis back via SSE. No timeout.
app.post('/analyze', async (req, res) => {
  const key = process.env.OPENAI_API_KEY;
  if (!key) return res.status(500).json({ error: 'OPENAI_API_KEY not set.' });

  const { transcript, segments = [], language = 'id', detectedLanguage = null } = req.body;
  if (!transcript?.trim()) return res.status(400).json({ error: 'Transcript is empty.' });

  const outputLang =
    language === 'en'   ? 'English' :
    language === 'both' ? 'Indonesian and English (write every field in BOTH, separated by " / ", e.g. "Rapat dimulai / Meeting started")' :
    'Indonesian (Bahasa Indonesia)';

  const srcRaw   = (detectedLanguage || '').trim().toLowerCase();
  const srcLabel = srcRaw
    ? srcRaw.charAt(0).toUpperCase() + srcRaw.slice(1)
    : 'the original language';

  const fmt = (s) => {
    const m = Math.floor(s / 60), ss = Math.floor(s % 60);
    return `${m}:${ss.toString().padStart(2, '0')}`;
  };

  const original = segments.length
    ? segments.map(s => `[${s.startFormatted || fmt(s.start)}] [Speaker]: ${s.text}`).join('\n')
    : transcript;

  const segBlock = segments.length
    ? segments.map(s => `[${s.startFormatted || fmt(s.start)}] ${s.text}`).join('\n')
    : transcript;

  const capped = segBlock.length > 14000
    ? segBlock.slice(0, 14000) + '\n[... truncated ...]'
    : segBlock;

  const system = `You are a professional meeting analyst and translator.

Source language: ${srcLabel}
Output language: ${outputLang}

LANGUAGE — ABSOLUTE RULES, NO EXCEPTIONS:
- Write EVERY field in ${outputLang}. No exceptions. Not even one field in another language.
- If output is Indonesian, write Indonesian. If English, write English. If both, write "ID / EN" for every field.
- Do NOT default to English.

CONTENT:
- SPEAKERS: Real names if said, else Speaker A/B/C. Specific summary of their role/contribution.
- CHAPTERS: By topic. Title = actual topic. No decisions here.
- SUMMARY: Past tense. Specific names, numbers, outcomes. No vague statements. Don't repeat key points.
- KEY POINTS: Decisions, commitments, action items only. WHO does WHAT.
- HIGHLIGHTS: 2-4 quotes that are decisive or surprising. Translate them. Explain why each matters.
- TRANSCRIPT: Every utterance translated. Format: "[M:SS] [Name]: text". Include fillers and short replies.

Translation: natural and contextual. Match the formality of the original. Never literal.

Return ONLY valid JSON. No markdown. No code fences. Nothing outside the object.

{
  "speakers": [{"id":"speaker_a","label":"Speaker A","name":null,"role":null,"summary":""}],
  "chapters": [{"title":"","timestamp":"0:00 - 2:30","summary":""}],
  "tabs": {
    "summary": [{"point":"","subPoints":[""]}],
    "keyPoints": [{"point":"","subPoints":[""]}],
    "highlights": [{"speaker":"","quote":"","context":""}]
  },
  "transcripts": {
    "translated": "[0:00] [Speaker A]: text\\n[0:05] [Speaker B]: text"
  }
}`;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.flushHeaders();

  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);
  const hb   = setInterval(() => send({ ping: true }), 5000);

  try {
    const gpt = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
      body: JSON.stringify({
        model: 'gpt-4o',
        max_tokens: 8192,
        temperature: 0.1,
        stream: true,
        messages: [
          { role: 'system', content: system },
          { role: 'user',   content: `Transcript:\n${capped}` },
        ],
      }),
    });

    if (!gpt.ok) {
      const t = await gpt.text();
      let msg = `OpenAI error (${gpt.status})`;
      try { msg = JSON.parse(t).error?.message || msg; } catch {}
      send({ error: msg });
      return;
    }

    let acc = '', buf = '';
    const reader = gpt.body.getReader();
    const dec    = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).trimEnd();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') continue;
        let p; try { p = JSON.parse(raw); } catch { continue; }
        const token = p.choices?.[0]?.delta?.content;
        if (token) { acc += token; send({ token }); }
      }
    }

    const clean = acc.replace(/^```json\s*/m,'').replace(/^```\s*/m,'').replace(/\s*```$/m,'').trim();
    const s = clean.indexOf('{'), e = clean.lastIndexOf('}');
    const jsonStr = s !== -1 && e !== -1 ? clean.slice(s, e+1) : clean;

    let result;
    try { result = JSON.parse(jsonStr); }
    catch { send({ error: 'Failed to parse GPT response. Try a shorter recording.' }); return; }

    if (!result.transcripts) result.transcripts = {};
    result.transcripts.translated = result.transcripts.translated || '';
    result.transcripts.original   = original;
    result._meta = { sourceLang: srcLabel, outputLang: language };

    send({ done: true, result });

  } catch (e) {
    send({ error: 'Analysis failed: ' + e.message });
  } finally {
    clearInterval(hb);
    res.end();
  }
});

app.listen(PORT, () => console.log(`Kaiwa running on port ${PORT}`));
