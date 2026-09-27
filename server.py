#!/usr/bin/env python3
# barehands: move things on your screen with your bare hands.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""barehands server — serves the hand-tracked air-board on localhost.

localhost = a secure context, which is what lets the browser open your
camera for the tracker page. Nothing here ever leaves your machine.

Endpoints:
  GET  /stage.html, /media/*   the pages + the media airlock
  POST /state                  tracker's ~45Hz scene heartbeat; the response
                               carries queued commands (the command channel)
  GET  /state                  the render page mirrors the scene from here
  POST /cmd                    board commands (your AI -> the board)
  GET  /config                 the barehands.json config (name + orbs)
  GET  /tree?orb=N             a notes orb's folder tree — read-only, JAILED
  GET  /note?f=N/<rel>         one note's text — read-only, JAILED
  GET  /props                  the media airlock as a browsable tree
  GET  /orb                    your assistant's live state (the ring reads it)

Config lives in barehands.json next to this file:
  { "name": "Assistant", "port": 8794,
    "orbs": [ { "title": "Notes", "path": "sample-notes", "kind": "notes" },
              { "title": "Props", "path": "media",        "kind": "media" } ] }

"notes" orbs may point at ANY folder of markdown (an Obsidian vault is
just a folder of markdown). The "media" orb may point anywhere too, so
your props can stay where they already live; a relative path resolves
against the repo. Wherever it points is the airlock: the only place
images and models ever stage from.

Your AI drives the ring by writing tiny files into ./state/ :
  state/state      one word: idle | listening | thinking | speaking
  state/mood.json  {"mood": "green"|"amber"|"red", "ts": <unix time>}
  state/wave.json  {"samples": [0..1 x 64], "ts": <unix time>}
Missing files are fine — the ring just idles.
"""
import json
import time
import os
import io
import wave
import hashlib
import urllib.parse
import urllib.request
import re
from google import genai
from google.genai import types
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Fast Google Images search via SerpApi. Three keys are rotated on quota/rate errors.
SERPAPI_KEYS = [
    "paste-api-here",
    "paste-api-here",
    "paste-api-here",
]
SERPAPI_INDEX = 0

def serpapi_key():
    return SERPAPI_KEYS[SERPAPI_INDEX] if SERPAPI_KEYS else None

def rotate_serpapi_key():
    global SERPAPI_INDEX
    if SERPAPI_KEYS:
        SERPAPI_INDEX = (SERPAPI_INDEX + 1) % len(SERPAPI_KEYS)


# Gemini keys stay server-side. Put one key per line in gemini_keys.txt.
# Up to 100 keys are supported and rotated automatically on quota/rate-limit errors.
KEY_FILE = HERE / "gemini_keys.txt"

def load_gemini_keys():
    keys = []
    env = os.environ.get("GEMINI_API_KEYS", "")
    if env:
        keys.extend(x.strip() for x in env.replace(";", "\n").splitlines() if x.strip())
    if os.environ.get("GEMINI_API_KEY"):
        keys.append(os.environ["GEMINI_API_KEY"].strip())
    if KEY_FILE.exists():
        keys.extend(x.strip() for x in KEY_FILE.read_text(encoding="utf-8").splitlines()
                    if x.strip() and not x.lstrip().startswith("#"))
    return list(dict.fromkeys(keys))[:100]

GEMINI_KEYS = load_gemini_keys()
GEMINI_INDEX = 0

def gemini_client():
    return genai.Client(api_key=GEMINI_KEYS[GEMINI_INDEX]) if GEMINI_KEYS else None

def rotate_gemini_key():
    global GEMINI_INDEX
    if GEMINI_KEYS:
        GEMINI_INDEX = (GEMINI_INDEX + 1) % len(GEMINI_KEYS)


def gemini_generate(**kwargs):
    """Make one request with a fresh client, then close it cleanly."""
    client = gemini_client()
    if client is None:
        raise RuntimeError("No Gemini API keys configured")
    try:
        return client.models.generate_content(**kwargs)
    finally:
        try:
            client.close()
        except Exception:
            pass


JARVIS_PROMPT = """
You are JARVIS, a sophisticated personal AI assistant.

Personality:
- Calm
- Intelligent
- Respectful
- Confident
- Formal
- Slightly witty
- Never annoying

Address the user as "sir" naturally when appropriate.

IMPORTANT:
Keep responses SHORT by default.

For a simple question:
- Usually one sentence.

For a normal question:
- 1 to 3 short sentences.

Only give a detailed explanation when the user specifically asks for one.

Never give unnecessary introductions.
Never repeat the user's question.
Never say "As an AI language model."
Never use excessive enthusiasm.

Speak like a highly capable futuristic personal assistant.

You can answer general questions, explain scientific concepts. Do not claim to have displayed,
created, or manipulated a visual object unless the interface actually performs
that action.

Examples:

User: Hello.
JARVIS: Good to see you, sir. At your service.

User: What is the Sun?
JARVIS: The Sun is the star at the center of our solar system, sir.

User: Who was Einstein?
JARVIS: Albert Einstein was a physicist best known for developing the theory of relativity, sir.

User: Thank you.
JARVIS: Always a pleasure, sir.
"""


def set_jarvis_state(state):
    """Update the ring state without ever making AI failure break the board."""
    try:
        sdir = HERE / "state"
        sdir.mkdir(exist_ok=True)
        (sdir / "state").write_text(state, encoding="utf-8")
    except Exception:
        pass


def load_config():
    cfg = {"name": "Assistant", "port": 8794, "orbs": [],
           # Seconds before a non-idle ring state is treated as stale and
           # shown as idle. Only ever rescues a writer that died without
           # saying goodbye; see the note in /orb.
           "state_timeout_s": 600}
    try:
        cfg.update(json.loads((HERE / "barehands.json").read_text()))
    except Exception:
        pass
    if not cfg.get("orbs"):
        cfg["orbs"] = [
            {"title": "Notes", "path": "sample-notes", "kind": "notes"},
            {"title": "Props", "path": "media", "kind": "media"},
        ]
    for orb in cfg["orbs"]:
        orb["path"] = str(Path(str(orb.get("path", ""))).expanduser())
    return cfg


CONFIG = load_config()
try:
    STATE_TIMEOUT = float(CONFIG.get("state_timeout_s", 600))
except (TypeError, ValueError):
    STATE_TIMEOUT = 600.0


def media_root():
    """The Props orb's folder, resolved. Defaults to the repo's own ./media.

    A notes orb could always point at any folder on disk while the media orb
    was pinned to ./media, and that asymmetry cost real users something: with
    an existing library of props you had to COPY it into the repo to use it.
    Two copies of your own files, and the second one sitting inside a git
    working tree where a single `git add -A` publishes them.

    The Props orb's `path` is honoured the same way a notes orb's is now.
    Point this at the folder you already have; your files stay yours and stay
    out of the repo. A relative path still resolves against the repo, so the
    shipped default is unchanged and an existing config keeps working.
    """
    for orb in CONFIG.get("orbs", []):
        if orb.get("kind") == "media":
            q = Path(str(orb.get("path") or "media")).expanduser()
            return (q if q.is_absolute() else HERE / q).resolve()
    return (HERE / "media").resolve()


def orb_root(i):
    """Resolve a notes orb's jail root, or None."""
    try:
        orb = CONFIG["orbs"][int(i)]
        assert orb.get("kind") == "notes"
        p = Path(orb["path"])
        if not p.is_absolute():
            p = HERE / p
        return p.resolve()
    except Exception:
        return None


_STATE = b"{}"          # latest scene state: tracker POSTs, render GETs
_CMDS = []              # queued board commands (your AI -> tracker)
_ALLOWED = ("add_img", "add_card", "clear", "reset", "hand", "give",
            "yank", "hover", "scroll_note", "widget", "explode", "assemble",
            "present")


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        """Serve /media/* from the configured Props folder, not blindly from ./media.

        THE THIRD PLACE, and the one that would have made this a half-fix. The
        airlock check and the props tree both honour media_root(), but static
        serving resolved against the repo because the base handler is built with
        directory=HERE. Left alone, the tree would have listed a viewer's real
        props and every one of them would have 404'd.
        """
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/media/"):
            root = media_root()
            rel = urllib.parse.unquote(clean[len("/media/"):]).lstrip("/")
            target = (root / rel).resolve()
            # Same containment rule as the airlock: resolve first, then prove
            # the result is inside. A prefix comparison on strings is not it.
            if root == target or root in target.parents:
                return str(target)
            return str(root)
        return super().translate_path(path)

    def end_headers(self):
        # no-store on the page itself so a plain reload always serves
        # current code (Chrome happily caches through reloads otherwise)
        if self.path.split("?")[0].endswith("stage.html"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _image_search(self, query):
        """Fast Google Images lookup: return the first Google image result."""
        q = re.sub(r"\s+", " ", str(query).strip())[:180]
        if not q:
            return []
        last_error = "No image result"
        for _ in range(max(1, len(SERPAPI_KEYS))):
            key = serpapi_key()
            if not key:
                raise RuntimeError("No SerpApi keys configured")
            try:
                params = urllib.parse.urlencode({
                    "engine": "google_images_light",
                    "q": q,
                    "google_domain": "google.com",
                    "hl": "en",
                    "gl": "np",
                    "device": "desktop",
                    "api_key": key,
                })
                req = urllib.request.Request(
                    "https://serpapi.com/search?" + params,
                    headers={"User-Agent": "JARVIS-Naman-Acharya/1.0"},
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    raw = r.read().decode("utf-8", "replace")
                data = json.loads(raw)
                if data.get("error"):
                    raise RuntimeError("SerpApi: " + str(data.get("error")))
                results = data.get("images_results") or []
                if not results:
                    last_error = "SerpApi returned 0 images"
                    raise RuntimeError(last_error)
                first = results[0]
                # SerpApi provides both a Google-hosted thumbnail and the source
                # original. Use the thumbnail first because it is much more reliable
                # for immediate browser display.
                url = first.get("thumbnail") or first.get("original")
                if not url:
                    last_error = "First Google image has no usable URL"
                    raise RuntimeError(last_error)
                return [{
                    "title": first.get("title") or q,
                    "url": url,
                    "source": first.get("source") or "Google Images",
                    "original": first.get("original") or "",
                }]
            except Exception as exc:
                last_error = str(exc)
                msg = last_error.lower()
                temporary = any(x in msg for x in (
                    "429", "quota", "rate", "limit", "503", "502", "500", "504", "unavailable"
                ))
                if temporary and len(SERPAPI_KEYS) > 1:
                    rotate_serpapi_key()
                    continue
                break
        print("[JARVIS image search ERROR]", last_error, flush=True)
        raise RuntimeError(last_error)

    def do_POST(self):
        global _STATE
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n) if 0 < n < 262144 else b"{}"
        if self.path == "/state":
            # the tracker's heartbeat doubles as the command channel
            _STATE = body
            out = json.dumps(_CMDS[:8]).encode()
            del _CMDS[:8]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        if self.path == "/cmd":
            try:
                cmd = json.loads(body)
                assert cmd.get("a") in _ALLOWED
                if cmd["a"] in ("add_img", "hand", "give", "present") and cmd.get("src"):
                    # THE AIRLOCK: only files really inside ./media/ ever
                    # stage — subfolders allowed, escapes 400. If the
                    # exact path misses, a UNIQUE basename match anywhere
                    # inside the airlock self-heals a wrong-folder guess;
                    # zero or many matches still 400.
                    rel = str(cmd.get("src", "")).lstrip("/")
                    if rel.startswith("media/"):
                        rel = rel[6:]
                    media = media_root()
                    target = (media / rel).resolve()
                    if media not in target.parents or not target.is_file():
                        name = Path(rel).name.lower()
                        hits = [p for p in media.rglob("*")
                                if p.is_file()
                                and p.name.lower() == name] if name else []
                        if len(hits) != 1:
                            raise ValueError("not in the media airlock")
                        target = hits[0]
                    cmd["src"] = "/media/" + target.relative_to(media).as_posix()
                _CMDS.append(cmd)
                self.send_response(204)
            except Exception:
                self.send_response(400)
            self.end_headers()
            return
        if self.path.startswith("/image-proxy"):
            try:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                target = (qs.get("url") or [""])[0]
                u = urllib.parse.urlparse(target)
                if u.scheme not in {"https", "http"} or u.hostname not in {"commons.wikimedia.org", "upload.wikimedia.org", "en.wikipedia.org"}:
                    raise ValueError("Image source not allowed")
                req = urllib.request.Request(target, headers={"User-Agent": "JARVIS-Naman-Acharya/1.0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = r.read()
                    ctype = r.headers.get("Content-Type", "image/jpeg")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 502)
            return

        if self.path == "/image-search":
            try:
                data = json.loads(body)
                query = str(data.get("query", "")).strip()[:160]
                if not query:
                    raise ValueError("No image query provided")
                results = self._image_search(query)
                self._json({"ok": True, "query": query, "results": results})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
            return

        if self.path == "/jarvis-state":
            try:
                data = json.loads(body)
                state = str(data.get("state", "idle")).strip().lower()
                if state not in {"idle", "listening", "thinking", "speaking"}:
                    raise ValueError("Invalid JARVIS state")
                set_jarvis_state(state)
                self._json({"ok": True, "state": state})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
            return

        if self.path == "/ask":
            try:
                if not GEMINI_KEYS:
                    raise RuntimeError("No Gemini API keys configured. Add keys to gemini_keys.txt")

                data = json.loads(body)
                question = str(data.get("question", "")).strip()
                if not question:
                    raise ValueError("No question provided")

                set_jarvis_state("thinking")

                # VISUAL REQUESTS ARE ROUTED BEFORE GEMINI. Gemini is never
                # asked whether this interface can display images.
                ql = question.lower().strip()
                is_image_request = (
                    re.search(r"\bshow(?: me| us| a| an)?\b", ql) or
                    re.search(r"\bdisplay\b", ql) or
                    re.search(r"\b(?:picture|photo|image|photograph)\s+(?:of|on)\b", ql) or
                    re.search(r"\b(?:find|get|pull up)\s+(?:a|an)?\s*(?:picture|photo|image|photograph)\b", ql)
                )
                if is_image_request:
                    # Extract ONLY the requested subject. Never search the whole command.
                    cleaned = re.sub(r"\s+", " ", ql).strip(" ?.,!")
                    patterns = (
                        r"^(?:hey\s+)?(?:jarvis\s+)?(?:please\s+)?(?:show|display|find|get|pull\s+up)\s+(?:me\s+|us\s+)?(?:(?:a|an|the)\s+)?(?:(?:picture|photo|image|photograph)\s+(?:of\s+)?)?(.+)$",
                        r"^(?:show|display|find|get)\s+(?:a|an|the)\s+(?:picture|photo|image|photograph)\s+of\s+(.+)$",
                    )
                    display_subject = None
                    for pat in patterns:
                        m = re.match(pat, cleaned, re.I)
                        if m:
                            display_subject = m.group(1).strip(" ?.,!")
                            break
                    if display_subject:
                        display_subject = re.sub(r"^(?:the|a|an)\s+", "", display_subject, flags=re.I).strip()
                        if display_subject and not display_subject.lower().startswith(("how ", "why ", "whether ", "if ")):
                            article = "an" if display_subject[:1].lower() in "aeiou" else "a"
                            acknowledgement = (
                                f"On your order, sir. I will find and show you {article} picture of "
                                f"{display_subject}."
                            )
                            self._json({"ok": True, "answer": acknowledgement, "image_query": display_subject})
                            return

                # JARVIS brain fallback: if the primary Gemini model is temporarily
                # unavailable/busy (503/5xx), automatically retry and then switch
                # to a secondary Flash model. This keeps the demo running even
                # when one model is under heavy demand.
                prompt = JARVIS_PROMPT + "\n\nUser: " + question
                # Fast path for ordinary conversation: Flash-Lite with minimal
                # thinking is designed for low-latency chat/routing.
                response = None
                last_error = None
                for attempt in range(2):
                    try:
                        response = gemini_generate(
                            model="gemini-3.5-flash-lite",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_level="minimal")
                            )
                        )
                        if response is not None and (response.text or "").strip():
                            break
                        raise RuntimeError("Gemini returned an empty response")
                    except Exception as exc:
                        last_error = exc
                        msg = str(exc).lower()
                        temporary = any(x in msg for x in (
                            "503", "unavailable", "429", "resource_exhausted",
                            "500", "502", "504", "quota", "rate limit"
                        ))
                        if not temporary or attempt == 1:
                            break
                        rotate_gemini_key()

                if response is None:
                    raise RuntimeError(f"All Gemini JARVIS models are temporarily unavailable: {last_error}")

                answer = (response.text or "").strip()
                if not answer:
                    raise RuntimeError("Gemini returned an empty response")

                # VISUAL REQUESTS ARE HANDLED BY THE SERVER, NOT BY GEMINI.
                # Gemini must never decide whether the interface can display an image.
                ql = question.lower().strip()
                image_query = None
                image_patterns = (
                    r"\bshow(?: me| us| a| an)?\b",
                    r"\bdisplay\b",
                    r"\b(?:picture|photo|image|photograph)\s+(?:of|on)\b",
                    r"\b(?:find|get|pull up)\s+(?:a|an)?\s*(?:picture|photo|image|photograph)\b",
                )
                is_image_request = any(re.search(pat, ql) for pat in image_patterns)
                if is_image_request:
                    cleaned = ql
                    for phrase in (
                        "show me a picture of", "show me an image of", "show me a photo of",
                        "show me a picture", "show me an image", "show me a photo",
                        "show a picture of", "show an image of", "show a photo of",
                        "show us a picture of", "show us an image of", "show us a photo of",
                        "display a picture of", "display an image of", "display a photo of",
                        "picture of", "photo of", "image of", "photograph of",
                        "find a picture of", "find an image of", "find a photo of",
                        "get a picture of", "get an image of", "get a photo of",
                        "pull up a picture of", "pull up an image of", "pull up a photo of",
                        "can you", "could you", "please", "jarvis"
                    ):
                        cleaned = cleaned.replace(phrase, " ")
                    cleaned = " ".join(cleaned.split()).strip(" ?.!,")
                    if cleaned:
                        image_query = cleaned
                        # The web UI performs the actual display. Give JARVIS a
                        # guaranteed truthful confirmation instead of Gemini's
                        # generic refusal or invented exhibit/location text.
                        article = "an" if cleaned[:1].lower() in "aeiou" else "a"
                        answer = f"On your order, sir. I will find and show you {article} picture of {cleaned}."

                self._json({"ok": True, "answer": answer, "image_query": image_query})
                return

            except Exception as e:
                set_jarvis_state("idle")
                self._json({"ok": False, "error": str(e)}, 500)
                return
            finally:
                # /speak will switch the ring to speaking; if no speech follows,
                # the next /orb read will naturally show idle after a short timeout.
                pass

        if self.path == "/speak":
            try:
                if not GEMINI_KEYS:
                    raise RuntimeError("No Gemini API keys configured. Add keys to gemini_keys.txt")

                data = json.loads(body)
                text = str(data.get("text", "")).strip()
                if not text:
                    raise ValueError("No text provided")

                # Stay in THINKING while TTS is being generated. The browser
                # switches to SPEAKING only after it receives valid audio.
                set_jarvis_state("thinking")
                response = None
                last_tts_error = None
                for _ in range(max(1, len(GEMINI_KEYS))):
                    try:
                        response = gemini_generate(
                            model="gemini-3.1-flash-tts-preview",
                            contents=(
                                "Speak as a sophisticated futuristic personal AI assistant. "
                                "Calm, confident, precise, formal, slightly witty. "
                                "Natural conversational pacing. Do not sound overly excited. "
                                "Address the listener as sir when it sounds natural.\n\n"
                                + text
                            ),
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name="Charon"
                                        )
                                    )
                                )
                            )
                        )
                        break
                    except Exception as exc:
                        last_tts_error = exc
                        msg = str(exc).lower()
                        temporary = any(x in msg for x in ("503", "unavailable", "429", "resource_exhausted", "quota", "rate limit", "500", "502", "504"))
                        if not temporary:
                            raise
                        rotate_gemini_key()
                if response is None:
                    raise RuntimeError(f"All Gemini keys failed for voice: {last_tts_error}")

                part = response.candidates[0].content.parts[0]
                pcm = part.inline_data.data
                if not pcm:
                    raise RuntimeError("Gemini returned no audio data")

                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(pcm)
                audio = wav_buffer.getvalue()

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(audio)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(audio)
                return

            except Exception as e:
                set_jarvis_state("idle")
                self._json({"ok": False, "error": str(e)}, 500)
                return

        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path == "/config":
            # the page builds its ring name + orb bloom from this
            self._json({"name": CONFIG.get("name", "Assistant"),
                        "orbs": [{"title": o.get("title", "?"),
                                  "kind": o.get("kind", "notes")}
                                 for o in CONFIG["orbs"]]})
            return
        if self.path.startswith("/tree"):
            # a notes orb's folder tree. Jailed to that orb's configured
            # folder, .md only, CLAUDE.md (AI config, not a note) excluded.
            q = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            idx = (q.get("orb") or ["0"])[0]
            root = orb_root(idx)
            if root is None or not root.is_dir():
                self._json({"name": "?", "notes": [], "dirs": []}, 404)
                return

            def walk(d):
                out = {"name": d.name, "notes": [], "dirs": []}
                for p in sorted(d.iterdir()):
                    if p.name.startswith("."):
                        continue
                    if p.is_dir():
                        sub = walk(p)
                        if sub["notes"] or sub["dirs"]:
                            out["dirs"].append(sub)
                    elif p.suffix == ".md" and p.name != "CLAUDE.md":
                        # note files travel as "<orb>/<relpath>" so /note
                        # knows which jail to resolve them against
                        out["notes"].append(
                            {"title": p.stem,
                             "file": f"{int(idx)}/{p.relative_to(root).as_posix()}"})
                return out
            try:
                tree = walk(root)
                tree["name"] = CONFIG["orbs"][int(idx)].get("title", tree["name"])
                self._json(tree)
            except Exception:
                self._json({"name": "?", "notes": [], "dirs": []}, 500)
            return
        if self.path == "/props":
            # the media airlock as a browsable tree — live filesystem
            # read: drop a file in media/, reopen the orb, it's there
            EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".webm",
                    ".glb", ".gltf"}
            mroot = media_root()

            def walkm(d):
                out = {"name": d.name, "items": [], "dirs": []}
                for p in sorted(d.iterdir()):
                    if p.name.startswith("."):
                        continue
                    if p.is_dir():
                        sub = walkm(p)
                        # A folder carrying a README was made on purpose, so
                        # it stays listed even while empty. holo/ and models/
                        # ship exactly that way -- nothing in them but a
                        # README saying what to drop in -- and hiding every
                        # folder with no stageable file made them invisible
                        # until you had already found them. This board is HOW
                        # you discover a folder, so the one that teaches you
                        # the hologram cannot be the one you must know about
                        # first. Arbitrary empty folders still stay hidden.
                        documented = (p / "README.md").is_file()
                        if sub["items"] or sub["dirs"] or documented:
                            out["dirs"].append(sub)
                    elif p.suffix.lower() in EXTS:
                        # as_posix, because THE FOLDER IS THE RENDER LAW and
                        # the law is read client-side with forward slashes.
                        # str() of a path yields BACKSLASHES on Windows, so
                        # "fx\fireball.png" never matched /\/fx\// in
                        # stage.html: props in fx/ silently kept their card
                        # frame and models in holo/ silently rendered solid
                        # instead of as the blue wire. These strings become
                        # URL fragments in the browser, where a backslash is
                        # not a separator at all, so POSIX is the only
                        # correct wire format here regardless of platform.
                        out["items"].append(p.relative_to(mroot).as_posix())
                return out
            try:
                tree = walkm(mroot)
                tree["name"] = "Props"
                self._json(tree)
            except Exception:
                self._json({"name": "Props", "items": [], "dirs": []}, 500)
            return
        if self.path == "/orb":
            # the ring's heartbeat: your assistant's live state, read from
            # tiny files in ./state/. Every read fails soft — no files,
            # no assistant, no problem: the ring just breathes.
            s_dir = HERE / "state"
            out = {"state": "idle", "mood": "green", "wave": None}
            try:
                f = s_dir / "state"
                s = f.read_text().strip().lower()
                if s in ("idle", "listening", "thinking", "speaking"):
                    # A STALE non-idle state DECAYS to idle, because the
                    # only thing that ever writes "idle" is the writer
                    # finishing. A writer that is killed, crashes, or is
                    # force-quit mid-turn never writes it -- so the ring
                    # sat on "thinking" forever, with no timeout, nothing
                    # to reset it, and no way for anyone to guess why.
                    #
                    # This is a safety net for a DEAD writer, not a
                    # liveness signal: a genuinely long turn will decay
                    # too, and showing idle during real work is a far
                    # smaller lie than claiming to think for eternity.
                    # Raise state_timeout_s if your turns run longer.
                    age = time.time() - f.stat().st_mtime
                    if s == "idle" or age < STATE_TIMEOUT:
                        out["state"] = s
            except Exception:
                pass
            try:
                m = json.loads((s_dir / "mood.json").read_text())
                if time.time() - float(m.get("ts", 0)) < 45.0:
                    out["mood"] = m.get("mood", "green")
            except Exception:
                pass
            if out["state"] == "speaking":
                try:
                    w = json.loads((s_dir / "wave.json").read_text())
                    if time.time() - float(w.get("ts", 0)) < 0.6:
                        out["wave"] = w.get("samples", [])[:64]
                except Exception:
                    pass
            self._json(out)
            return
        if self.path == "/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_STATE)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(_STATE)
            return
        if not self.path.startswith("/note?"):
            return super().do_GET()
        # one note's text: f=<orb>/<relpath>, resolved against that orb's
        # jail. Inside the root, .md only, must exist — anything else 404s.
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        rel = (q.get("f") or [""])[0]
        idx, _, rel = rel.partition("/")
        root = orb_root(idx)
        if root is None:
            self.send_response(404)
            self.end_headers()
            return
        target = (root / rel).resolve()
        if (root not in target.parents) or target.suffix != ".md" \
                or not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        body = target.read_text(encoding="utf-8", errors="replace").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    (HERE / "state").mkdir(exist_ok=True)   # the ring's runtime files land here
    port = int(CONFIG.get("port", 8794))
    print(f"barehands up: http://127.0.0.1:{port}/stage.html", flush=True)
    print("  tracker (camera): open that URL in Chrome", flush=True)
    print("  render (overlay): same URL + ?role=render", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
