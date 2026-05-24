from pathlib import Path
from bs4 import element
from bs4 import BeautifulSoup
from dataclasses import dataclass
from ebooklib import epub
import urllib
from lingua import LanguageDetectorBuilder

import pysbd
from tqdm import tqdm


def fix_end_of_quotes(lines):
    fixed = []
    for i, line in enumerate(lines):
        if i > 0 and line and line[0] in ["」", "’"]:
            fixed[-1] += line[0]
            line = line[1:]
        if line:
            fixed.append(line)
    return fixed


def merge_short_lines_with_quotes(lines):
    fixed = []
    for line in lines:
        if fixed and fixed[-1][0] in ["」", "’"] and len(fixed[-1] + line) <= 1:
            fixed[-1] += line
        else:
            fixed.append(line)
    return fixed


def split_sentences(input_file, output_path, lang, nlp):
    # for file_name in input_file:
    with open(input_file, "r", encoding="UTF-8") as file:
        input_lines = file.readlines()

    split_sentences_from_input(input_lines, output_path, lang, nlp)


def get_segments(input_lines, lang, nlp):
    if nlp is None:
        seg = pysbd.Segmenter(language=lang, clean=False)
    lines = []
    print("✂️  Splitting transcript into sentences")
    if nlp is None:
        for text in tqdm(input_lines):
            text = text.rstrip("\n")
            s = seg.segment(text)
            lines += s
    else:
        lines = [
            sentence.text
            for sentence in nlp("\n\n".join(input_lines)).sentences
            if sentence.text.rstrip("\n")
        ]
    return lines


def split_sentences_from_input(input_lines, output_path, lang, nlp):
    lines = get_segments(input_lines, lang, nlp)
    lines = fix_end_of_quotes(lines)
    lines = merge_short_lines_with_quotes(lines)

    with open(output_path, "w", encoding="utf-8") as fo:
        for line in lines:
            if line:
                fo.write(line + "\n")


def flatten(t):
    return (
        [j for i in t for j in flatten(i)]
        if isinstance(t, (tuple, list))
        else [t] if isinstance(t, epub.Link) else []
    )


@dataclass(eq=True, frozen=True)
class EpubParagraph:
    chapter: int
    element: element.Tag
    references: list

    def text(self):
        return "".join(self.element.strings)


@dataclass(eq=True, frozen=True)
class EpubChapter:
    content: BeautifulSoup
    title: str
    is_linear: bool
    idx: int

    def text(self):
        paragraphs = self.content.find("body").find_all(
            ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]
        )
        r = []
        for p in paragraphs:
            if "id" in p.attrs:
                continue
            r.append(EpubParagraph(chapter=self.idx, element=p, references=[]))
        return r


FRONT_MATTER_KEYWORDS = (
    "目次",        # table of contents
    "もくじ",       # table of contents (kana)
    "Contents",
    "Table of Contents",
    "凡例",        # legend / explanatory notes
    "奥付",        # colophon (back matter — also worth dropping)
    "著作権",       # copyright
    "Copyright",
    "電子書籍",     # ebook disclaimer ("this e-book is laid out vertically...")
    "サムネイル",   # thumbnail (ebook image disclaimer)
    "再ダウンロード",  # re-download disclaimer
    "リーディングシステム",  # reading-system disclaimer
    "カバー",       # cover
    "表紙",        # title page
    "本書は",      # "this book is..." typical disclaimer prefix
    "登録商標",     # registered trademarks
)

CHAPTER_START_MARKERS = (
    "プロローグ",    # prologue
    "Prologue",
    "序章",        # introduction chapter
    "序文",        # preface
    "はじめに",     # foreword
    "第一章",      # chapter 1
    "第1章",
    "Chapter 1",
    "Chapter I",
)


def _chapter_text_for_filter(chapter) -> str:
    """Return concatenated paragraph text of a chapter, for keyword scanning."""
    try:
        return "".join(p.text() for p in chapter.text())
    except Exception:
        return ""


def _looks_like_front_matter(chapter) -> bool:
    """Heuristic: small chapter dominated by ebook-meta keywords."""
    title = (chapter.title or "").strip()
    body = _chapter_text_for_filter(chapter)
    body_len = len(body)

    if any(k in title for k in FRONT_MATTER_KEYWORDS):
        return True
    if body_len < 600 and any(k in body for k in FRONT_MATTER_KEYWORDS):
        return True
    # Long disclaimer pages occasionally exceed 600 chars; require explicit keyword hit then.
    if body_len < 1500 and sum(k in body for k in FRONT_MATTER_KEYWORDS) >= 2:
        return True
    return False


BACK_MATTER_KEYWORDS = (
    "奥付",        # colophon
    "装幀",        # book design credit
    "装丁",        # book design credit (alt kanji)
    "本電子書籍",   # ebook usage disclaimers
    "無断で複製",   # "no unauthorized reproduction"
    "無断複製",
    "転載を禁",
    "改変、改ざん",
    "サポート",     # "support"
    "問い合わせ",   # "inquiries"
    "Japanese text only",
    "発行所",      # publisher
    "発行者",      # publisher contact
    "印刷",        # printing info
    "ISBN",
    "著作権",      # copyright
    "Copyright",
    "©",
    "(c)",
    # Author-bio markers — typical structure is "{name}（ふりがな）{year}年
    # {city}生まれ、{university}{department}卒業。{year}年『{title}』で{prize}を受賞..."
    # Multiple of these in one chapter is a strong bio signal.
    "受賞",        # "received award"
    "文学賞",      # "literary prize"
    "文学部",      # "department of literature"
    "著者紹介",    # "author introduction"
    "略歴",        # "brief biography"
)


def filter_back_matter(chapters: list) -> list:
    """Symmetric to filter_front_matter, but walks from the END.

    Audiobook narrators usually don't read the colophon, copyright disclaimer,
    or "no unauthorized redistribution" boilerplate at the back of an epub.
    Walk backward from the last chapter, dropping ones dominated by
    BACK_MATTER_KEYWORDS, until we hit a chapter with substantial prose (or
    a likely afterword/epilogue — those *are* often narrated).

    Stops dropping at:
      - A chapter with a long prose paragraph (>200 chars) that isn't
        keyword-dominated.
      - A chapter whose title contains エピローグ / Epilogue / あとがき /
        Afterword — those are usually narrated.
    """
    if not chapters:
        return chapters

    NARRATED_TAIL_MARKERS = (
        "エピローグ",     # epilogue
        "Epilogue",
        "あとがき",       # afterword
        "Afterword",
        "解説",          # commentary/critical essay (sometimes narrated)
    )

    end_idx = len(chapters)
    for i in range(len(chapters) - 1, -1, -1):
        chapter = chapters[i]
        title = (chapter.title or "").strip()
        body = _chapter_text_for_filter(chapter)

        # Stop dropping at narrated-tail markers — these are story content.
        if any(m in title for m in NARRATED_TAIL_MARKERS):
            break
        if any(m in body[:200] for m in NARRATED_TAIL_MARKERS):
            break

        # Heuristic: dominated by back-matter keywords?
        keyword_hits = sum(k in body for k in BACK_MATTER_KEYWORDS)
        title_hit = any(k in title for k in BACK_MATTER_KEYWORDS)
        body_len = len(body)

        if title_hit or keyword_hits >= 2:
            end_idx = i
            continue
        if body_len < 600 and keyword_hits >= 1:
            end_idx = i
            continue
        # Otherwise this is real prose — stop here.
        break

    if end_idx == len(chapters):
        return chapters

    dropped = chapters[end_idx:]
    print(
        f"✂️  Dropped {len(dropped)} back-matter chapter(s) after "
        f"'{(chapters[end_idx - 1].title or '?').strip()[:40]}': "
        f"{[(c.title or '?')[:30] for c in dropped]}"
    )
    return chapters[:end_idx]


def filter_front_matter(chapters: list) -> list:
    """Drop epub spine items the audiobook narrator wouldn't read.

    Strategy: walk from the start of the spine, skipping chapters that look
    like front matter (title pages, copyright/legal disclaimers, TOC), until
    we reach either an explicit chapter-start marker (`プロローグ`, `第一章`,
    `Prologue`, ...) or a substantial prose paragraph (~>200 chars).

    Falls back to the original list if it can't find a confident start — we'd
    rather over-include than silently drop real prose.
    """
    if not chapters:
        return chapters

    start_idx = None
    for i, chapter in enumerate(chapters):
        title = (chapter.title or "").strip()
        body = _chapter_text_for_filter(chapter)

        # Front-matter detection FIRST. A TOC chapter literally lists chapter
        # names like プロローグ / 第一章, so checking CHAPTER_START_MARKERS in
        # its body would incorrectly accept it. The title-based front-matter
        # check (e.g. title == "目次") catches that case.
        if _looks_like_front_matter(chapter):
            continue

        if any(m in title for m in CHAPTER_START_MARKERS):
            start_idx = i
            break
        if any(m in body[:200] for m in CHAPTER_START_MARKERS):
            start_idx = i
            break
        # Substantial prose paragraph = likely the real story has started.
        try:
            paragraphs = chapter.text()
        except Exception:
            paragraphs = []
        long_para = any(len(p.text()) > 200 for p in paragraphs)
        if long_para:
            start_idx = i
            break

    if start_idx is None or start_idx == 0:
        return chapters

    dropped = chapters[:start_idx]
    print(
        f"✂️  Dropped {len(dropped)} front-matter chapter(s) before "
        f"'{(chapters[start_idx].title or '?').strip()[:40]}': "
        f"{[(c.title or '?')[:30] for c in dropped]}"
    )
    return chapters[start_idx:]


@dataclass(eq=True, frozen=True)
class Epub:
    epub: epub.EpubBook
    path: Path
    title: str
    chapters: list

    def text(self):
        return [p for c in self.chapters for p in c.text()]

    @classmethod
    def from_file(cls, path):
        file = epub.read_epub(path, {"ignore_ncx": True})

        flat_toc = flatten(file.toc)
        m = {
            it.id: i
            for i, e in enumerate(flat_toc)
            if (
                it := file.get_item_with_href(
                    urllib.parse.unquote(e.href.split("#")[0])
                )
            )
        }
        if len(m) != len(flat_toc):
            print(
                "WARNING: Couldn't fully map toc to chapters, contact the dev, preferably with the epub"
            )

        chapters = []
        prev_title = ""
        for i, v in enumerate(file.spine):
            item = file.get_item_with_id(v[0])
            title = flat_toc[m[v[0]]].title if v[0] in m else ""

            if item.media_type != "application/xhtml+xml":
                if title:
                    prev_title = title
                continue

            content = BeautifulSoup(item.get_content(), "html.parser")

            r = content.find("body").find_all(
                ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]
            )
            # Most of the time chapter names are on images
            idx = 0
            while idx < len(r) and not r[idx].get_text().strip():
                idx += 1
            if idx >= len(r):
                if title:
                    prev_title = title
                continue

            if not title:
                if t := prev_title.strip():
                    title = t
                    prev_title = ""
                elif len(t := r[idx].get_text().strip()) < 25:
                    title = t
                else:
                    title = item.get_name()

            chapter = EpubChapter(content=content, title=title, is_linear=v[1], idx=i)
            chapters.append(chapter)
        chapters = filter_front_matter(chapters)
        chapters = filter_back_matter(chapters)
        return cls(
            epub=file,
            path=path,
            title=file.title.strip() or path.name,
            chapters=chapters,
        )


def detect_language(file_path: Path) -> str | None:
    """
    Detects the language of a given text file.

    Args:
        file_path: The path to the text or subtitle file.

    Returns:
        A two-letter ISO 639-1 language code (e.g., 'en', 'ja') if detection is successful,
        otherwise None.
    """
    DETECTOR = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )

    if not file_path.exists():
        print(f"❗ Cannot detect language: File does not exist at '{file_path}'")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            print(f"⚠️ Verification skipped for '{file_path.name}': No content found.")
            return None

        detected_language = DETECTOR.detect_language_of(content)

        if detected_language:
            detected_code = detected_language.iso_code_639_1.name.lower()
            print(f"Detected language of '{file_path.name}' as '{detected_code}'.")
            return detected_code
        else:
            print(f"⚠️ Could not reliably detect language for '{file_path.name}'.")
            return None

    except Exception as e:
        print(
            f"❗An error occurred during language detection for '{file_path.name}': {e}"
        )
        return None
