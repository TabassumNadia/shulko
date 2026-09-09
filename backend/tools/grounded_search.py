"""
Google Search grounding via Gemini's native tool.

This is grounding in the strict sense the brief asks for: the model is
given a search tool, it decides what to look up, and the API returns
`groundingMetadata` naming the pages that supported each span of the
answer. That metadata is the evidence -- it is what makes the claim
checkable rather than merely cited.

Tavily is kept as a fallback for when the Gemini SDK or key is missing,
so the pipeline degrades instead of failing. The fallback is a plain
search: it retrieves text, and the model is instructed to answer only
from that text. Weaker, but honest about being weaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langsmith import traceable

from backend.core.config import get_settings


@dataclass
class GroundedAnswer:
    text: str
    sources: list[dict] = field(default_factory=list)   # {title, url}
    queries: list[str] = field(default_factory=list)    # what it searched for
    mechanism: str = "none"                             # google_search | tavily | none

    @property
    def is_grounded(self) -> bool:
        return bool(self.sources)


@traceable(name="tool_google_search_grounding", run_type="tool")
def grounded_answer(system_prompt: str, user_prompt: str,
                    query: str | None = None) -> GroundedAnswer:
    """Answer a question with live Google Search grounding.

    Returns the answer plus the sources the model actually retrieved, so
    a caller can refuse to use ungrounded claims.

    Args:
        system_prompt: the discipline the answer must obey.
        user_prompt: the full question, for the model.
        query: a short search string for the fallback engine. Gemini
            picks its own queries, but Tavily searches literally, and a
            multi-line prompt makes a poor search query. Defaults to the
            user prompt when not given.
    """
    settings = get_settings()

    if settings.google_api_key:
        try:
            return _via_gemini(system_prompt, user_prompt, settings.google_api_key)
        except Exception as exc:                     # SDK/version/quota issues
            print(f"[grounded_search] Gemini grounding unavailable: {exc}")

    if settings.tavily_api_key:
        try:
            return _via_tavily(system_prompt, user_prompt,
                               query or user_prompt, settings.tavily_api_key)
        except Exception as exc:
            print(f"[grounded_search] Tavily fallback failed: {exc}")

    return GroundedAnswer(text="", mechanism="none")


def _via_gemini(system_prompt: str, user_prompt: str, api_key: str) -> GroundedAnswer:
    from google import genai
    from google.genai import types

    from backend.core.llm import acquire_quota

    # This path talks to google-genai directly, so it is invisible to the
    # limiter the LangChain models share. Draw from the same bucket
    # explicitly, or a grounded search silently spends the quota that the
    # classifier is about to queue for.
    acquire_quota(get_settings().grounding_model)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=get_settings().grounding_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
        ),
    )

    sources, queries = [], []
    for candidate in getattr(response, "candidates", []) or []:
        meta = getattr(candidate, "grounding_metadata", None)
        if not meta:
            continue
        queries.extend(getattr(meta, "web_search_queries", None) or [])
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None):
                sources.append({
                    "title": getattr(web, "title", "") or web.uri,
                    "url": web.uri,
                })

    # Deduplicate while preserving the model's ordering.
    seen, unique = set(), []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique.append(s)

    return GroundedAnswer(
        text=(response.text or "").strip(),
        sources=unique,
        queries=list(dict.fromkeys(queries)),
        mechanism="google_search",
    )


def _via_tavily(system_prompt: str, user_prompt: str, query: str,
                api_key: str) -> GroundedAnswer:
    """Retrieve with Tavily, then answer ONLY from what was retrieved.

    Tavily will happily return its own synthesised `answer`, but that
    answer obeys Tavily's prompt, not ours -- none of the refusal
    discipline in REGULATORY_SYSTEM applies to it. Since the whole point
    of this agent is that an unsourced claim must not ship, the retrieved
    page text is fed to our own model under our own system prompt
    instead.

    Weaker than Gemini's native grounding, which reports which span of
    the answer came from which page. Honest about being weaker.
    """
    from tavily import TavilyClient

    results = TavilyClient(api_key=api_key).search(
        query=query, max_results=4, include_answer=False,
        # Official sources first. Restricting the domain list entirely
        # would return nothing on codes NBR has not published about, so
        # this ranks rather than filters.
        include_domains=["nbr.gov.bd", "bangladeshcustoms.gov.bd",
                         "mincom.gov.bd", "bsti.gov.bd"],
    )
    hits = [r for r in results.get("results", []) if r.get("url")]
    if not hits:
        return GroundedAnswer(text="", sources=[], queries=[query],
                              mechanism="tavily")

    retrieved = "\n\n".join(
        f"[{r.get('title', '') or r['url']}] ({r['url']})\n"
        f"{(r.get('content') or '')[:1500]}"
        for r in hits
    )

    from backend.core.llm import get_llm, with_retry

    reply = with_retry(get_llm("judge")).invoke([
        ("system", system_prompt + "\n\nAnswer ONLY from the retrieved "
                   "pages below. If they do not answer the question, say "
                   "so and state nothing further."),
        ("user", f"{user_prompt}\n\n--- retrieved pages ---\n{retrieved}"),
    ])
    text = getattr(reply, "content", "") or ""

    return GroundedAnswer(
        text=str(text).strip(),
        sources=[{"title": r.get("title", "") or r["url"], "url": r["url"]}
                 for r in hits],
        queries=[query],
        mechanism="tavily",
    )
