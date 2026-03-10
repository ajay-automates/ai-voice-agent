"""
Agentic RAG Agent — Async, optimized pipeline orchestrator.

Optimizations applied:
  Phase 1:  Batch grading — 5 docs graded in ONE LLM call (see grader.py)
  Phase 2:  AsyncOpenAI throughout — non-blocking API calls
  Phase 3:  skip_hallucination flag — caller runs hallucination check in parallel with TTS
  Phase 5:  Early exit — skip reformulation the moment we have enough relevant docs
  Phase 7:  Context reuse — last 6 conversation turns passed to generation
  Phase 8:  max_tokens 300 → 120 — shorter answers = faster TTS audio

Pipeline per query:
  1. Guardrails check
  2. Hybrid retrieve (vector + BM25, top 5)
  3. Grade ALL documents in ONE LLM call (batch)
  4. Decide: enough relevant docs? → Generate OR Reformulate & retry (max 2 attempts)
  5. Generate answer from graded-relevant docs (120 tokens max)
  6. Hallucination check (skipped when skip_hallucination=True — caller does it in parallel)
  7. Return full result with pipeline trace
"""

import asyncio
import os
import time
from typing import Dict, List
from openai import AsyncOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from app.vector_store import retrieve
from app.hybrid_retrieval import BM25Searcher, hybrid_search
from app.grader import grade_documents, count_relevant_documents
from app.reformulator import reformulate_query
from app.hallucination_checker import check_hallucination
from app.guardrails import check_guardrails
from app.prompts import SYSTEM_PROMPT

raw_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
client = wrap_openai(raw_client)

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "ai-voice-agent")


class AgenticRAGAgent:
    """Production voice agent with self-correcting agentic RAG — fully async."""

    def __init__(self):
        self.model = "gpt-4o-mini"
        self.bm25_searcher = BM25Searcher()
        self.conversation_history: List[Dict] = []

    def rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from current vector store. Call after every document upload."""
        from app.vector_store import get_all_chunks as _get
        all_chunks = _get()
        if all_chunks:
            self.bm25_searcher.build_index(all_chunks)

    def clear(self) -> None:
        """Reset conversation history and BM25 index."""
        self.conversation_history = []
        self.bm25_searcher = BM25Searcher()

    @traceable(name="agentic_rag_query", run_type="chain")
    async def query(
        self,
        question: str,
        max_attempts: int = 2,
        skip_hallucination: bool = False,
    ) -> Dict:
        """
        Run the full agentic RAG pipeline for a user question.

        Args:
            question: User's question text
            max_attempts: Max retrieval+grade attempts before giving up
            skip_hallucination: When True, skip the hallucination check and include
                relevant_docs in the result so the caller can run it in parallel with TTS.

        Returns dict with:
          answer, confidence, grounded, sources, relevant_docs, pipeline_trace,
          retrieval_attempts, latency_seconds, blocked
        """
        start = time.time()
        pipeline_trace: List[Dict] = []

        # ── Step 1: Guardrails ─────────────────────────────────────────────
        guard = check_guardrails(question)
        if not guard["allowed"]:
            return {
                "answer": f"I can't process that request. {guard['reason']}.",
                "confidence": 1.0,
                "grounded": True,
                "sources": [],
                "relevant_docs": [],
                "pipeline_trace": [{"step": "Guardrails", "action": f"Blocked: {guard['reason']}"}],
                "retrieval_attempts": 0,
                "latency_seconds": round(time.time() - start, 2),
                "blocked": True,
                "block_reason": guard["reason"],
            }

        pipeline_trace.append({"step": "Guardrails", "action": "Passed"})

        # ── Agentic Retrieve → Grade → Decide loop ─────────────────────────
        current_query = question
        attempt = 0
        relevant_docs: List[Dict] = []

        while attempt < max_attempts:
            attempt += 1

            # Step 2: Hybrid Retrieve
            vector_results = await retrieve(current_query, k=5)
            retrieved = hybrid_search(current_query, vector_results, self.bm25_searcher, n_results=5)

            pipeline_trace.append({
                "step": f"Attempt {attempt} — Retrieve",
                "query": current_query,
                "docs_retrieved": len(retrieved),
                "method": "hybrid" if self.bm25_searcher.bm25 else "vector",
            })

            if not retrieved:
                pipeline_trace.append({
                    "step": f"Attempt {attempt} — Decision",
                    "action": "No documents in knowledge base",
                })
                break

            # Step 3: Grade ALL docs in ONE LLM call (Phase 1 batch grading)
            graded = await grade_documents(question, retrieved)
            rel_count, partial_count, not_rel_count = count_relevant_documents(graded)

            pipeline_trace.append({
                "step": f"Attempt {attempt} — Grade",
                "relevant": rel_count,
                "partially_relevant": partial_count,
                "not_relevant": not_rel_count,
                "grades": [
                    {
                        "source": d.get("source", "?")[:40],
                        "grade": d.get("grade", "?"),
                        "reason": d.get("grade_reason", "")[:80],
                    }
                    for d in graded
                ],
            })

            # Step 4: Decide
            usable = [d for d in graded if d.get("grade") in ("relevant", "partially_relevant")]
            fully_relevant = [d for d in graded if d.get("grade") == "relevant"]

            # Phase 5: Early exit — as soon as we have enough, stop immediately
            sufficient = len(fully_relevant) >= 2 or (len(usable) >= 1 and attempt >= max_attempts)

            if sufficient:
                relevant_docs = usable if usable else graded
                pipeline_trace.append({
                    "step": f"Attempt {attempt} — Decision",
                    "action": f"Sufficient docs ({len(fully_relevant)} relevant, {partial_count} partial). Proceeding.",
                    "docs_for_generation": len(relevant_docs),
                })
                break
            elif attempt < max_attempts:
                new_query = await reformulate_query(question)
                pipeline_trace.append({
                    "step": f"Attempt {attempt} — Reformulate",
                    "original_query": current_query,
                    "reformulated_query": new_query,
                    "reason": f"Only {len(fully_relevant)} relevant docs found",
                })
                current_query = new_query
            else:
                relevant_docs = usable if usable else graded[:3]
                pipeline_trace.append({
                    "step": f"Attempt {attempt} — Decision",
                    "action": f"Max attempts reached. Using best available ({len(relevant_docs)} docs).",
                    "docs_for_generation": len(relevant_docs),
                })

        # ── Step 5: Generate ───────────────────────────────────────────────
        if not relevant_docs:
            answer = "I don't have relevant information in my knowledge base to answer that question."
            hallucination_result = {"grounded": True, "confidence": 1.0, "issues": []}
            sources = []
            pipeline_trace.append({"step": "Generate", "action": "No relevant docs — returning fallback"})
        else:
            answer = await self._generate_answer(question, relevant_docs)
            pipeline_trace.append({
                "step": "Generate",
                "docs_used": len(relevant_docs),
                "answer_length": len(answer),
            })

            if skip_hallucination:
                # Phase 3: caller will run hallucination check in parallel with TTS
                hallucination_result = {"grounded": True, "confidence": 0.8, "issues": []}
            else:
                # Step 6: Hallucination check (sequential path)
                hallucination_result = await check_hallucination(answer, relevant_docs)
                pipeline_trace.append({
                    "step": "Hallucination Check",
                    "grounded": hallucination_result["grounded"],
                    "confidence": hallucination_result["confidence"],
                    "issues": hallucination_result.get("issues", []),
                })

            sources = [
                {
                    "source": d.get("source", "Unknown"),
                    "grade": d.get("grade", "N/A"),
                    "relevance_score": round(d.get("relevance_score", 0.0), 3),
                }
                for d in relevant_docs
            ]

        # Update conversation history (keep last 6 turns = 12 messages)
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

        return {
            "answer": answer,
            "confidence": hallucination_result.get("confidence", 0.5),
            "grounded": hallucination_result.get("grounded", True),
            "sources": sources,
            "relevant_docs": relevant_docs,   # exposed for parallel hallucination check
            "pipeline_trace": pipeline_trace,
            "retrieval_attempts": attempt,
            "latency_seconds": round(time.time() - start, 2),
            "blocked": False,
        }

    async def _generate_answer(self, question: str, relevant_docs: List[Dict]) -> str:
        """
        Generate answer using relevant docs (already retrieved and graded) and
        the last 6 conversation turns for context (Phase 7 context reuse).
        max_tokens=120 keeps answers short for faster TTS (Phase 8).
        """
        # Build context directly from the graded docs — no extra retrieval call needed
        context_parts = [
            f"[Source: {d.get('source', 'Unknown')}]\n{d.get('text', '')}"
            for d in relevant_docs
        ]
        context = "\n\n---\n\n".join(context_parts)

        full_system = f"""{SYSTEM_PROMPT}

Retrieved Context:
---
{context}
---"""
        # Phase 7: include recent conversation history for context continuity
        messages = [{"role": "system", "content": full_system}]
        for h in self.conversation_history[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": question})

        try:
            completion = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=120,   # Phase 8: reduced from 300 — shorter = faster TTS
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"


# Global singleton — imported by main.py
agent = AgenticRAGAgent()
