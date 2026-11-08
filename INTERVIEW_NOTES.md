# Interview notes: EU Compliance Agent

Fifteen questions an interviewer is likely to ask, with answers based only on what this project actually built and measured. Every number comes from `results/`.

---

**1. In one minute, what did you build?**

A question-answering assistant over four official documents: the EU AI Act, GDPR and two Central Bank of Ireland guidance papers. It answers only from those documents, cites the document and page, and says "I don't know" when they do not cover the question. I built it twice, a basic RAG and an agentic RAG with LangGraph, and tested both on the same 30 questions with RAGAS. Everything runs locally with free models through Ollama.

**2. Why build a baseline at all?**

Without a baseline, I cannot show that the agent's extra steps are worth their cost. Both versions use the same model, the same passages, the same answer prompt and the same top 5, so the only difference is how passages are found and checked. That makes the comparison fair.

**3. What does the agent do that the baseline does not?**

Four things.
1. It rewrites the question into the documents' formal wording and picks which documents to search.
2. It runs hybrid search, vector plus BM25 keyword search, merged with Reciprocal Rank Fusion.
3. It reranks the candidates with a cross-encoder (`bge-reranker-base`) and keeps the best 5.
4. It grades each passage. If none is relevant it retries once with new wording across all documents, and if that also fails it says "I don't know".

The retry limit of one means it can never loop forever.

**4. Did the agent actually beat the baseline?**

Partly, and I report it honestly. On 25 answerable questions, the agent found better passages: context precision 0.8547 against 0.7071, and context recall 0.7833 against 0.7533. Answer relevancy was 0.8388 against 0.8134. But factual correctness was about the same, 0.4756 against 0.4836. So better retrieval did not turn into more correct answers with a 7B model. The agent is also slower: 11.39 s per answer on average against 4.42 s.

**5. Why do you think better passages did not give better answers?**

The final answer is written by the same small model, `qwen2.5:7b`, so the writing step limits the result. In the saved answers I can see cases where the right passage was given but the answer was incomplete, or cited passage numbers that do not exist. A stronger answer model is the obvious next test, and the code allows that by changing two lines in `.env`.

**6. Why hybrid search instead of only vectors?**

Legal questions often depend on exact terms such as "Article 33" or "impact tolerance". Vector search is good at meaning but can miss exact words; BM25 is good at exact words but misses paraphrases. I merge the two ranked lists with Reciprocal Rank Fusion (k = 60). RRF uses only rank positions, so I do not have to make the two very different score scales comparable.

**7. Why add a reranker after hybrid search?**

The first search is fast but rough. A cross-encoder reads the question and each passage together, so it judges relevance more precisely. It is too slow to run over all 1,497 passages, so it only reranks the merged candidates (up to 10 from each search) and keeps 5. It runs on the Mac's GPU (MPS).

**8. How did you split the documents, and why those settings?**

PyMuPDF extracts the text page by page. I split it into passages of 1,000 characters with 150 characters of overlap, so each citation can point to a real page. The overlap stops a rule being cut in half at a passage boundary. For the EU laws I also tag each passage with every Article or Annex it touches, so a citation reads "GDPR, Article 33, page 52". That gave 1,497 passages in total. A known weakness is that a rule running across two pages is still split.

**9. Why `nomic-embed-text` with "search_document" and "search_query" prefixes?**

It is a free, local embedding model that runs through Ollama. It was trained with those prefixes, so passages get `search_document:` and questions get `search_query:`. I wrapped this in a small class so it happens everywhere automatically. For the RAGAS judge I used plain embeddings with no prefixes, because answer relevancy compares a question with a question.

**10. How did you evaluate it?**

- **Test set:** 30 questions with checked reference answers: 25 answerable, including 2 that need two documents, and 5 that the documents cannot answer.
- **RAGAS measures:** faithfulness, answer relevancy, context precision, context recall and factual correctness, with the local model as judge, on the 25 answerable questions.
- **Refusal scores:** how many of the 5 unanswerable questions each system refused, and how many of the 25 answerable ones it wrongly refused.

The script saves every answer as it goes, so a stopped run carries on where it left off. It keeps the Mac awake with `caffeinate`. The full run took 5096 seconds.

**11. Why did you add factual correctness?**

In a trial run, faithfulness gave a perfect score to an answer that was wrong. Faithfulness only checks that the answer matches the passages, not that it matches the truth. Factual correctness compares the answer's claims with the checked reference answer, so it measures whether the answer is right.

**12. Do you trust the judge?**

Only partly, and I say so in the README. The judge is the same 7B model that wrote the answers, so it can be lenient towards its own style, and single scores can be wrong. For example, in Q17 factual correctness gave the baseline's wrong "Yes" 0.55 and the agent's correct "No" 0.40. That is why I treat small gaps with care and show the real answers on the dashboard. The next step is a stronger judge model and a blind human check of a sample, to measure how often the judge agrees with a person.

**13. How does it handle questions it cannot answer?**

In three layers:
1. The answer prompt tells the model to reply with one exact refusal sentence if the passages do not contain the answer.
2. The agent's grading step refuses before answering if no passage is relevant after a retry.
3. Citation numbers that do not match a real passage are dropped.

Both systems refused all 5 unanswerable questions, including a trap about which Irish authority enforces the AI Act. The cost is wrong refusals: the agent refused 2 answerable questions and the baseline 1.

**14. What went wrong, and what would you change?**

From the agent's saved steps:
- **Q22:** the agent found the right pages, but the grader rejected all 5. The retry then used exactly the same query.
- **Q05:** the rewrite turned an AI Act question about reading employees' emotions into a GDPR question, so it never reached Article 5.

I would:
- make the grader less strict, or keep the best passage when all are rejected;
- show the retry the passages it rejected, so it really changes its wording.

I did not make these changes and re-run on the same 30 questions, because that would be tuning to the test. They need new questions.

**15. Why local models, and could this run with a paid model?**

The project rule was free tools only, and local models also keep confidential documents on the machine, which matters for banks. All models go through LangChain's `init_chat_model`, so switching to OpenAI, Claude or Gemini means changing `LLM_PROVIDER` and `LLM_MODEL` in `.env`, adding the key, and installing that provider's LangChain package. No code changes are needed. Temperature 0 and a fixed seed of 42 make the local runs as repeatable as possible.
