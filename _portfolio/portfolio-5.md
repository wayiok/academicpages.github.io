---
title: "YouTube RAG — Ask Questions About Any Video"
excerpt: "Given a YouTube link, the notebook pulls the audio, turns it into a text transcript, breaks the transcript into small, searchable chunks, and builds a lightweight index over those chunks. When you ask a question, it looks up the most relevant transcript pieces and then drafts an answer grounded in those snippets on the vector database (optionally showing the supporting lines)."
header:
    teaser: "https://wayiok.github.io/academicpages.github.io/images/portfolio/p5.png"
repo: "https://github.com/wayiok/youtube-rag"
collection: portfolio
---

# Introduction

This is a high-level overview of the system built by this notebook. The objective is to create a RAG application from scratch, starting with the extraction of text from a YouTube video, the creation of a vector database, and the use of LangChain with cosine similarity to find related questions to the transcript.

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p5-1.png)




