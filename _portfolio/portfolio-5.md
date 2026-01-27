---
title: "YouTube RAG — Ask Questions About Any Video"
excerpt: "Given a YouTube link, the notebook pulls the audio, turns it into a text transcript, breaks the transcript into small, searchable chunks, and builds a lightweight index over those chunks. When you ask a question, it looks up the most relevant transcript pieces and then drafts an answer grounded in those snippets on the vector database (optionally showing the supporting lines)."
header:
    teaser: "https://wayiok.github.io/academicpages.github.io/images/portfolio/p5_0.png"
repo: "https://github.com/wayiok/youtube-rag"
collection: portfolio
---

# Introduction

This is a high-level overview of the system built by this notebook. This notebook builds an end-to-end RAG workflow around a YouTube video. Starting from a public video URL, it pulls the audio, turns it into a clean transcript, breaks that text into small overlapping chunks, and prepares those chunks for semantic search. When you ask a question, the system looks up the most relevant transcript pieces and drafts an answer grounded in what was actually said in the video.

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p5-1.png)

# Objectives

The goal is to create a simple but solid pipeline you can reuse: extract and transcribe a video, index the transcript in a vector store, and retrieve the right excerpts to support question-answering. Along the way, the notebook aims to keep the setup practical on a MacBook M1, to use a better Whisper model than the base option for higher accuracy, and to harden the YouTube download and SSL steps so they work reliably without requiring Google sign-in for public videos.

# Results

By the end, you have a transcription.txt file for the video and an indexed collection of transcript chunks ready for retrieval. You can pose questions and get answers that cite the most relevant parts of the transcript, which makes the responses traceable and reduces hallucinations. 


# Conclusion

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p5-3.png)

You now have a repeatable RAG workflow for YouTube content: fetch, transcribe, chunk, index, retrieve, and answer. It’s lightweight enough to run locally and robust enough to handle common environment and network hurdles. From here, you can tune chunk sizes and top-k retrieval, try faster-whisper or GPU acceleration for speed, and wire the pipeline into a small app or API to turn any long video into a searchable, question-answerable resource.
