---
title: "Using Data Mining Techniques for the Detection of SQL Injection Attacks on Database Systems"
collection: publications
category: manuscripts
permalink: /publication/2023-05-01-paper-title-number-1
#excerpt: 'This paper is about the number 1. The number 2 is left for future work.'
date: 2023-05-01
venue: '
Vol. 51 Num. 2 (2023): Revista Politécnica
(May-July 2023)'
#slidesurl: 'http://academicpages.github.io/files/slides1.pdf'
paperurl: 'https://www.researchgate.net/publication/370481672_Using_Data_Mining_Techniques_for_the_Detection_of_SQL_Injection_Attacks_on_Database_Systems'
#bibtexurl: 'http://academicpages.github.io/files/bibtex1.bib'
#citation: 'Your Name, You. (2009). &quot;Paper Title Number 1.&quot; <i>Journal 1</i>. 1(1).'
---



In any business organization, database infrastructures are subject to various structured query language (SQL) injection attacks, such as tautologies, alternative coding, stored procedures, use of the union operator, piggyback, among others. This article describes a data mining project developed to mitigate the problem of identifying SQL injection attacks on databases. The project was conducted using an adaptation of the cross-industry standard process for data mining (CRISP-DM) methodology. A total of 12 python libraries was used for cleaning, transformation, and modeling. The anomaly detection model was carried out using clustering by the k – nearest neighbors (kNN) algorithm. The query text was analyzed for the groups with anomalies to identify sentences presenting attack traces. A web interface was implemented to display the daily summary of the attacks found. The information source was obtained from the transactions log of a PostgreSQL database server. Our results allowed the identification of different attacks by injection of SQL code above 80%. The execution time for processing half a million transaction log was approximately 60 minutes using a computer with the following characteristics: Intel® Core i7 processor 7th generation, 12GB RAM and 500GB SSD.