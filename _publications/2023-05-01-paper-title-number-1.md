---
title: "Using Data Mining Techniques for the Detection of SQL Injection Attacks on Database Systems"
collection: publications
category: manuscripts
permalink: /publication/2023-05-01-paper-title-number-1
excerpt: 'This article presents a data mining project using CRISP-DM and kNN clustering to detect SQL injection attacks in PostgreSQL logs via Python, achieving over 80% accuracy with a 60-minute runtime on 500k entries.'
date: 2023-05-01
venue: 'Vol. 51 Num. 2 (2023): Revista Politécnica (May-July 2023)'
#slidesurl: 'http://academicpages.github.io/files/slides1.pdf'
paperurl: 'https://www.researchgate.net/publication/370481672_Using_Data_Mining_Techniques_for_the_Detection_of_SQL_Injection_Attacks_on_Database_Systems'
citation: 'Añasco Loor, C., Morocho, K. y Hallo, M. 2023. Uso de Técnicas de Minería de Datos para la Detección de Ataques de Inyección de SQL en Sistemas de Bases de Datos . Revista Politécnica. 51, 2 (may 2023), 19–28. DOI:https://doi.org/10.33333/rp.vol51n2.02.'
#bibtexurl: 'http://academicpages.github.io/files/bibtex1.bib'
---


![Download Paper](https://www.researchgate.net/publication/370481672_Using_Data_Mining_Techniques_for_the_Detection_of_SQL_Injection_Attacks_on_Database_Systems/fulltext/6452805797449a0e1a76042a/Using-Data-Mining-Techniques-for-the-Detection-of-SQL-Injection-Attacks-on-Database-Systems.pdf?origin=publicationDetail&_sg%5B0%5D=U2cyMrghNJW5RlzLJ7sepgz8vsGHTVfGO1hZoiBhwNNZ8mXWPF8PhG80aE5rloliI-ri5EdNUjYxb5e5JTYcog.18C1VQ7tcDzB4S6eBcJX6TC97n4i7x4A_a9w5o4TT1SpoISg6trQ2CYbjqlsAX_zaiT2dp69Y9N0Ty6TDXQ8fA&_sg%5B1%5D=k1CYbx_UuWFv5NTPVENL0hl8y2Z1NYNGI6eI8TBBi2d4W0WlFWLeP1s09sWWLXpCW3banvvNjenX4qxPLtjwb81rzCS3JkIloSHuqJsgSl5_.18C1VQ7tcDzB4S6eBcJX6TC97n4i7x4A_a9w5o4TT1SpoISg6trQ2CYbjqlsAX_zaiT2dp69Y9N0Ty6TDXQ8fA&_iepl=&_rtd=eyJjb250ZW50SW50ZW50IjoibWFpbkl0ZW0ifQ%3D%3D&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIiwicG9zaXRpb24iOiJwYWdlSGVhZGVyIn19) 

In any business organization, database infrastructures are subject to various structured query language (SQL) injection attacks, such as tautologies, alternative coding, stored procedures, use of the union operator, piggyback, among others. This article describes a data mining project developed to mitigate the problem of identifying SQL injection attacks on databases. The project was conducted using an adaptation of the cross-industry standard process for data mining (CRISP-DM) methodology. A total of 12 python libraries was used for cleaning, transformation, and modeling. The anomaly detection model was carried out using clustering by the k – nearest neighbors (kNN) algorithm. The query text was analyzed for the groups with anomalies to identify sentences presenting attack traces. A web interface was implemented to display the daily summary of the attacks found. The information source was obtained from the transactions log of a PostgreSQL database server. Our results allowed the identification of different attacks by injection of SQL code above 80%. The execution time for processing half a million transaction log was approximately 60 minutes using a computer with the following characteristics: Intel® Core i7 processor 7th generation, 12GB RAM and 500GB SSD.
