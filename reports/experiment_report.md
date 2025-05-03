# Report Esperimenti (2025-05-03 19:01)

## Statistiche Generali
- **Totale Esperimenti**: 5
- **Raccomandazioni Finali Uniche**: 13

### Film più Raccomandati (Finali)
| Film | Frequenza |
|---|---|
| Dirty Dancing (1987) | 2 |
| Cinema Paradiso (1988) | 2 |
| Pleasantville (1998) | 1 |
| Gordy (1995) | 1 |
| 187 (1997) | 1 |
| Jeffrey (1995) | 1 |
| Rob Roy (1995) | 1 |
| Cosi (1996) | 1 |
| Matrix, The (1999) | 1 |
| Casino (1995) | 1 |
| Sense and Sensibility (1995) | 1 |
| Jerry Maguire (1996) | 1 |
| Pyromaniac's Love Story, A (1995) | 1 |

## Performance per Metrica (Aggregata)
### Metrica: precision_at_k
- **Raccomandazioni Uniche (precision_at_k)**: 25
#### Temi nelle Spiegazioni
| Tema | Frequenza |
|---|---|
| generi | 5 |

### Metrica: coverage
- **Raccomandazioni Uniche (coverage)**: 21
#### Temi nelle Spiegazioni
| Tema | Frequenza |
|---|---|
| diversità | 7 |
| generi | 7 |

## Dettagli Esperimenti
### combined_serendipity_temporal
*Data: 2025-05-03T19:00:50.557991*

#### Varianti Prompt
```json
{
  "precision_at_k": "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K con focus sulla SERENDIPITY...",
  "coverage": "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE TEMPORALE..."
}
```

#### Raccomandazioni per Metrica (per Utente)
##### Utente 1
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| precision_at_k | Pleasantville (1998), Meet Joe Black (1998), 10 Things I Hate About You (1999) | Le raccomandazioni sono ottimizzate per PRECISION@K con un focus sulla SERENDIPITY. I film selezionati includono elementi di sorpresa e scoperta, come 'Pleasantville' e 'Meet Joe Black', che sono film drammatici con elementi romantici, e '10 Things I Hate About You', una commedia romantica, che offrono esperienze cinematografiche uniche e potenzialmente inaspettate per l'utente. |
| coverage | Gordy (1995), North (1994), Smile Like Yours, A (1997) | Queste raccomandazioni ottimizzano la copertura temporale includendo film da diversi anni per offrire una varietà cronologica. I film selezionati includono 'Gordy (1995)', 'North (1994)' e 'Smile Like Yours, A (1997)', coprendo un arco di tempo che va dal 1994 al 1997, garantendo così una varietà temporale nelle raccomandazioni. |

##### Utente 2
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| precision_at_k | Dirty Dancing (1987), Kolya (1996), General, The (1998) | Le raccomandazioni sono basate sulla precisione e la serendipità. 'Dirty Dancing' (1088) è un film musicale romantico che potrebbe sorprendere l'utente con la sua energia e passione. 'Kolya' (1446) è una commedia che potrebbe offrire una piacevole variazione rispetto ai film drammatici che l'utente ha apprezzato. 'The General' (2425) è un film criminale che potrebbe interessare l'utente per la sua trama avvincente e la regia unica. |
| coverage | Toy Story (1995), Leaving Las Vegas (1995), City of Lost Children, The (1995) | Questi film sono stati scelti per massimizzare la copertura temporale, includendo film di vari anni per garantire una varietà di esperienze cinematografiche. |

#### Valutazione Finale Aggregata
- **Raccomandazioni Finali**: Pleasantville (1998), Gordy (1995), Dirty Dancing (1987)
- **Giustificazione**: La logica di selezione si basa sul bilanciamento tra precisione e copertura temporale. Il film con ID 2321 è stato scelto per la sua serendipità e precisione, offrendo un'esperienza cinematografica unica e potenzialmente inaspettata per l'utente, come 'Pleasantville'. Il film con ID 243 è stato selezionato per la sua copertura temporale, rappresentando un film del 1995 e offrendo una varietà cronologica. Infine, il film con ID 1088 è stato incluso per la sua precisione e serendipità, essendo un film musicale romantico che potrebbe sorprendere l'utente con la sua energia e passione, come 'Dirty Dancing'.
- **Trade-offs**: Nel processo di selezione, abbiamo dovuto sacrificare alcuni aspetti di copertura temporale per mantenere una buona precisione e serendipità. Ad esempio, abbiamo scelto film che offrono esperienze uniche e inaspettate piuttosto che massimizzare la varietà cronologica. Inoltre, abbiamo bilanciato tra film popolari e di nicchia, scegliendo film che potrebbero sia soddisfare le aspettative dell'utente che offrire nuove scoperte.

#### Metriche Calcolate
| Metrica | Precision@k | Genre Coverage |
|---|---|---|
| Precision | 0.1667 | 0.2222 |
| Coverage | 0.3333 | 0.3889 |
| Final | 0.0000 | 0.1667 |

- **Total Item Coverage**: 0.0031

---
### coverage_genre_balance
*Data: 2025-05-03T19:01:02.100133*

#### Varianti Prompt
```json
{
  "coverage": "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE con BILANCIAMENTO DEI GENERI...",
  "precision_at_k": "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K. Il tuo obiettivo è raccomandare film che l'utente valuterà con un rating 4 o 5 su 5. Analizza attentamente il profilo dell'utente e concentrati sui seguenti elementi:\n1. Generi che l'utente ha costantemente valutato con punteggi elevati\n2. Attributi specifici (attori, registi, temi, epoche) presenti nei film apprezzati\n3. Modelli nelle valutazioni negative per evitare film simili\n\nLa precision@k misura quanti dei film raccomandati saranno effettivamente valutati positivamente. Quando analizzi il catalogo, presta particolare attenzione a:\n- Corrispondenza di genere con i film valutati positivamente\n- Somiglianza tematica e stilistica ai film preferiti\n- Evita film simili a quelli che l'utente non ha apprezzato\n\nNON raccomandare film in base alla popolarità generale o alle tendenze, a meno che queste caratteristiche non si allineino alle preferenze specifiche di questo utente. Fornisci ESATTAMENTE 3 film che l'utente molto probabilmente valuterà positivamente."
}
```

#### Raccomandazioni per Metrica (per Utente)
##### Utente 1
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| coverage | Sense and Sensibility (1995), Get Shorty (1995), Copycat (1995) | Queste raccomandazioni sono state ottimizzate per la copertura dei generi, garantendo una varietà di film che coprono diversi generi come Drama, Romance, Action, Comedy, Crime, e Thriller. Questo approccio assicura che l'utente abbia un'esperienza di visione diversificata. |
| precision_at_k | Cinema Paradiso (1988), When Harry Met Sally... (1989), Benny & Joon (1993) | L'utente ha mostrato una forte preferenza per film di genere Comedy e Romance. 'Cinema Paradiso' (1172) è un film che combina elementi di commedia, dramma e romanticismo, simile a molti film apprezzati dall'utente. 'When Harry Met Sally...' (1307) è una classica commedia romantica che si allinea con il gusto dell'utente per film di questo genere. 'Benny & Joon' (1441) è un'altra commedia romantica che si adatta bene alle preferenze dell'utente, evitando generi e temi che l'utente ha dimostrato di non apprezzare. |

##### Utente 2
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| coverage | Dirty Dancing (1987), Private Benjamin (1980), Fandango (1985) | Questi film sono stati selezionati per ottimizzare la copertura dei generi e bilanciare i generi popolari con quelli meno comuni nel catalogo. 'Dirty Dancing' (1088) copre il genere Musical/Romance, 'Private Benjamin' (1135) copre il genere Comedy, e 'Fandango' (2073) copre il genere Comedy, offrendo una varietà di opzioni che l'utente potrebbe non aver ancora esplorato. |
| precision_at_k | 187 (1997), Friends & Lovers (1999), Krippendorf's Tribe (1998) | L'utente ha mostrato una preferenza per i film drammatici e commedie romantiche. '187' (1609) è un dramma intenso che potrebbe risuonare con le preferenze drammatiche dell'utente. 'Friends & Lovers' (2589) è una commedia romantica che combina elementi di dramma, simile a molti dei film apprezzati dall'utente. 'Krippendorf's Tribe' (1855) è una commedia che potrebbe offrire un'esperienza divertente e leggera, in linea con i gusti dell'utente per le commedie. |

#### Valutazione Finale Aggregata
- **Raccomandazioni Finali**: Cinema Paradiso (1988), Dirty Dancing (1987), 187 (1997)
- **Giustificazione**: La selezione dei film è stata fatta considerando sia la precisione che la copertura dei generi. 'Cinema Paradiso' (1172) è stato scelto per la sua alta precisione e perché combina elementi di commedia, dramma e romanticismo, che sono generi apprezzati dall'utente. 'Dirty Dancing' (1088) è stato scelto per la sua capacità di coprire il genere Musical/Romance, offrendo una varietà di opzioni che l'utente potrebbe non aver ancora esplorato. Infine, '187' (1609) è stato scelto per la sua rilevanza come dramma intenso, che risuona con le preferenze drammatiche dell'utente.
- **Trade-offs**: Abbiamo sacrificato un po' di precisione per ottenere una maggiore copertura dei generi, assicurando che l'utente abbia un'esperienza di visione diversificata. Inoltre, abbiamo bilanciato la scelta tra film popolari e di nicchia per garantire che l'utente possa scoprire nuovi generi senza perdere il focus sui suoi gusti principali.

#### Metriche Calcolate
| Metrica | Precision@k | Genre Coverage |
|---|---|---|
| Precision | 0.3333 | 0.1667 |
| Coverage | 0.1667 | 0.3889 |
| Final | 0.3333 | 0.2222 |

- **Total Item Coverage**: 0.0031

---
### coverage_temporal
*Data: 2025-05-03T19:01:11.980605*

#### Varianti Prompt
```json
{
  "coverage": "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE TEMPORALE...",
  "precision_at_k": "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K. Il tuo obiettivo è raccomandare film che l'utente valuterà con un rating 4 o 5 su 5. Analizza attentamente il profilo dell'utente e concentrati sui seguenti elementi:\n1. Generi che l'utente ha costantemente valutato con punteggi elevati\n2. Attributi specifici (attori, registi, temi, epoche) presenti nei film apprezzati\n3. Modelli nelle valutazioni negative per evitare film simili\n\nLa precision@k misura quanti dei film raccomandati saranno effettivamente valutati positivamente. Quando analizzi il catalogo, presta particolare attenzione a:\n- Corrispondenza di genere con i film valutati positivamente\n- Somiglianza tematica e stilistica ai film preferiti\n- Evita film simili a quelli che l'utente non ha apprezzato\n\nNON raccomandare film in base alla popolarità generale o alle tendenze, a meno che queste caratteristiche non si allineino alle preferenze specifiche di questo utente. Fornisci ESATTAMENTE 3 film che l'utente molto probabilmente valuterà positivamente."
}
```

#### Raccomandazioni per Metrica (per Utente)
##### Utente 1
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| coverage | Jeffrey (1995), Sum of Us, The (1994), Smile Like Yours, A (1997) | Queste raccomandazioni sono state ottimizzate per la copertura temporale, includendo film di diversi anni per garantire una varietà di contenuti e mantenere l'interesse dell'utente nel tempo. |
| precision_at_k | City of Lost Children, The (1995), Cinema Paradiso (1988), Strictly Ballroom (1992) | L'utente ha mostrato una forte preferenza per film di genere Drama e Comedy, con una predilezione per film romantici e d'avventura. 'The City of Lost Children' (1995) è un film d'avventura e fantascienza che potrebbe piacere per il suo mix di generi. 'Cinema Paradiso' (1988) è un film drammatico e romantico che ha elementi di commedia, in linea con le preferenze dell'utente. 'Strictly Ballroom' (1992) è una commedia romantica che si adatta perfettamente ai gusti dell'utente per film che combinano romanticismo e umorismo. |

##### Utente 2
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| coverage | Casino (1995), Sense and Sensibility (1995), Rob Roy (1995) | Questi film sono stati raccomandati per ottimizzare la copertura temporale del catalogo. 'Casino (1995)', 'Sense and Sensibility (1995)' e 'Rob Roy (1995)' coprono vari generi come Drama, Thriller, Romance e War, garantendo una varietà di esperienze cinematografiche che l'utente potrebbe apprezzare. |
| precision_at_k | English Patient, The (1996), Amos & Andrew (1993), Mr. & Mrs. Smith (1941) | L'utente ha mostrato una forte preferenza per i film drammatici e romantici, come 'The English Patient' (1183), che combina elementi di dramma, romanticismo e guerra. Inoltre, 'Amos & Andrew' (1440) e 'Mr. & Mrs. Smith' (2205) sono commedie che si allineano con la tendenza dell'utente a gradire film con una buona dose di umorismo e romanticismo. |

#### Valutazione Finale Aggregata
- **Raccomandazioni Finali**: Cinema Paradiso (1988), Jeffrey (1995), Rob Roy (1995)
- **Giustificazione**: Ho selezionato questi 3 film per bilanciare rilevanza e diversità temporale. 'Cinema Paradiso' (1172) è stato scelto per la sua alta rilevanza nei confronti delle preferenze dell'utente per film drammatici e romantici, con un buon mix di commedia. 'Jeffrey' (171) è stato scelto per la sua copertura temporale e per il genere Comedy, che diversifica le raccomandazioni. Infine, 'Rob Roy' (151) è stato scelto per la sua copertura temporale e per il genere Drama, che aggiunge ulteriore varietà e mantiene l'interesse dell'utente nel tempo.
- **Trade-offs**: Ho considerato un trade-off tra precisione e copertura. 'Cinema Paradiso' (1172) è altamente rilevante ma meno diversificato temporalmente. 'Jeffrey' (171) e 'Rob Roy' (151) sono meno rilevanti rispetto alle preferenze specifiche dell'utente, ma offrono una copertura temporale e di genere che garantisce varietà e mantiene l'interesse dell'utente nel tempo. Questo bilanciamento permette di soddisfare sia la rilevanza che la diversità temporale.

#### Metriche Calcolate
| Metrica | Precision@k | Genre Coverage |
|---|---|---|
| Precision | 0.3333 | 0.3333 |
| Coverage | 0.1667 | 0.2778 |
| Final | 0.3333 | 0.2222 |

- **Total Item Coverage**: 0.0031

---
### precision_at_k_recency
*Data: 2025-05-03T19:00:47.666650*

#### Varianti Prompt
```json
{
  "precision_at_k": "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K con focus sulla RECENCY...",
  "coverage": "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE. Data una lista di film, consiglia 3 film che massimizzano la copertura di diversi generi cinematografici, MA che siano comunque rilevanti per le preferenze specifiche dell'utente di cui stai analizzando il profilo. La coverage misura la proporzione dell'intero catalogo che il sistema è in grado di raccomandare. L'obiettivo è esplorare meglio lo spazio dei film disponibili e ridurre il rischio di filter bubble. Assicurati che le tue raccomandazioni coprano generi diversi tra loro, ma che siano allineati con i gusti dell'utente. IMPORTANTE: Fai riferimento specifico ai film che l'utente ha apprezzato per scoprire generi correlati ma diversi. Ogni utente deve ricevere raccomandazioni personalizzate in base al suo profilo unico. NON raccomandare più di 3 film."
}
```

#### Raccomandazioni per Metrica (per Utente)
##### Utente 1
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| precision_at_k | Guy (1996), Pleasantville (1998), You've Got Mail (1998) | I film raccomandati sono stati selezionati per ottimizzare la PRECISION@K concentrandosi su film recenti e basandosi sui generi preferiti dall'utente, come Comedy, Drama, e Romance, che sono ricorrenti nei film che l'utente ha apprezzato. |
| coverage | From Dusk Till Dawn (1996), Leaving Las Vegas (1995), Species (1995) | L'utente ha mostrato un apprezzamento per film di vari generi, inclusi Comedy, Drama e Adventure. Per massimizzare la coverage e coprire generi diversi ma rilevanti, ho raccomandato 'From Dusk Till Dawn' (Action|Comedy|Crime|Horror|Thriller), 'Leaving Las Vegas' (Drama|Romance) e 'Species' (Horror|Sci-Fi). Questi film coprono generi diversi tra loro, mantenendo comunque una rilevanza con le preferenze dell'utente. |

##### Utente 2
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| precision_at_k | Matrix, The (1999), October Sky (1999), 20 Dates (1998) | Le raccomandazioni si basano su film recenti che l'utente potrebbe apprezzare, considerando generi e preferenze passate. 'The Matrix' (2571) è un film d'azione e fantascienza del 1999, 'October Sky' (2501) è un dramma del 1999, e '20 Dates' (2492) è una commedia del 1998, tutti generi che l'utente ha dimostrato di apprezzare. |
| coverage | Toy Story (1995), Casino (1995), Sense and Sensibility (1995) | Basandomi sulle preferenze dell'utente e sulla necessità di massimizzare la copertura dei generi, ho scelto 'Toy Story (1995)' per il genere 'Animation|Children's|Comedy', 'Casino (1995)' per 'Drama|Thriller' e 'Sense and Sensibility (1995)' per 'Drama|Romance'. Questi film non solo coprono generi diversi, ma sono anche allineati con i gusti dell'utente, garantendo una buona esplorazione del catalogo e riducendo il rischio di filter bubble. |

#### Valutazione Finale Aggregata
- **Raccomandazioni Finali**: Cosi (1996), Matrix, The (1999), Casino (1995)
- **Giustificazione**: Ho scelto questi 3 film specifici per bilanciare al meglio precisione e copertura. Il film 'Cosi (1996)' (ID 705) è stato selezionato per la sua alta rilevanza, essendo una commedia che è un genere preferito dall'utente. 'The Matrix' (ID 2571) è stato incluso per la sua precisione, essendo un film d'azione e fantascienza che è stato apprezzato dall'utente. Infine, 'Casino (1995)' (ID 16) è stato scelto per aumentare la copertura dei generi, introducendo 'Drama|Thriller' e garantendo una buona esplorazione del catalogo.
- **Trade-offs**: Ho dovuto sacrificare un po' di precisione per ottenere una maggiore copertura dei generi. Ad esempio, 'Casino' (ID 16) potrebbe non essere il film più preciso per l'utente, ma offre una diversità di generi che è importante per evitare la filter bubble. Inoltre, ho scelto di includere film popolari come 'The Matrix' (ID 2571) per garantire una buona accettazione da parte dell'utente, anche se questo potrebbe ridurre leggermente la copertura dei generi meno comuni.

#### Metriche Calcolate
| Metrica | Precision@k | Genre Coverage |
|---|---|---|
| Precision | 0.3333 | 0.3333 |
| Coverage | 0.3333 | 0.5556 |
| Final | 0.3333 | 0.2778 |

- **Total Item Coverage**: 0.0033

---
### precision_at_k_serendipity
*Data: 2025-05-03T19:00:54.184444*

#### Varianti Prompt
```json
{
  "precision_at_k": "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K con focus sulla SERENDIPITY...",
  "coverage": "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE. Data una lista di film, consiglia 3 film che massimizzano la copertura di diversi generi cinematografici, MA che siano comunque rilevanti per le preferenze specifiche dell'utente di cui stai analizzando il profilo. La coverage misura la proporzione dell'intero catalogo che il sistema è in grado di raccomandare. L'obiettivo è esplorare meglio lo spazio dei film disponibili e ridurre il rischio di filter bubble. Assicurati che le tue raccomandazioni coprano generi diversi tra loro, ma che siano allineati con i gusti dell'utente. IMPORTANTE: Fai riferimento specifico ai film che l'utente ha apprezzato per scoprire generi correlati ma diversi. Ogni utente deve ricevere raccomandazioni personalizzate in base al suo profilo unico. NON raccomandare più di 3 film."
}
```

#### Raccomandazioni per Metrica (per Utente)
##### Utente 1
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| precision_at_k | English Patient, The (1996), Jerry Maguire (1996), Wild Things (1998) | Questi film sono stati selezionati per ottimizzare la PRECISION_AT_K con un focus sulla SERENDIPITY. 'The English Patient' (1183) è un dramma romantico che potrebbe piacere all'utente, data la sua preferenza per film drammatici e romantici. 'Jerry Maguire' (1393) è un altro film drammatico-romantico che aggiunge un elemento di sorpresa. 'Wild Things' (1805) è un thriller che combina elementi di crimine e mistero, offrendo una scelta inaspettata ma potenzialmente gradita. |
| coverage | Sense and Sensibility (1995), Copycat (1995), Pyromaniac's Love Story, A (1995) | L'utente ha mostrato interesse per film di vari generi, inclusi Drama, Romance e Comedy. Per massimizzare la coverage e mantenere la rilevanza, ho selezionato 'Sense and Sensibility (1995)' (movie_id: 17) per coprire Drama e Romance, 'Copycat (1995)' (movie_id: 22) per Crime, Drama e Thriller, e 'Pyromaniac's Love Story, A (1995)' (movie_id: 295) per Comedy e Romance. Questi film coprono diversi generi correlati ma distinti, aumentando la varietà delle raccomandazioni. |

##### Utente 2
| Metrica | Raccomandazioni | Spiegazione |
|---|---|---|
| precision_at_k | Sense and Sensibility (1995), City of Lost Children, The (1995), Krippendorf's Tribe (1998) | Ho scelto questi film perché combinano elementi di dramma e commedia, generi che l'utente ha mostrato di apprezzare, ma con un tocco di serendipità, offrendo titoli che potrebbero sorprendere positivamente l'utente. |
| coverage | Schindler's List (1993), Male and Female (1919), Dirty Dancing (1987) | L'utente ha apprezzato film di vari generi, inclusi Drama, Comedy e Romance. Per massimizzare la copertura dei generi, ho raccomandato 'Schindler's List' (Drama|War), 'Male and Female' (Adventure|Drama) e 'Dirty Dancing' (Musical|Romance). Questi film coprono generi come Drama, War, Adventure e Musical, offrendo una diversità di esperienze cinematografiche pur rimanendo rilevanti per le preferenze dell'utente. |

#### Valutazione Finale Aggregata
- **Raccomandazioni Finali**: Sense and Sensibility (1995), Jerry Maguire (1996), Pyromaniac's Love Story, A (1995)
- **Giustificazione**: Ho scelto questi 3 film specifici come bilanciamento ottimale per l'utente medio rappresentato dai dati forniti. Il film 'Sense and Sensibility' (movie_id: 17) è stato selezionato per la sua capacità di coprire i generi Drama e Romance, che sono di interesse per l'utente. Questo film proviene dalla raccomandazione basata sulla 'coverage', garantendo una buona diversità di generi. 'Jerry Maguire' (movie_id: 1393) è stato scelto per la sua alta precisione, combinando elementi di dramma e commedia con un tocco di serendipità, offrendo titoli che potrebbero sorprendere positivamente l'utente. Infine, 'A Pyromaniac's Love Story' (movie_id: 295) è stato incluso per la sua capacità di coprire i generi Comedy e Romance, contribuendo ulteriormente alla diversità delle raccomandazioni.
- **Trade-offs**: Nel processo di selezione, ho considerato l'importanza di bilanciare la precisione con la copertura dei generi. Ho sacrificato un po' di precisione per includere film che coprono una gamma più ampia di generi, come 'Sense and Sensibility' e 'A Pyromaniac's Love Story', per garantire una maggiore diversità. Tuttavia, ho mantenuto un film ad alta precisione, 'Jerry Maguire', per assicurare che la lista finale sia ancora rilevante per le preferenze dell'utente.

#### Metriche Calcolate
| Metrica | Precision@k | Genre Coverage |
|---|---|---|
| Precision | 0.3333 | 0.5000 |
| Coverage | 0.3333 | 0.4444 |
| Final | 0.3333 | 0.1667 |

- **Total Item Coverage**: 0.0028

---
