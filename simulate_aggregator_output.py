#!/usr/bin/env python3
"""
Simulazione dell'output dell'agente aggregatore con le nuove metriche.
Questo script mostra come appaiono i risultati aggregati nel sistema dopo l'estensione.
"""

def simulate_aggregated_output():
    """Simula l'output JSON dell'agente aggregatore esteso."""
    
    # Simulazione dei risultati aggregati che l'agente aggregatore ora produce
    simulated_aggregate_output = {
        "experiment_name": "coverage_genre_balance",
        "timestamp": "2025-06-17T15:30:00",
        "aggregate_mean": {
            "precision_at_k": {
                "map_at_k": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.0000,
                    "20": 0.0000,
                    "50": 0.0200
                },
                "mean_genre_coverage": 0.7593,
                "num_recommendations": 150,
                "num_unique_recommendations": 145
            },
            "coverage": {
                "map_at_k": {
                    "1": 0.3333,
                    "5": 0.1333,
                    "10": 0.0667,
                    "20": 0.0833,
                    "50": 0.0667
                },
                "mean_genre_coverage": 0.7963,
                # ⭐ NUOVE METRICHE AGGREGATE - GenreEntropy per coverage_genre_balance ⭐
                "genre_entropy": 3.0691,
                "avg_genre_entropy": 3.0691,  # Nome alternativo per agente aggregatore
                "num_recommendations": 150,
                "num_unique_recommendations": 134
            },
            "final": {
                "precision_scores_agg": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.1000,
                    "20": 0.0500,
                    "50": 0.0600
                },
                "genre_coverage": 0.7778,
                # ⭐ NUOVE METRICHE AGGREGATE - Tutte e tre le metriche per final ⭐
                "average_release_year": 1980.9,
                "temporal_dispersion": 8.75,
                "genre_entropy": 2.7843
            }
        }
    }
    
    # Simulazione per coverage_temporal
    simulated_temporal_output = {
        "experiment_name": "coverage_temporal",
        "timestamp": "2025-06-17T15:35:00",
        "aggregate_mean": {
            "precision_at_k": {
                "map_at_k": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.0000,
                    "20": 0.0000,
                    "50": 0.0000
                },
                "mean_genre_coverage": 0.7593,
                "num_recommendations": 150,
                "num_unique_recommendations": 140
            },
            "coverage": {
                "map_at_k": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.0000,
                    "20": 0.0000,
                    "50": 0.0000
                },
                "mean_genre_coverage": 0.7593,
                # ⭐ NUOVE METRICHE AGGREGATE - TempDisp per coverage_temporal ⭐
                "temporal_dispersion": 16.26,
                "avg_temporal_dispersion": 16.26,  # Nome alternativo per agente aggregatore
                "num_recommendations": 150,
                "num_unique_recommendations": 138
            },
            "final": {
                "precision_scores_agg": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.1000,
                    "20": 0.0500,
                    "50": 0.0400
                },
                "genre_coverage": 0.8333,
                # ⭐ NUOVE METRICHE AGGREGATE - Tutte e tre le metriche per final ⭐
                "average_release_year": 1985.6,
                "temporal_dispersion": 16.91,
                "genre_entropy": 3.2738
            }
        }
    }
    
    # Simulazione per precision_at_k_recency
    simulated_recency_output = {
        "experiment_name": "precision_at_k_recency",
        "timestamp": "2025-06-17T15:40:00",
        "aggregate_mean": {
            "precision_at_k": {
                "map_at_k": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.0000,
                    "20": 0.0000,
                    "50": 0.0000
                },
                "mean_genre_coverage": 0.7593,
                # ⭐ NUOVE METRICHE AGGREGATE - AvgYear per precision_at_k_recency ⭐
                "average_release_year": 1983.3,
                "avg_year": 1983.3,  # Nome alternativo per agente aggregatore
                "num_recommendations": 150,
                "num_unique_recommendations": 142
            },
            "coverage": {
                "map_at_k": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.0000,
                    "20": 0.0000,
                    "50": 0.0000
                },
                "mean_genre_coverage": 0.7593,
                "num_recommendations": 150,
                "num_unique_recommendations": 139
            },
            "final": {
                "precision_scores_agg": {
                    "1": 0.0000,
                    "5": 0.0000,
                    "10": 0.1000,
                    "20": 0.0500,
                    "50": 0.0400
                },
                "genre_coverage": 0.8333,
                # ⭐ NUOVE METRICHE AGGREGATE - Tutte e tre le metriche per final ⭐
                "average_release_year": 1985.6,
                "temporal_dispersion": 16.91,
                "genre_entropy": 3.2738
            }
        }
    }
    
    return [simulated_aggregate_output, simulated_temporal_output, simulated_recency_output]

def format_console_output():
    """Simula l'output della console dell'agente aggregatore."""
    
    console_outputs = [
        """
=== Agente Aggregatore - coverage_genre_balance ===
Metriche Aggregate (Medie su Utenti):
  Mean Precision_at_k: MAP@10=0.0000, Mean GenreCoverage=0.7593
  Mean Coverage: MAP@10=0.0667, Mean GenreCoverage=0.7963, GenreEntropy=3.0691
  Final Aggregated: P@10=0.1000, GenreCoverage=0.7778, AvgYear=1980.9, TempDisp=8.75, GenreEntropy=2.7843
  Total Item Coverage (all recs): 0.0452
        """,
        """
=== Agente Aggregatore - coverage_temporal ===
Metriche Aggregate (Medie su Utenti):
  Mean Precision_at_k: MAP@10=0.0000, Mean GenreCoverage=0.7593
  Mean Coverage: MAP@10=0.0000, Mean GenreCoverage=0.7593, TempDisp=16.26
  Final Aggregated: P@10=0.1000, GenreCoverage=0.8333, AvgYear=1985.6, TempDisp=16.91, GenreEntropy=3.2738
  Total Item Coverage (all recs): 0.0458
        """,
        """
=== Agente Aggregatore - precision_at_k_recency ===
Metriche Aggregate (Medie su Utenti):
  Mean Precision_at_k: MAP@10=0.0000, Mean GenreCoverage=0.7593, AvgYear=1983.3
  Mean Coverage: MAP@10=0.0000, Mean GenreCoverage=0.7593
  Final Aggregated: P@10=0.1000, GenreCoverage=0.8333, AvgYear=1985.6, TempDisp=16.91, GenreEntropy=3.2738
  Total Item Coverage (all recs): 0.0461
        """
    ]
    
    return console_outputs

def format_html_report_section():
    """Simula come appare la sezione HTML nel report."""
    
    html_section = """
    <div class='metric-block'>
        <h5>Precision/MAP @k (e.g., @10)</h5>
        <table>
            <tr><th>Metrica</th><th>Valore @10</th></tr>
            <tr><td>MAP@10 (precision_at_k)</td><td>0.0000</td></tr>
            <tr><td>AvgYear (precision_at_k)</td><td>1983.3</td></tr>  <!-- ⭐ NUOVA -->
            <tr><td>MAP@10 (coverage)</td><td>0.0000</td></tr>
            <tr><td>AvgTempDisp (coverage)</td><td>16.26</td></tr>     <!-- ⭐ NUOVA -->
            <tr><td>AvgGenreEntropy (coverage)</td><td>3.0691</td></tr> <!-- ⭐ NUOVA -->
        </table>
    </div>
    """
    
    return html_section

def format_markdown_report_section():
    """Simula come appare la sezione Markdown nel report."""
    
    markdown_section = """
#### Metriche Calcolate (Aggregate)
| Strategia/Tipo | Indicatore | Valore @10 (o Aggregato) | Altre Metriche Specifiche |
|---|---|---|---|
| Precision Strategy | MAP@10 | 0.0000 | AvgYear: 1983.3 |  <!-- ⭐ NUOVA -->
| Coverage Strategy | MAP@10 | 0.0000 | TempDisp: 16.26; GenreEntropy: 3.0691 |  <!-- ⭐ NUOVE -->
| Final Recs | P@10 | 0.1000 | AvgYear: 1985.6; TempDisp: 16.91; GenreEntr: 3.2738 |
| Precision Strategy | Mean Genre Coverage | 0.7593 | - |
| Coverage Strategy | Mean Genre Coverage | 0.7593 | - |
| Final Recs | Genre Coverage | 0.8333 | - |
    """
    
    return markdown_section

if __name__ == "__main__":
    print("=== ESEMPIO SIMULATO DELL'OUTPUT DELL'AGENTE AGGREGATORE ESTESO ===\n")
    
    print("1. STRUTTURA JSON DEI RISULTATI AGGREGATI:\n")
    outputs = simulate_aggregated_output()
    
    for i, output in enumerate(outputs):
        experiment_name = output['experiment_name']
        print(f"Esperimento: {experiment_name}")
        print("Nuove metriche aggregate presenti:")
        
        aggregate_mean = output['aggregate_mean']
        for strategy, data in aggregate_mean.items():
            new_metrics = []
            if 'average_release_year' in data or 'avg_year' in data:
                avg_year = data.get('average_release_year', data.get('avg_year'))
                new_metrics.append(f"AvgYear={avg_year}")
            if 'temporal_dispersion' in data or 'avg_temporal_dispersion' in data:
                temp_disp = data.get('temporal_dispersion', data.get('avg_temporal_dispersion'))
                new_metrics.append(f"TempDisp={temp_disp}")
            if 'genre_entropy' in data or 'avg_genre_entropy' in data:
                genre_ent = data.get('genre_entropy', data.get('avg_genre_entropy'))
                new_metrics.append(f"GenreEntropy={genre_ent}")
            
            if new_metrics:
                print(f"  - {strategy}: {', '.join(new_metrics)}")
        print()
    
    print("2. OUTPUT DELLA CONSOLE:\n")
    console_outputs = format_console_output()
    for output in console_outputs:
        print(output.strip())
        print()
    
    print("3. SEZIONE HTML NEL REPORT:\n")
    print(format_html_report_section())
    
    print("4. SEZIONE MARKDOWN NEL REPORT:\n")
    print(format_markdown_report_section())
    
    print("\n=== RIEPILOGO DELLE IMPLEMENTAZIONI ===")
    print("✅ GenreEntropy: Calcolata e aggregata per esperimenti coverage_genre_balance")
    print("✅ TempDisp: Calcolata e aggregata per esperimenti coverage_temporal")  
    print("✅ AvgYear: Calcolata e aggregata per esperimenti precision_at_k_recency")
    print("✅ Tutte e tre le metriche: Sempre calcolate per le raccomandazioni finali aggregate")
    print("✅ Report HTML: Esteso per mostrare le nuove metriche aggregate")
    print("✅ Report Markdown: Esteso per mostrare le nuove metriche aggregate")
    print("✅ Console Output: Esteso per mostrare le nuove metriche aggregate")
    print("✅ Nomi alternativi: Supportati per compatibilità con l'agente aggregatore")
