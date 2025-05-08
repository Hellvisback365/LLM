"""
Schemi Pydantic per il sistema di raccomandazione.

Questo modulo contiene le definizioni degli schemi di input e output per il sistema di raccomandazione,
utilizzando Pydantic per la validazione dei dati.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator

# Importa la costante dal modulo prompt_manager
from src.recommender.core.prompt_manager import NUM_RECOMMENDATIONS

class RecommendationOutput(BaseModel):
    """Schema per l'output dei tool di raccomandazione per metrica."""
    recommendations: List[int] = Field(
        ..., 
        description=f"Lista ORDINATA di esattamente {NUM_RECOMMENDATIONS} ID numerici di film raccomandati. Il primo ID è il più raccomandato, l'ultimo il meno.",
        min_items=NUM_RECOMMENDATIONS,
        max_items=NUM_RECOMMENDATIONS
    )
    explanation: str = Field(..., description="Breve spiegazione testuale del motivo per cui questi film sono stati scelti e ordinati in base alla metrica richiesta.")
    
    @field_validator('recommendations')
    def validate_exactly_50_items(cls, v):
        """Validatore che garantisce esattamente NUM_RECOMMENDATIONS elementi."""
        if len(v) != NUM_RECOMMENDATIONS:
            raise ValueError(f"L'array deve contenere esattamente {NUM_RECOMMENDATIONS} elementi, trovati {len(v)}")
        return v
    
    @field_validator('recommendations')
    def validate_ids_are_integers(cls, v_list: List[int]):
        """Validatore che garantisce che ogni elemento della lista sia un ID numerico intero."""
        for item in v_list:
            if not isinstance(item, int):
                raise ValueError(f"Ogni ID film deve essere un intero, trovato {type(item)}")
        return v_list

class EvaluationOutput(BaseModel):
    """Schema per l'output del tool di valutazione finale."""
    final_recommendations: List[int] = Field(
        ..., 
        description=f"Lista finale OTTIMALE e ORDINATA di esattamente {NUM_RECOMMENDATIONS} ID numerici di film, bilanciando le metriche. Il primo ID è il più raccomandato.",
        min_items=NUM_RECOMMENDATIONS,
        max_items=NUM_RECOMMENDATIONS
    )
    justification: str = Field(..., description="Spiegazione dettagliata della logica di selezione, bilanciamento e ORDINAMENTO per la lista finale aggregata.")
    trade_offs: str = Field(..., description="Descrizione dei trade-off considerati tra le diverse metriche (es. precisione vs copertura) nell'ordinamento finale.")
    
    @field_validator('final_recommendations')
    def validate_exactly_50_items(cls, v):
        """Validatore che garantisce esattamente NUM_RECOMMENDATIONS elementi."""
        if len(v) != NUM_RECOMMENDATIONS:
            raise ValueError(f"L'array deve contenere esattamente {NUM_RECOMMENDATIONS} elementi, trovati {len(v)}")
        return v 